# train.py
"""
Fine-tunes MedGemma 1.5 4B on RSNA 2024 Lumbar Spine dataset using Unsloth.

v3.0 fixes:
  - Use UnslothVisionDataCollator (required for Unsloth vision fine-tuning)
  - collate_fn correctly handles image_paths list (multi-image per example)
  - Proper prompt masking: user-turn tokens set to -100 so the model only
    learns to predict the assistant response, not the prompt
  - Removed unused formatting_func that incorrectly applied text-only chat
    template to multi-image message dicts
"""

import os
import json
from pathlib import Path
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from PIL import Image
from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from config import cfg, set_seed

# Module-level references populated in main()
tokenizer = None
processor = None


def load_dataset_file(path: str) -> Dataset:
    """
    Load a dataset from .jsonl (one example per line) or legacy .json (a single
    list). JSONL is preferred — it lets us stream-load multi-GB dumps instead
    of holding the whole list in RAM.
    """
    rows = []
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, "r") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with open(p, "r") as f:
            rows = json.load(f)

    cleaned = []
    for entry in rows:
        for msg in entry.get("messages", []):
            # The assistant content is now compact JSON text, but the user
            # role keeps its list-of-parts structure. Wrap stray strings to
            # keep the collator's expectations satisfied.
            if isinstance(msg["content"], str):
                msg["content"] = [{"type": "text", "text": msg["content"]}]
        if "image_paths" in entry and isinstance(entry["image_paths"], str):
            entry["image_paths"] = [entry["image_paths"]]
        cleaned.append(entry)

    return Dataset.from_list(cleaned)


def _resolve_dataset_path(base_dir: Path, stem: str) -> Path:
    """Prefer .jsonl, fall back to .json for backward compat."""
    jsonl = base_dir / f"{stem}.jsonl"
    if jsonl.exists():
        return jsonl
    return base_dir / f"{stem}.json"


def convert_to_conversation(sample: dict) -> dict:
    """
    Convert a dataset sample to the format expected by UnslothVisionDataCollator.

    UnslothVisionDataCollator expects each example to have:
      - "messages": the conversation in OpenAI-style chat format
      - "images":   a list of PIL Image objects (one per <image> token in messages)

    BUG FIX v3.0: The previous code used a bare collate_fn that read
    example["image_path"] (singular). The dataset stores image_paths (list)
    because each study can have multiple series/slices. We now load ALL images
    and pass them as the "images" list. The number of {"type": "image"} entries
    in the user message content must match len(images).
    """
    image_paths = sample.get("image_paths", [])
    images = []
    for p in image_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"  [WARN] Could not load image {p}: {e}")

    # If any images failed to load, rebuild the message content to match
    # the actual number of successfully loaded images.
    messages = sample["messages"]
    if len(images) != len(image_paths):
        user_msg = next(m for m in messages if m["role"] == "user")
        assistant_msg = next(m for m in messages if m["role"] == "assistant")
        user_content = [{"type": "image"} for _ in images]
        for item in user_msg["content"]:
            if item["type"] == "text":
                user_content.append(item)
                break
        messages = [
            {"role": "user", "content": user_content},
            assistant_msg,
        ]

    return {"messages": messages, "images": images}


def batch_transform(examples: dict) -> dict:
    """
    Batch-level wrapper around `convert_to_conversation` for use with
    `Dataset.set_transform`. Unlike `Dataset.map`, `set_transform` is applied
    lazily at `__getitem__` time inside the dataloader, so PIL images are NEVER
    materialized into the pyarrow cache. This is critical: serialising PIL
    Image objects via Arrow either crashes or balloons disk usage by tens of
    GB on a real dataset.

    `examples` is a dict-of-lists (one list per column, equal length).
    Returns a dict-of-lists with the columns the collator expects.
    """
    keys = list(examples.keys())
    batch_size = len(examples[keys[0]])
    out_messages, out_images = [], []
    for i in range(batch_size):
        sample = {k: examples[k][i] for k in keys}
        converted = convert_to_conversation(sample)
        out_messages.append(converted["messages"])
        out_images.append(converted["images"])
    return {"messages": out_messages, "images": out_images}


def main():
    global tokenizer, processor

    set_seed(cfg.seed)

    print("=" * 60)
    print("MedGemma 1.5 4B Fine-tuning on RSNA 2024 Lumbar Spine")
    print("=" * 60)

    # Load model with Unsloth optimizations
    print("\n[1/5] Loading model with Unsloth...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,          # Auto-detect best dtype
        load_in_4bit=True,   # QLoRA
        token=os.environ.get("HF_TOKEN"),
    )

    # Get processor for image handling
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        cfg.model_name,
        token=os.environ.get("HF_TOKEN")
    )

    # Apply LoRA adapters. Which stacks get adapters is now driven by cfg,
    # so the rtx_4090 profile can freeze the vision encoder without editing
    # this file.
    print("\n[2/5] Applying LoRA adapters...")
    print(f"  finetune_vision_layers:    {cfg.finetune_vision_layers}")
    print(f"  finetune_language_layers:  {cfg.finetune_language_layers}")
    print(f"  finetune_attention_modules:{cfg.finetune_attention_modules}")
    print(f"  finetune_mlp_modules:      {cfg.finetune_mlp_modules}")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision_layers,
        finetune_language_layers=cfg.finetune_language_layers,
        finetune_attention_modules=cfg.finetune_attention_modules,
        finetune_mlp_modules=cfg.finetune_mlp_modules,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters:     {sum(p.numel() for p in model.parameters()):,}")

    # Load datasets (prefers .jsonl, falls back to .json)
    print("\n[3/5] Loading datasets...")
    processed = Path(cfg.processed_data_dir)
    train_dataset_raw = load_dataset_file(str(_resolve_dataset_path(processed, "train_dataset")))
    val_dataset_raw   = load_dataset_file(str(_resolve_dataset_path(processed, "val_dataset")))

    # Apply the message+image conversion lazily via set_transform. Using
    # Dataset.map() here would either crash on PIL Image arrow-serialisation
    # or write a multi-GB cache; set_transform runs per-batch at __getitem__
    # time so PIL Images live in memory only as long as the dataloader needs
    # them.
    train_dataset_raw.set_transform(batch_transform)
    val_dataset_raw.set_transform(batch_transform)
    train_dataset = train_dataset_raw
    val_dataset = val_dataset_raw

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples:   {len(val_dataset)}")

    # Training arguments
    print("\n[4/5] Setting up training...")
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # HF Transformers expects the literal string "none", not the list
        # ["none"] (which is parsed as a reporter named "none" and errors on
        # newer versions).
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="medgemma-lumbar-spine-finetune",
        dataloader_num_workers=0,   # 0 avoids multiprocessing issues with PIL images
        remove_unused_columns=False,
        seed=cfg.seed,
    )

    # BUG FIX v3.0: Use UnslothVisionDataCollator instead of a bare custom
    # collate_fn. This collator:
    #   1. Applies the processor to images + text together
    #   2. Correctly masks prompt tokens (sets user-turn labels to -100) so
    #      the model only learns to predict the assistant response
    #   3. Handles variable-length image lists per example
    data_collator = UnslothVisionDataCollator(model, tokenizer)

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=data_collator,
        dataset_text_field=None,    # We use custom collator
        max_seq_length=cfg.max_seq_length,
        packing=False,              # Must be False for vision models
    )

    # Stop training if eval loss stops improving. Pairs with
    # load_best_model_at_end + metric_for_best_model="eval_loss" above.
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # Train!
    print("\n[5/5] Starting training...")
    print(f"Output directory: {cfg.output_dir}")
    print(f"Epochs: {cfg.num_train_epochs}")
    print(f"Effective batch size: "
          f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")

    output_dir = Path(cfg.output_dir)
    if any(output_dir.glob("checkpoint-*")):
        print("Found existing checkpoint → resuming training")
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoint found → starting fresh training")
        trainer_stats = trainer.train(resume_from_checkpoint=False)

    print("\n=== Training Complete ===")
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f}s "
          f"({trainer_stats.metrics['train_runtime']/3600:.2f}h)")

    # Save adapters only. Merging into a single 16-bit checkpoint is the job
    # of merge_and_save.py; keeping it separate means a failed merge does not
    # destroy a successful training run.
    print("\nSaving LoRA adapters...")
    model.save_pretrained(cfg.output_dir + "/final")
    tokenizer.save_pretrained(cfg.output_dir + "/final")

    print(f"\nAdapters saved to: {cfg.output_dir}/final")
    print("Run `python merge_and_save.py` to produce a merged 16-bit model.")
    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()