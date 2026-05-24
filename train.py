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
import torch
from pathlib import Path
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from PIL import Image
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from config import cfg

# Module-level references populated in main()
tokenizer = None
processor = None


def load_dataset_from_json(json_path: str) -> Dataset:
    """Load and format dataset for training."""
    with open(json_path, "r") as f:
        data = json.load(f)

    cleaned_data = []
    for entry in data:
        # Standardize 'messages' content
        for msg in entry.get("messages", []):
            # If content is just a string (common for assistant roles), wrap it in a list/dict
            if isinstance(msg["content"], str):
                msg["content"] = [{"type": "text", "text": msg["content"]}]
        
        # Double check image_paths just in case
        if "image_paths" in entry and isinstance(entry["image_paths"], str):
            entry["image_paths"] = [entry["image_paths"]]
            
        cleaned_data.append(entry)
        
    return Dataset.from_list(data)


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
    # the actual number of successfully loaded images
    messages = sample["messages"]
    if len(images) != len(image_paths):
        user_content = []
        for _ in images:
            user_content.append({"type": "image"})
        # Preserve the original text prompt
        for item in messages[0]["content"]:
            if item["type"] == "text":
                user_content.append(item)
                break
        messages = [
            {"role": "user", "content": user_content},
            messages[1],  # assistant turn unchanged
        ]

    return {"messages": messages, "images": images}


def main():
    global tokenizer, processor

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

    # Apply LoRA adapters
    print("\n[2/5] Applying LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,     # Fine-tune vision encoder too
        finetune_language_layers=True,   # Fine-tune language layers
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
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

    # Load datasets
    print("\n[3/5] Loading datasets...")
    train_dataset_raw = load_dataset_from_json(
        str(Path(cfg.processed_data_dir) / "train_dataset.json")
    )
    val_dataset_raw = load_dataset_from_json(
        str(Path(cfg.processed_data_dir) / "val_dataset.json")
    )

    # BUG FIX v3.0: Convert to the format UnslothVisionDataCollator expects.
    # This loads PIL images into memory per example and builds the correct
    # messages structure. map() is lazy so images are loaded per batch.
    train_dataset = train_dataset_raw.map(convert_to_conversation, batched=False)
    val_dataset   = val_dataset_raw.map(convert_to_conversation, batched=False)

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
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
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

    # Train!
    print("\n[5/5] Starting training...")
    print(f"Output directory: {cfg.output_dir}")
    print(f"Epochs: {cfg.num_train_epochs}")
    print(f"Effective batch size: "
          f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")

    output_dir = Path(cfg.output_dir)
    if (output_dir / "checkpoint-").exists() or any(output_dir.glob("checkpoint-*")):
        print("Found existing checkpoint → resuming training")
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoint found → starting fresh training")
        trainer_stats = trainer.train(resume_from_checkpoint=False)

    print("\n=== Training Complete ===")
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f}s "
          f"({trainer_stats.metrics['train_runtime']/3600:.2f}h)")

    # Save final model
    print("\nSaving model...")
    model.save_pretrained(cfg.output_dir + "/final")
    tokenizer.save_pretrained(cfg.output_dir + "/final")

    # Save as merged model (optional, larger file but easier to deploy)
    print("Saving merged model (full weights)...")
    model.save_pretrained_merged(
        cfg.output_dir + "/merged",
        tokenizer,
        save_method="merged_16bit"  # or "lora" to save only adapters
    )

    print(f"\nModel saved to: {cfg.output_dir}")
    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()