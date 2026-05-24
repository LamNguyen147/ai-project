# merge_and_save.py
import torch
from unsloth import FastVisionModel
from config import cfg

print("Loading fine-tuned model...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=cfg.output_dir + "/final",
    max_seq_length=cfg.max_seq_length,
    load_in_4bit=True,
)

print("Saving LoRA adapters only (safe copy)...")
model.save_pretrained(cfg.output_dir + "/final_backup")
tokenizer.save_pretrained(cfg.output_dir + "/final_backup")

print("Merging and saving full model...")
model.save_pretrained_merged(
    cfg.output_dir + "/merged",
    tokenizer,
    save_method="merged_16bit"
)

print("Done!")