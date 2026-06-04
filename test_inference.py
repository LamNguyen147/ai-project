# test_inference.py — minimal sanity check for the LoRA adapter
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from unsloth import FastVisionModel

from config import cfg
from data_prep import build_user_prompt

ADAPTER_PATH = "/workspace/models/medgemma-lumbar/final"
VAL_JSONL = Path(cfg.processed_data_dir) / "val_dataset.jsonl"

print(f"Loading base model + LoRA adapter from {ADAPTER_PATH}...")
# FastVisionModel.from_pretrained detects the adapter_config.json and pulls
# the base model automatically — no extra steps needed.
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=cfg.max_seq_length,
    load_in_4bit=True,  # keeps memory low for inference (~6 GB peak)
)
FastVisionModel.for_inference(model)

# Grab one study from the val set — each row already has the correct
# image_paths list for that study. Globbing the PNG dir would load all
# ~15K files because PNGs are stored flat (not nested per study).
with open(VAL_JSONL) as f:
    sample = json.loads(f.readline())

image_paths = sample["image_paths"]
series_types = sample.get("series_types", ["sagittal_t2"])
ground_truth = sample["messages"][-1]["content"]

images = [Image.open(p).convert("RGB") for p in image_paths]
print(f"Loaded {len(images)} images for study {sample['study_id']}")
print(f"Series types: {series_types}")

user_prompt = build_user_prompt(len(images), series_types=series_types)
messages = [
    {
        "role": "user",
        "content": [
            *[{"type": "image"} for _ in images],
            {"type": "text", "text": user_prompt},
        ],
    }
]

text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(images, text, return_tensors="pt").to("cuda")

print("Generating...")
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False, use_cache=True)

# Slice off the prompt tokens — decoding the full sequence picks up the
# schema example JSON from the prompt and confuses the parser.
input_len = inputs["input_ids"].shape[1]
generated = out[:, input_len:]
response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
response = response[response.find("{") : response.rfind("}") + 1]

print("\nGround truth:\n", ground_truth)
print("\nModel response:\n", response)

try:
    parsed = json.loads(response)
    print("\nParsed JSON (pretty):")
    print(json.dumps(parsed, indent=2))
except Exception as e:
    print(f"\nJSON parse failed: {e}")
    sys.exit(1)

print("\n[OK] Inference smoke test passed.")
