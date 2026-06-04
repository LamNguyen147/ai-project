# Storing, Deploying, and Testing the Fine-tuned Model

End-to-end guide for what to do after training finishes so you can shut the pod down and bring the model back later for inference.

> **About merging:** This guide uses **LoRA adapters directly** rather than a merged 16-bit checkpoint. The adapters are ~200 MB vs ~8 GB merged, upload/download 40× faster, and produce bit-identical inference results when loaded via Unsloth or `transformers + peft`. Merging is only useful if you plan to deploy to frameworks that can't load adapters at runtime (vLLM, GGUF, AWQ, etc.) — none of which apply to the RunPod / Unsloth path below. See the bottom of this file for notes on merging if you need it.

## What you have after training

```
/workspace/models/medgemma-lumbar/
├── final/              ← LoRA adapters (~200 MB) ← this is what you'll deploy
└── checkpoint-*/       ← intermediate checkpoints (safe to delete)
```

The `final/` directory contains the LoRA delta weights plus an `adapter_config.json` that records which base model they were trained on top of (`unsloth/medgemma-1.5-4b-it`). When you load it with `FastVisionModel.from_pretrained(final/)`, Unsloth automatically pulls the base model and applies the adapters.

## 1. Test inference on the pod (before shutting down)

Before storing or shutting anything down, verify the adapter actually generates valid output. Save this to `/workspace/medgemma-finetune/test_inference.py`:

```python
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
    load_in_4bit=True,    # keeps memory low for inference (~6 GB peak)
)
FastVisionModel.for_inference(model)

# Grab one study from the val set — each row already has the correct
# image_paths list for that study. Globbing the PNG dir would load all
# ~15K files because PNGs are stored flat (not nested per study).
with open(VAL_JSONL) as f:
    sample = json.loads(f.readline())

image_paths   = sample["image_paths"]
series_types  = sample.get("series_types", ["sagittal_t2"])
ground_truth  = sample["messages"][-1]["content"]

images = [Image.open(p).convert("RGB") for p in image_paths]
print(f"Loaded {len(images)} images for study {sample['study_id']}")
print(f"Series types: {series_types}")

user_prompt = build_user_prompt(len(images), series_types=series_types)
messages = [{"role": "user", "content": [
    *[{"type": "image"} for _ in images],
    {"type": "text", "text": user_prompt},
]}]

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
```

Run it:
```bash
cd /workspace/medgemma-finetune
python test_inference.py
```

Expected output:
- 5 keys (`L1L2` … `L5S1`)
- Each value has `canal`, `lf`, `rf`, `ls`, `rs` → severity codes `N` / `M` / `S`

If you see a valid parsed JSON, the adapter is working. **Now you can safely shut the pod down.**

## 2. Store the adapter so the pod can be deleted

### Option A — Push to HuggingFace Hub (recommended)

Most portable. The adapter lives off-pod and can be pulled to any GPU from anywhere.

```bash
# Make sure you're logged in
huggingface-cli login   # paste your HF_TOKEN

# Create a private repo (replace YOUR_USERNAME)
huggingface-cli repo create medgemma-lumbar-finetune --type model --private

# Upload the LoRA adapters (~200 MB → fast)
huggingface-cli upload YOUR_USERNAME/medgemma-lumbar-finetune \
    /workspace/models/medgemma-lumbar/final . \
    --repo-type model
```

Upload time: ~30 sec to a few minutes for 200 MB.

After upload:
- The adapter lives at `https://huggingface.co/YOUR_USERNAME/medgemma-lumbar-finetune`
- You can **terminate** the pod (not just stop) and the model is safe.

### Option B — Keep on the RunPod network volume

If your pod uses a network volume mounted at `/workspace`, the adapter survives pod stop/terminate as long as the volume itself stays alive.

- **Stop** the pod (not delete) → pay only for volume storage (~$0.10/GB/month).
- For ~50 GB of stuff on the volume → roughly $5/month idle.
- Faster to resume than re-downloading from HF, but tied to one RunPod region.

### Option C — Both

Push to HF Hub as the canonical copy, also keep the network volume around for fast spin-up. Belt-and-suspenders. Recommended if you can afford the volume storage.

## 3. Bring the model back later

### To a fresh RunPod pod

```bash
# Launch any GPU pod with at least 24 GB VRAM
# (RTX 4090 / RTX 3090 / A40 / L40S / RTX 6000 Ada / A100 all work)
# Use the same CUDA 12.6+ image as before (see RUNPOD.md step 1)

# Install just the inference dependencies (no training stack needed)
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install "transformers>=4.56.1,!=4.57.4,!=4.57.5,<=5.5.0" \
            "peft>=0.18.0" \
            "bitsandbytes>=0.43.1" \
            "pillow"

# Pull the adapter from HF Hub
huggingface-cli login
huggingface-cli download YOUR_USERNAME/medgemma-lumbar-finetune \
    --local-dir /workspace/medgemma-lumbar-adapter

# Or, if the network volume is reattached, the adapter is already at
# /workspace/models/medgemma-lumbar/final
```

Then run inference by pointing `ADAPTER_PATH` in `test_inference.py` at the local directory (or at the HF repo id directly — Unsloth resolves both).

You also need the project files for `data_prep.build_user_prompt` and `cfg`:
```bash
cd /workspace
git clone <your-repo-url> medgemma-finetune
cd medgemma-finetune
python test_inference.py
```

### Minimum hardware for inference

Inference is much cheaper than training. With `load_in_4bit=True` the model uses ~6 GB VRAM.

| GPU | VRAM | Works? | Approx. RunPod cost |
|---|---|---|---|
| RTX 3090 | 24 GB | ✓ | ~$0.30/hr |
| RTX 4090 | 24 GB | ✓ | ~$0.40/hr |
| A40 | 48 GB | ✓ (overkill) | ~$0.50/hr |
| L40S | 48 GB | ✓ (overkill) | ~$0.90/hr |
| RTX 6000 Ada | 48 GB | ✓ (overkill) | ~$0.80/hr |
| A100 40/80 GB | 40/80 GB | ✓ (overkill) | ~$2.00/hr |

For occasional inference, a **RTX 4090 pod** is the cheapest sensible choice.

## 4. Deploy as an HTTP service

### Option A — RunPod Serverless (cheapest, pay-per-request)

Wrap inference in a `handler.py` and deploy as a RunPod Serverless endpoint:

```python
# handler.py — RunPod Serverless entry point
import base64
import io
import json

import runpod
import torch
from PIL import Image
from unsloth import FastVisionModel

from config import cfg
from data_prep import build_user_prompt

# Load once at cold start — adapter path can be local or an HF repo id
ADAPTER_PATH = "YOUR_USERNAME/medgemma-lumbar-finetune"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=cfg.max_seq_length,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)


def predict(job):
    inp = job["input"]
    images = [
        Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        for b64 in inp["images_b64"]
    ]
    series_types = inp.get("series_types", ["sagittal_t2"])

    user_prompt = build_user_prompt(len(images), series_types)
    messages = [{"role": "user", "content": [
        *[{"type": "image"} for _ in images],
        {"type": "text", "text": user_prompt},
    ]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(images, text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    response = response[response.find("{") : response.rfind("}") + 1]
    return json.loads(response)


runpod.serverless.start({"handler": predict})
```

Deploy steps:
1. Build a Docker image that includes the handler + project files.
2. Push the image and configure a RunPod Serverless endpoint pointing at it.
3. Call the endpoint over HTTPS with `{"input": {"images_b64": [...]}}`.

RunPod Serverless docs: https://docs.runpod.io/serverless/overview

Cost model: pay per inference second only. Idle = $0.

### Option B — HuggingFace Inference Endpoints

HF Inference Endpoints doesn't natively understand adapter-only repos — it expects a full model. If you want this path you'd have to either:
- Merge the adapter (see § Merging at the bottom) and push the merged model, **or**
- Use a custom handler (`handler.py`) that loads base + adapter at startup.

Custom handler approach:
1. Push the adapter to HF Hub (see step 2 Option A).
2. Open the repo page → "Deploy" → "Inference Endpoints" → "Advanced" → enable custom handler.
3. Upload a `handler.py` similar to the RunPod one above.
4. Pick a GPU instance (A10G or L4 is the cheapest GPU tier).

Cost: ~$0.60–$1.30/hr depending on GPU class, billed continuously while the endpoint is up.

### Option C — Always-on small GPU pod with FastAPI

If you need low cold-start latency and predictable cost, run a small FastAPI server on a 24 GB pod. Cheapest predictable option but pays even when idle.

## 5. Cost summary

| Scenario | Storage cost | Per-inference cost |
|---|---|---|
| Pod stopped, network volume kept | ~$5/mo for 50 GB | n/a — must restart pod to infer |
| HF Hub (private repo) | free (under 100 GB) | n/a — download to a GPU each time |
| RunPod Serverless | volume + image storage | ~$0.0002/second of GPU time used |
| HF Inference Endpoint (always-on) | free | ~$0.60–$1.30/hr continuous |

**Recommendation if you're not getting daily traffic:**

1. Push the adapter to HF Hub (free storage, fast upload).
2. **Terminate** the training pod (saves volume storage too).
3. When you need to test/use the model, spin up a 24 GB pod, pull the adapter from HF, run inference.
4. Skip "deploy as a service" until you have actual traffic to justify it.

## 6. Quick post-training checklist

Before shutting down the pod, confirm:

- [ ] `python test_inference.py` produced a parseable JSON response.
- [ ] You have a backup somewhere off-pod:
  - [ ] Pushed to HF Hub, **or**
  - [ ] Network volume confirmed (not the ephemeral container disk).
- [ ] `evaluate.py` results are saved (e.g. `/workspace/logs/evaluation_results/metrics.json`).
- [ ] `compare.py` results are saved if you want the side-by-side later.
- [ ] You noted the **profile** (`demo_a100_40g`) and **training run details** (full dataset, no oversample) somewhere — you'll want them when interpreting metrics later.
- [ ] (Optional) checkpoint-* folders deleted from `/workspace/models/medgemma-lumbar/` to free volume space.

Once all boxes are checked, the pod is safe to **terminate** (not just stop) if you've pushed to HF Hub.

---

## Appendix: Merging (only if you actually need it)

You only need a merged 16-bit checkpoint if you plan to:
- Deploy to vLLM, TGI, or other servers that don't support adapter loading
- Convert to GGUF / AWQ / GPTQ for edge deployment
- Distribute as a standalone HuggingFace model to non-PEFT users

The `merge_and_save.py` script in this repo currently fails for vision models due to an unsloth bug (`# of LoRAs = 400 does not match # of saved modules = 0`). Workaround using PEFT directly:

```python
# merge_with_peft.py
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

BASE = "unsloth/medgemma-1.5-4b-it"
ADAPTER = "/workspace/models/medgemma-lumbar/final"
OUT = "/workspace/models/medgemma-lumbar/merged"

print("Loading base in bf16...")
base = AutoModelForImageTextToText.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto"
)
print("Applying adapters...")
model = PeftModel.from_pretrained(base, ADAPTER)
print("Merging...")
merged = model.merge_and_unload()
print(f"Saving to {OUT}...")
merged.save_pretrained(OUT, safe_serialization=True)
AutoProcessor.from_pretrained(ADAPTER).save_pretrained(OUT)
print("Done.")
```

Caveat: needs ~30 GB free VRAM during the merge (base loaded in bf16, not 4-bit). Fits on an A100 80GB.
