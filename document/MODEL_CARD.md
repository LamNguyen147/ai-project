---
license: gemma
license_link: https://ai.google.dev/gemma/terms
base_model: unsloth/medgemma-1.5-4b-it
tags:
- medical
- radiology
- lumbar-spine
- mri
- vision-language
- lora
- peft
- unsloth
- medgemma
library_name: peft
pipeline_tag: image-text-to-text
language:
- en
datasets:
- rsna-2024-lumbar-spine-degenerative-classification
metrics:
- accuracy
- f1
- cohen_kappa
---

# MedGemma 1.5 4B — Lumbar Spine Degenerative Classification (LoRA)

LoRA adapters for [unsloth/medgemma-1.5-4b-it](https://huggingface.co/unsloth/medgemma-1.5-4b-it) fine-tuned on the [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) dataset. The model takes one or more lumbar MRI slices and produces a compact JSON object classifying five degenerative conditions at five spinal levels (L1/L2 → L5/S1) on a three-tier severity scale.

> ⚠️ **Research only.** This model is a student / research project and is **not** intended for clinical decision making. It has not been reviewed, validated, or cleared by any regulatory body.

## Model details

- **Base model:** `unsloth/medgemma-1.5-4b-it` (Google MedGemma 1.5 4B-IT, a SigLIP + Gemma3 vision-language model)
- **Adapter type:** LoRA (PEFT)
- **LoRA config:** r=16, α=32, dropout=0.05, applied to `q/k/v/o_proj` + `gate/up/down_proj`, vision tower + language layers (see `finetune_*` flags in `config.py`)
- **Trainable parameters:** ≈ 1.4% of the full model
- **Precision:** 4-bit base + bf16 LoRA (QLoRA)
- **Languages:** English (medical / radiology vocabulary)
- **License:** Inherits the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). The RSNA 2024 dataset has separate terms — do not redistribute imagery or derived PNGs.

## Intended use

**In scope**
- Educational / research demos of medical vision-language fine-tuning
- A baseline for the RSNA 2024 lumbar-spine task
- Comparing structured-prediction fine-tuning vs base-model behaviour
- A worked example of LoRA on a small medical VLM

**Out of scope**
- Any clinical decision support
- Real-time triage or prioritisation
- Use on imaging modalities other than lumbar MRI
- Use on patient demographics outside those represented in RSNA 2024

## Answer schema

The model emits a single compact JSON object (~100 tokens) with five spinal-level keys, each containing five condition keys with a severity code.

```json
{
  "L1L2": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
  "L2L3": {"canal": "M", "lf": "N", "rf": "N", "ls": "N", "rs": "S"},
  "L3L4": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
  "L4L5": {"canal": "M", "lf": "M", "rf": "N", "ls": "N", "rs": "N"},
  "L5S1": {"canal": "N", "lf": "S", "rf": "M", "ls": "N", "rs": "N"}
}
```

| Severity code | Meaning |
|---|---|
| `N` | Normal/Mild |
| `M` | Moderate |
| `S` | Severe |

| Condition key | Full name |
|---|---|
| `canal` | Spinal Canal Stenosis |
| `lf` | Left Neural Foraminal Narrowing |
| `rf` | Right Neural Foraminal Narrowing |
| `ls` | Left Subarticular Stenosis |
| `rs` | Right Subarticular Stenosis |

## Training data

- **Source:** [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
- **Pre-processing:** DICOM → 448×448 PNG via percentile windowing / DICOM window-width-center (`data_prep.py`)
- **Slice selection:** Multi-modality picker — 1 midline sagittal T2 (canal stenosis), up to 2 parasagittal T1 slices (foraminal narrowing), up to 5 axial T2 slices (subarticular stenosis), one per disc level when coordinates are available
- **Studies used:** 1,975 of the ~2,000 in the public training set (others dropped for missing coordinates or token-budget overflow)
- **Split:** 90 % train / 10 % validation (≈ 1,777 / 198 studies)
- **Class distribution:** ≈ 85 % Normal/Mild, ≈ 12 % Moderate, ≈ 3 % Severe (heavy imbalance — see "Limitations")
- **Oversampling:** not used in this version

## Training procedure

- **Framework:** [Unsloth](https://github.com/unslothai/unsloth) `FastVisionModel` + TRL `SFTTrainer`
- **Profile:** `demo_a100_40g` (see `config.py`)
- **Hardware:** 1 × NVIDIA A100 80 GB
- **Image resolution:** 448 × 448
- **Max sequence length:** 9,216 tokens (≈ 8 × 1024 image tokens + ~1024 text tokens)
- **Epochs:** 3 (early stopping enabled with patience 3 on `eval_loss`)
- **Batch size:** 1 per device × 8 gradient accumulation = effective 8
- **Optimizer:** AdamW, lr = 2e-4, cosine schedule, warmup ratio 0.05, weight decay 0.01
- **Loss:** standard causal LM cross-entropy with user-turn labels masked to −100 (via `UnslothVisionDataCollator`)
- **Best-checkpoint selection:** lowest `eval_loss` across the 19 evaluations triggered by `eval_strategy="steps", eval_steps=100`

## Evaluation

Evaluation is on the 198-study held-out validation split using `evaluate.py`. The model autoregressively generates the JSON; `evaluate.parse_severity_from_response` parses it with `json.loads` and falls back to "Normal/Mild" on parse failure (tracked separately as `parse_failure_rate`).

### Overall metrics

| Metric | Base MedGemma 1.5 4B-IT | Fine-tuned |
|---|---|---|
| Cohen's κ (overall) | 0.000 | **0.502** |
| Weighted accuracy ([1,2,4] weights) | 0.586 | **0.673** |
| F1 (weighted) | 0.684 | **0.724** |
| Raw accuracy | 0.780 | 0.792 |
| RSNA weighted score proxy *(lower is better)* | 6.668 | **6.169** |
| Parse failure rate | 0.0 % | **0.0 %** |

### Per-condition Cohen's κ (fine-tuned)

| Condition | κ |
|---|---|
| Spinal canal stenosis | **0.54** |
| Right subarticular stenosis | 0.50 |
| Left subarticular stenosis | 0.44 |
| Right neural foraminal narrowing | 0.07 |
| Left neural foraminal narrowing | 0.03 |

**How to read these numbers:** raw accuracy is inflated by the ~85 % Normal/Mild class prior. Cohen's κ corrects for that. The base model always predicts "Normal/Mild" (κ = 0), while the fine-tuned model genuinely classifies non-Normal cases for canal and subarticular stenosis. Foraminal narrowing κ stays near zero because Moderate/Severe foraminal cases are too rare in the training data — oversampling is planned as future work.

### Important caveat on the RSNA weighted score

A generative LM does not emit calibrated softmax probabilities, so the published `rsna_weighted_score_proxy` is a **hard-prediction proxy** for the official RSNA log-loss, not a true log-loss. Treat it as a weighted error rate scaled by ≈ 16.1, not a competition-equivalent submission score.

## How to use

This repository contains only the LoRA adapter (~200 MB). Unsloth's `FastVisionModel.from_pretrained` will resolve and pull the base model automatically.

```python
import json
import torch
from PIL import Image
from unsloth import FastVisionModel

ADAPTER = "YOUR_USERNAME/medgemma-lumbar-finetune"   # this repo

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=ADAPTER,
    max_seq_length=9216,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

images = [Image.open(p).convert("RGB") for p in [
    "midline_sagittal_t2.png",
    "parasagittal_t1_left.png",
    "parasagittal_t1_right.png",
    "axial_t2_l4l5.png",
]]

# The EXACT prompt format the model was trained on. Any deviation degrades quality.
schema_block = (
    "Use a compact JSON object. Keys are spinal levels (L1L2..L5S1); each value "
    "is an object whose keys are conditions (canal=spinal canal stenosis, lf=left "
    "neural foraminal narrowing, rf=right neural foraminal narrowing, ls=left "
    "subarticular stenosis, rs=right subarticular stenosis) and whose values are "
    'severity codes ("N"=Normal/Mild, "M"=Moderate, "S"=Severe).'
)
view_desc = "Sagittal T2, Sagittal T1, Sagittal T1, Axial T2"
user_prompt = (
    f"You are provided with {len(images)} lumbar spine MRI image(s) ({view_desc}). "
    "Classify all degenerative conditions at each spinal level from L1/L2 to L5/S1.\n\n"
    f"{schema_block}"
)

messages = [{"role": "user", "content": [
    *[{"type": "image"} for _ in images],
    {"type": "text", "text": user_prompt},
]}]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(images, text, return_tensors="pt").to("cuda")

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False, use_cache=True)

input_len = inputs["input_ids"].shape[1]
response = tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)[0]
response = response[response.find("{") : response.rfind("}") + 1]
print(json.loads(response))
```

For a higher-fidelity prompt (the actual training-time helper), use `data_prep.build_user_prompt(n_images, series_types)` from the [training repo](https://github.com/YOUR_USERNAME/YOUR_REPO).

## Limitations & biases

- **Severe class imbalance.** ~85 % of labels are Normal/Mild; the model is reluctant to predict Moderate/Severe.
- **Foraminal narrowing under-trained.** Per-condition κ is near zero — the model effectively defaults to "Normal/Mild" for foraminal labels.
- **Generative classification is fragile.** A classification head over pooled vision features would give calibrated probabilities and a real log-loss. The JSON parsing path is robust (0 % failures on validation) but the metric is still a proxy.
- **Single-institution / single-dataset training.** RSNA 2024 covers a finite distribution of scanners, vendors, and patient demographics. Performance off-distribution is unknown and likely worse.
- **Slice-selection bias.** The training pipeline uses `train_label_coordinates.csv` to pick slices that overlap with annotated levels. At inference time the user must supply equivalent slices (one midline sag T2 + parasagittal T1s + axial T2 per disc) or the model will see information it was not trained for.
- **Not a clinical device.** Outputs are unreviewed and unvalidated.

## Future work

- Train with `--oversample` (Moderate ×2, Severe ×4) to push foraminal κ above zero.
- Retrain with the `a100_80g_multimodal` profile (672² resolution, LoRA r=32, 5 epochs).
- Replace generative classification with a small classification head over pooled vision features → calibrated probabilities and a true RSNA log-loss.
- Add a sanity-check classification head that flags out-of-distribution images.

## Citations

If you build on this work, please cite the underlying base model and dataset:

```bibtex
@misc{medgemma2025,
  title  = {MedGemma 1.5: A Vision-Language Model for Medical Image Understanding},
  author = {Google DeepMind},
  year   = {2025},
  url    = {https://huggingface.co/google/medgemma-1.5-4b-it}
}

@misc{rsna2024lumbar,
  title  = {RSNA 2024 Lumbar Spine Degenerative Classification},
  author = {Radiological Society of North America},
  year   = {2024},
  url    = {https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification}
}
```

## Acknowledgements

- Google DeepMind for the MedGemma base model.
- [Unsloth](https://github.com/unslothai/unsloth) for the QLoRA tooling and the patched MedGemma weights.
- RSNA + Kaggle for releasing the labelled lumbar-spine dataset.
