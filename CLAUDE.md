# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning [MedGemma 1.5](https://huggingface.co/google/medgemma-1.5-4b-it) (4B vision-language model) on the RSNA 2024 Lumbar Spine Degenerative Classification dataset. The model learns to emit a compact JSON object classifying 5 degenerative conditions × 5 spinal levels × 3 severity labels from MRI images.

Designed to run on a CUDA GPU (typically RunPod). All paths default to `$WORKSPACE/` (env var, default `/workspace/`).

## Commands

```bash
# Environment setup (run once on a fresh RunPod instance)
# Aborts unless HF_TOKEN / KAGGLE_USERNAME / KAGGLE_KEY are set.
bash setup.sh

# Data preparation: DICOM → PNG + JSONL instruction examples
python data_prep.py
python data_prep.py --max_samples 50           # smoke test
python data_prep.py --oversample               # duplicate Moderate ×2, Severe ×4
python data_prep.py --allow-no-coords          # fabricated slice mapping; smoke test only

# Fine-tune
PROFILE=a100_80g python train.py               # pick a profile from config.PROFILES

# Evaluation
python evaluate.py --model_path $WORKSPACE/models/medgemma-lumbar/final
python evaluate.py --max_samples 20 --output_dir $WORKSPACE/logs/eval_results

# Base vs fine-tuned comparison
python compare.py

# Merge LoRA weights into a full 16-bit model for deployment
python merge_and_save.py
```

## Architecture

### Data flow

```
RSNA DICOM files  →  data_prep.py  →  PNG (cfg.image_size²)  +  train/val_dataset.jsonl
                                             ↓
                              fine-tuning with Unsloth FastVisionModel + LoRA
                                             ↓
                     evaluate.py / compare.py  →  metrics + plots
                                             ↓
                              merge_and_save.py  →  merged 16-bit weights
```

### Module responsibilities

| File | Purpose |
|---|---|
| `config.py` | Central `Config` dataclass + `PROFILES` overlays + shared label schema (`COND_KEYS`, `LEVEL_KEYS`, `SEVERITY_CODES`) + `set_seed()` |
| `data_prep.py` | DICOM→PNG conversion, slice selection, JSONL instruction examples, token-budget filtering, optional severity oversampling |
| `train.py` | Unsloth `FastVisionModel` + TRL `SFTTrainer` with `UnslothVisionDataCollator`; lazy image loading via `Dataset.set_transform` |
| `evaluate.py` | Inference, JSON parsing of model responses, metrics (accuracy / F1 / QWK / weighted accuracy / RSNA score proxy) |
| `compare.py` | Runs base and fine-tuned model on the same samples; generates comparison charts |
| `merge_and_save.py` | Merges LoRA adapters into a full 16-bit model via Unsloth |
| `setup.sh` | pip installs, HuggingFace login, Kaggle config; refuses to run without required env vars |

### Config (`config.py`)

All hyperparameters live in `Config` (singleton `cfg`). `__post_init__` applies the profile named by the `PROFILE` env var, e.g. `PROFILE=rtx_4090`.

Available profiles (see `PROFILES` dict):
- `a100_80g` — 896², 3 slices/study, `max_seq_length=13312`, vision LoRA on
- `a100_40g` — 896², 1 slice/study, `max_seq_length=5120`, vision LoRA on
- `rtx_4090` — 448², 1 slice/study, `max_seq_length=2048`, vision LoRA off

Key invariants encoded in the defaults:
- `cfg.max_seq_length` must exceed `(image_size/14)² × n_images + ~1024 (prompt+answer+chat-template overhead)`, otherwise the assistant label gets truncated and SFT loss collapses. `data_prep.py` estimates this per example and drops oversized ones.
- `cfg.severity_labels = ["Normal/Mild", "Moderate", "Severe"]` — raw strings; `cfg.rsna_loss_weights = [1, 2, 4]`.

### Label schema (compact JSON)

Defined once in `config.py` and imported everywhere — never duplicate these mappings:

```python
COND_KEYS  = {"spinal_canal_stenosis": "canal",
              "left_neural_foraminal_narrowing":  "lf",
              "right_neural_foraminal_narrowing": "rf",
              "left_subarticular_stenosis":       "ls",
              "right_subarticular_stenosis":      "rs"}
LEVEL_KEYS     = {"l1_l2": "L1L2", ..., "l5_s1": "L5S1"}
SEVERITY_CODES = {"Normal/Mild": "N", "Moderate": "M", "Severe": "S"}
```

The assistant target produced by `data_prep.build_instruction` is a single compact JSON object, e.g.

```json
{"L1L2":{"canal":"N","lf":"N","rf":"N","ls":"N","rs":"N"}, ..., "L5S1":{...}}
```

`evaluate.parse_severity_from_response` parses model output via `json.loads` (with a balanced-brace extractor for chatty outputs), not regex. The eval prompt in `evaluate.py` / `compare.py` embeds the schema — keep it synchronised with `data_prep.SCHEMA_DESCRIPTION` or evaluation will silently diverge from training.

### Dataset files

`data_prep.py` writes JSONL (`train_dataset.jsonl` / `val_dataset.jsonl`). Loaders prefer `.jsonl` and fall back to legacy `.json` for backward compatibility. The dataset row schema is:

```python
{
  "study_id": int,
  "image_paths": list[str],
  "series_types": list[str],
  "max_severity": int,           # 0=Normal/Mild only, 1=any Moderate, 2=any Severe
  "messages": [
    {"role": "user", "content": [{"type": "image"}, ..., {"type": "text", "text": ...}]},
    {"role": "assistant", "content": "<compact JSON string>"},
  ],
}
```

### Key invariants

- `train.py` uses `Dataset.set_transform`, NOT `Dataset.map`. PIL images are loaded lazily per batch; using `.map` here would either crash on PyArrow serialisation or write a multi-GB cache.
- The Unsloth `FastVisionModel.from_pretrained` return is `(model, tokenizer)` but the "tokenizer" is actually a processor; `AutoProcessor` is loaded separately because `apply_chat_template` lives on the tokenizer half.
- Series selection priority in `data_prep.py`: `sagittal_t2` > `sagittal_t1` > `axial_t2` > unknown.
- `select_representative_slices` returns `[]` when no coordinates are available unless `allow_no_coords=True` — refusing to fabricate level→slice mappings is intentional.

## Dependencies

Installed by `setup.sh` (pinned to API-compatible bands):
- `unsloth` (from GitHub) — `FastVisionModel` for 4-bit QLoRA training/inference
- `transformers >= 4.45, < 4.50`, `trl >= 0.11, < 0.12`, `peft`, `accelerate`, `bitsandbytes`
- `pydicom`, `pillow`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python-headless`
- `kaggle`, `huggingface_hub`

Environment variables: `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY` (required); `WORKSPACE` (optional, default `/workspace`); `PROFILE` (optional, default A100 40 GB-equivalent); `WANDB_API_KEY` (optional, enables W&B reporting).
