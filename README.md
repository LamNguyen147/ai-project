# MedGemma Lumbar Spine Fine-tuning

Fine-tunes [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) (a vision-language model) on the [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) dataset using Unsloth + QLoRA. The model is trained to output a compact JSON object classifying five degenerative conditions at five spinal levels (L1/L2 … L5/S1) on a three-tier severity scale (Normal/Mild, Moderate, Severe).

> **Compute:** this codebase is **not intended to run on a laptop**. It expects a CUDA GPU with bf16 support. Reference platform is RunPod with an A100. See profiles below.

## Hardware profiles

| `PROFILE` env var | GPU                     | Image size | Slices/study | `max_seq_length` | Vision LoRA |
| ----------------- | ----------------------- | ---------- | ------------ | ---------------- | ----------- |
| (default)         | A100 40 GB / equivalent | 896²       | 1            | 5120             | on          |
| `a100_80g`        | A100 80 GB              | 896²       | 3            | 13312            | on          |
| `a100_40g`        | A100 40 GB              | 896²       | 1            | 5120             | on          |
| `rtx_4090`        | RTX 4090 (24 GB)        | 448²       | 1            | 2048             | off         |

Pick one at launch: `PROFILE=a100_80g python train.py`. Profile values override the defaults in `config.py`.

## Setup

```bash
export HF_TOKEN=...
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
bash setup.sh
```

`setup.sh` aborts if those env vars are unset. It installs Unsloth (from GitHub) plus pinned versions of `transformers`, `trl`, `datasets`, `pydicom`, etc.

Download the RSNA 2024 dataset into `$WORKSPACE/data/rsna-2024-lumbar-spine/` (`train.csv`, `train_series_descriptions.csv`, `train_label_coordinates.csv`, `train_images/<study_id>/<series_id>/*.dcm`).

## Pipeline

```bash
# 1. DICOM → PNG + JSONL instruction examples
python data_prep.py [--max_samples N] [--oversample]

# 2. Fine-tune (writes LoRA adapters to $output_dir/final)
PROFILE=a100_80g python train.py

# 3. Evaluate on the held-out 10% split
python evaluate.py --model_path $WORKSPACE/models/medgemma-lumbar/final

# 4. Side-by-side base vs fine-tuned comparison
python compare.py

# 5. Merge LoRA into a single 16-bit checkpoint for deployment
python merge_and_save.py
```

`data_prep.py` flags worth knowing:

- `--max_samples N` — limit number of studies (smoke test).
- `--oversample` — duplicate Moderate (×2) and Severe (×4) studies in train split to counter the natural ~85% Normal/Mild imbalance.
- `--allow-no-coords` — opt in to evenly-spaced slice fallback when `train_label_coordinates.csv` is missing. **Smoke tests only**; the level → slice mapping is fabricated.

## Answer schema

`data_prep.py` writes one user → assistant turn per study. The assistant target is a compact JSON object:

```json
{
  "L1L2": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
  "L2L3": {"canal": "M", "lf": "N", "rf": "N", "ls": "N", "rs": "S"},
  "L3L4": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
  "L4L5": {"canal": "M", "lf": "M", "rf": "N", "ls": "N", "rs": "N"},
  "L5S1": {"canal": "N", "lf": "S", "rf": "M", "ls": "N", "rs": "N"}
}
```

| Code | Meaning              |
| ---- | -------------------- |
| `N`  | Normal/Mild          |
| `M`  | Moderate             |
| `S`  | Severe               |

| Condition key | Full name                          |
| ------------- | ---------------------------------- |
| `canal`       | Spinal Canal Stenosis              |
| `lf`          | Left Neural Foraminal Narrowing    |
| `rf`          | Right Neural Foraminal Narrowing   |
| `ls`          | Left Subarticular Stenosis         |
| `rs`          | Right Subarticular Stenosis        |

The schema is defined once in `config.py` (`COND_KEYS`, `LEVEL_KEYS`, `SEVERITY_CODES`) and imported by all scripts so the training prompt, eval prompt, and parser cannot drift apart.

## Metrics

`evaluate.py` reports:

- **Accuracy / F1 (weighted) / Cohen's Kappa (QW)** — standard multiclass metrics.
- **Weighted Accuracy** — class-weighted with `[1, 2, 4]`; higher is better.
- **RSNA Weighted Score Proxy** (`rsna_weighted_score_proxy`) — the *hard-prediction proxy* for the RSNA log-loss. Because a generative LM does not emit calibrated softmax probabilities, this number is fundamentally a weighted error rate scaled by ≈ 16.1, not a true log-loss. Treat it as a proxy.

The legacy key `rsna_weighted_log_loss` is kept as an alias for backward compatibility.

## Known limitations / future work

- Generative classification is fragile. A classification head on pooled vision features would give calibrated probabilities and make the RSNA log-loss real (currently sketched as `train_head.py` in the roadmap, not implemented).
- Default profile uses 1 slice/study, so the model is asked to predict 25 outputs from a single 2D image. The `a100_80g` profile bumps this to 3.
- `--allow-no-coords` is documented but unsafe to use for real training runs.

## Repository layout

```
config.py          Single Config dataclass + hardware PROFILES + label schema
setup.sh           One-shot dependency installer (RunPod)
data_prep.py       DICOM → PNG + JSONL instruction examples
train.py           Unsloth + TRL SFTTrainer fine-tuning
evaluate.py        Inference, JSON parsing, metrics
compare.py         Base vs fine-tuned side-by-side
merge_and_save.py  LoRA → 16-bit merged checkpoint
```

## License

Code: MIT (see `LICENSE`). The RSNA 2024 dataset has its own competition terms — do not redistribute DICOMs or derived PNGs.
