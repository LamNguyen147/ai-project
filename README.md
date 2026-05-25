# MedGemma Lumbar Spine Fine-tuning

Fine-tunes [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) (a vision-language model) on the [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) dataset using Unsloth + QLoRA. The model is trained to output a compact JSON object classifying five degenerative conditions at five spinal levels (L1/L2 … L5/S1) on a three-tier severity scale (Normal/Mild, Moderate, Severe).

> **Compute:** this codebase is **not intended to run on a laptop**. It expects a CUDA GPU with bf16 support. Reference platform is RunPod with an A100. See profiles below.

## Hardware profiles

| `PROFILE` env var      | GPU                | Image size | Modalities / slices       | `max_seq_length` | Vision LoRA |
| ---------------------- | ------------------ | ---------- | ------------------------- | ---------------- | ----------- |
| `demo_a100_40g` (default) | A100 40 GB         | 448²       | up to 3 series, ≤ 8 slices | 9216             | on          |
| `a100_80g_multimodal`  | A100 80 GB         | 672²       | up to 3 series, ≤ 8 slices | 19456            | on (r=32)   |
| `a100_80g`             | A100 80 GB         | 896²       | 1 series, 3 slices         | 13312            | on          |
| `a100_40g`             | A100 40 GB         | 896²       | 1 series, 1 slice          | 5120             | on          |
| `rtx_4090`             | RTX 4090 (24 GB)   | 448²       | 1 series, 1 slice          | 2048             | off         |

Pick one at launch: `PROFILE=a100_80g_multimodal python train.py`. Profile values override the defaults in `config.py`. When `PROFILE` is unset the `demo_a100_40g` profile is applied automatically — it is the smallest setup that honestly covers all 25 labels (sagittal T2 + sagittal T1 + axial T2). On an 80 GB A100 prefer `a100_80g_multimodal`: same modality coverage at 1.5× linear resolution, LoRA rank 32, 5 epochs. See `FINE_TUNING_PLAN.md` for the methodology behind these choices.

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

The schema is defined once in `config.py` (`COND_KEYS`, `LEVEL_KEYS`, `SEVERITY_CODES`) and imported by all scripts so the training prompt, eval prompt, and parser cannot drift apart. The user-turn text is built by a single helper `data_prep.build_user_prompt(n_images, series_types)`, which both `evaluate.py` and `compare.py` call so the model sees the exact same prompt at inference time as during SFT.

## Metrics

`evaluate.py` reports:

- **Accuracy / F1 (weighted) / Cohen's Kappa (QW)** — standard multiclass metrics.
- **Weighted Accuracy** — class-weighted with `[1, 2, 4]`; higher is better.
- **RSNA Weighted Score Proxy** (`rsna_weighted_score_proxy`) — the *hard-prediction proxy* for the RSNA log-loss. Because a generative LM does not emit calibrated softmax probabilities, this number is fundamentally a weighted error rate scaled by ≈ 16.1, not a true log-loss. Treat it as a proxy.
- **Parse Failure Rate** (`parse_failure_rate` / `parse_failure_count`) — fraction of predicted responses where no JSON object could be extracted. Unparseable responses silently fall back to "Normal/Mild" in the per-label parser, so without this counter a base-model run can look accidentally accurate by class prior alone. `compare.py` shows this row in the side-by-side report.

The legacy key `rsna_weighted_log_loss` is kept as an alias for backward compatibility.

## Known limitations / future work

- Generative classification is fragile. A classification head on pooled vision features would give calibrated probabilities and make the RSNA log-loss real (Option 3 in `FINE_TUNING_PLAN.md`; sketched as `train_head.py` in the roadmap, not implemented).
- The `a100_40g`, `a100_80g`, and `rtx_4090` profiles all use one series (sagittal T2 only), so on those profiles only canal stenosis is honestly assessed — foraminal and subarticular labels are guessed from the class prior. Use `demo_a100_40g` (the default) or `a100_80g_multimodal` for honest 25-label coverage.
- `--allow-no-coords` is documented but unsafe to use for real training runs.
- `train.py` uses `SFTConfig` (trl ≥ 0.16) and `processing_class=` (trl ≥ 0.15). The old `SFTTrainer(tokenizer=…, max_seq_length=…)` API no longer works with current unsloth-zoo; both are now updated.

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
