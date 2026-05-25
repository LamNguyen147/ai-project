# Running on RunPod

End-to-end guide to set up and run this project on a RunPod GPU pod. The codebase assumes CUDA + bf16 hardware (Ampere or newer) — A100 is the reference platform.

## 1. Pick a pod

| Profile (`PROFILE=`) | Recommended pod        | VRAM   | Notes                                                            |
| -------------------- | ---------------------- | ------ | ---------------------------------------------------------------- |
| `demo_a100_40g` (default) | A100 40 GB        | 40 GB  | Smallest honest 25-label config (sag T2 + sag T1 + axial T2 at 448²). |
| `a100_80g_multimodal` | A100 80 GB            | 80 GB  | **Recommended on 80 GB.** Same modality coverage at 672², LoRA r=32, 5 epochs. |
| `a100_80g`           | A100 80 GB             | 80 GB  | 896² images, 3 slices, sag T2 only — canal stenosis specialist.  |
| `a100_40g`           | A100 40 GB             | 40 GB  | 896² but only 1 slice, 1 series — canal stenosis only.           |
| `rtx_4090`           | RTX 4090               | 24 GB  | 448², vision LoRA off — smoke tests / consumer GPU runs.         |

Suggested RunPod template settings:

- **Image:** pick the **latest RunPod PyTorch image with CUDA 12.6 or higher** (cu126 or cu128). Do **not** use a CUDA 12.1 or 12.4 image — `unsloth-zoo` requires `torchao >= 0.13.0` which requires PyTorch 2.7+, and PyTorch 2.7 wheels only ship for CUDA 12.6+. `setup.sh` auto-detects your CUDA version, skips the torch install if you already have 2.7+, and tries cu128 → cu126 → cu124 as a fallback chain.
- **Container disk:** 50 GB minimum.
- **Volume disk:** 200 GB+ recommended (RSNA DICOMs are ~35 GB, PNGs add another ~15 GB at 448², checkpoints add ~20 GB per run).
- **Volume mount path:** `/workspace` (matches the project's default `WORKSPACE`).
- **Expose HTTP / TCP:** not required unless you want Jupyter.

## 2. Get credentials ready (do this once, before launching)

You will need three secrets:

| Variable          | Where to get it                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| `HF_TOKEN`        | https://huggingface.co/settings/tokens — needs read access to `google/medgemma-1.5-4b-it`.       |
| `KAGGLE_USERNAME` | Your Kaggle username (top right of https://www.kaggle.com).                                      |
| `KAGGLE_KEY`      | https://www.kaggle.com/settings/account → "Create New Token" (downloads `kaggle.json`).         |

You must also visit the model and competition pages and **click accept** on the licenses:

1. https://huggingface.co/google/medgemma-1.5-4b-it — accept the Gemma terms.
2. https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification — accept the competition rules.

Put the secrets into the RunPod template's **Environment Variables** section before launching, or export them in the shell after launch (next step).

## 3. Connect to the pod and bootstrap

After the pod is `RUNNING`, click **Connect → Web Terminal** (or SSH in).

```bash
# Export the secrets if you didn't set them as pod env vars
export HF_TOKEN=hf_xxx
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=xxxxxxxxxxxxxxxx

# Optional but recommended — enables W&B logging during training
export WANDB_API_KEY=xxxxxxxxxxxxxxxx

cd /workspace
git clone <your-repo-url> medgemma-finetune
cd medgemma-finetune

bash setup.sh
```

`setup.sh` will:

- Refuse to run unless `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY` are set.
- Create `/workspace/{data,models,logs,medgemma-finetune}`.
- Install Unsloth (from GitHub) plus pinned `transformers`, `trl`, `datasets`, `pydicom`, `peft`, `bitsandbytes`, etc.
- Log into HuggingFace.
- Write `~/.kaggle/kaggle.json` so the Kaggle CLI works.

Expected runtime: 5–10 minutes on a fresh image.

## 4. Download the RSNA 2024 dataset

```bash
mkdir -p /workspace/data/rsna-2024-lumbar-spine
cd /workspace/data/rsna-2024-lumbar-spine

kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
apt-get update -qq && apt-get install -y -qq unzip
unzip -q rsna-2024-lumbar-spine-degenerative-classification.zip
rm rsna-2024-lumbar-spine-degenerative-classification.zip
```

The download is ~35 GB; allow 10–30 minutes depending on RunPod's network. Afterwards `/workspace/data/rsna-2024-lumbar-spine/` should contain:

```
train.csv
train_series_descriptions.csv
train_label_coordinates.csv
train_images/<study_id>/<series_id>/*.dcm
test_images/...        # not used here
sample_submission.csv  # not used here
```

If `kaggle` complains about credentials, double-check `~/.kaggle/kaggle.json` exists and is `chmod 600`, and that you have accepted the competition rules in the web UI.

## 5. Prepare the training data

```bash
cd /workspace/medgemma-finetune
python data_prep.py
```

This converts DICOM → PNG and writes `train_dataset.jsonl` / `val_dataset.jsonl` under `/workspace/data/processed/`. Expected runtime: 20–60 minutes for the full ~2,000 studies.

Useful flags:

```bash
# Smoke test on 50 studies
python data_prep.py --max_samples 50

# Counter the ~85% Normal/Mild imbalance by duplicating studies
python data_prep.py --oversample

# Allow fabricated slice mapping when train_label_coordinates.csv is missing.
# SMOKE TESTS ONLY — teaches the model anatomy it cannot see.
python data_prep.py --allow-no-coords
```

After it finishes, sanity-check the printed summary:

- **Avg images per example** ≈ 8 on `demo_a100_40g` (1 midline sag T2 + 2 parasagittal sag T1 + up to 5 axial T2). Numbers near 3 mean the axial picker is returning nothing — check that `train_label_coordinates.csv` is present.
- **Dropped (over token budget)** should be small; if it's most of the dataset, your `PROFILE` is mis-set.

## 6. Fine-tune

```bash
# Default profile (demo_a100_40g — works on 40 GB A100)
python train.py

# Or pick a profile explicitly
PROFILE=a100_80g_multimodal python train.py   # recommended on 80 GB A100
PROFILE=a100_80g python train.py              # canal-stenosis specialist (sag T2 only)
PROFILE=rtx_4090 python train.py
```

Training writes LoRA adapters to `/workspace/models/medgemma-lumbar/`. Checkpoints land in `checkpoint-*/` subdirs and the best one is exported to `final/` at the end.

Expected runtime on A100 40 GB (`demo_a100_40g`, ~2,000 studies, 3 epochs): roughly 8–14 hours. Use `--max_samples 50` in `data_prep.py` first to confirm the pipeline runs end-to-end before committing GPU hours.

If the pod restarts mid-training, just run `python train.py` again — it auto-resumes from the latest `checkpoint-*` directory.

### Keep training alive after disconnecting

Training runs as a child of your SSH/VSCode session — closing the connection kills it. Use **tmux** to detach it:

```bash
tmux new -s train    # start a named session

# inside tmux — start training with logging to file
PROFILE=a100_80g_multimodal python train.py 2>&1 | tee /workspace/logs/train.log

# detach (training keeps running): Ctrl+B then D

# reconnect later
tmux attach -t train
```

`tmux` is installed by `setup.sh`. You can also tail the log from a separate terminal without attaching:

```bash
tail -f /workspace/logs/train.log
```

### W&B logging (optional)

If `WANDB_API_KEY` is exported, training reports to W&B under project `medgemma-lumbar-spine`. Otherwise reporting is disabled.

## 7. Evaluate

```bash
# Fine-tuned model on the held-out 10% validation split
python evaluate.py --model_path /workspace/models/medgemma-lumbar/final

# Base model (no fine-tuning) for a baseline number
python evaluate.py --no-finetuned

# Quick run on 20 samples
python evaluate.py --max_samples 20 --output_dir /workspace/logs/eval_quick
```

Outputs (default `/workspace/logs/evaluation_results/`):

- `metrics.json` — overall accuracy, F1, quadratic-weighted Kappa, RSNA weighted score proxy, parse-failure rate.
- `finetuned_confusion_matrices.png` — 5×5 grid of per-condition × per-level confusion matrices.

Watch the **Parse Failure Rate** in the printed summary: if it's high (> 10%) on the fine-tuned model, the model isn't emitting valid JSON and the per-label accuracy will be inflated by the silent fallback to "Normal/Mild".

## 8. Compare base vs fine-tuned

```bash
python compare.py
```

Runs both models on the same 20 validation samples and writes side-by-side metrics, a comparison chart, and text examples to `/workspace/logs/comparison/`. This is the cleanest way to see whether fine-tuning actually moved the needle on Kappa.

## 9. Merge LoRA → full 16-bit checkpoint

Only do this once training looks good — merging is irreversible-ish (the LoRA adapters at `final/` are preserved as `final_backup/`, but the merged weights take ~8 GB).

```bash
python merge_and_save.py
```

Writes the merged model to `/workspace/models/medgemma-lumbar/merged/`. This is what you'd push to HuggingFace Hub or load for production inference.

## Persistence & cost tips

- Keep `/workspace` on a **Network Volume** rather than the ephemeral container disk — pod stops/restarts wipe the container but not the volume.
- The pod can be **stopped** (not terminated) between sessions; you pay only volume storage when stopped.
- DICOM → PNG conversion is CPU-heavy; consider running `data_prep.py` on a cheaper CPU pod and only spinning up the A100 for `train.py` if cost matters.
- Always `kill` `wandb` / `tqdm` processes before stopping a pod, otherwise the next `train.py` may inherit a stale W&B run.

## Troubleshooting

| Symptom                                                                 | Likely cause / fix                                                                                                       |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `setup.sh` aborts with `HF_TOKEN: parameter null or not set`            | Export the three required env vars first. See step 2.                                                                    |
| `OSError: You are trying to access a gated repo`                        | You haven't accepted the Gemma license on the HuggingFace model page, or `HF_TOKEN` lacks read access.                   |
| `kaggle: 403 Forbidden`                                                 | Accept the RSNA 2024 competition rules in the Kaggle UI.                                                                 |
| `data_prep.py` reports "Dropped (over token budget): N" for most rows   | Your `PROFILE` env var doesn't match `cfg.max_seq_length` (e.g. running `a100_80g` data through `rtx_4090` budget).      |
| Training OOMs on A100 40 GB                                             | You're on the wrong profile. Confirm `PROFILE=demo_a100_40g` (default) or `PROFILE=a100_40g`. Don't run `a100_80g` on 40 GB. |
| Eval shows ~85% accuracy but Kappa near 0                               | Parse-failure fallback is masking bad output. Check `parse_failure_rate` in `metrics.json`.                              |
| Avg images per example ≈ 3 on `demo_a100_40g`                           | Axial picker is returning empty. Make sure `train_label_coordinates.csv` is in the data dir and you have the latest `data_prep.py` (level-format normalization). |
| `unzip: command not found`                                              | Run `apt-get update && apt-get install -y unzip` first. `setup.sh` does this automatically on fresh pods. |
| `AttributeError: module 'torch' has no attribute 'int1'` or `register_constant` | PyTorch version is too old. `unsloth-zoo` requires torch 2.7+ via `torchao>=0.13.0`. Run `pip install "torch>=2.7.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`, then reinstall unsloth. |
| `unsloth-zoo requires transformers<=5.5.0 but you have transformers X.Y` | pip resolved too-new transformers. Run `pip install "transformers>=4.56.1,!=4.57.4,!=4.57.5,!=5.0.0,!=5.1.0,<=5.5.0"`. |
