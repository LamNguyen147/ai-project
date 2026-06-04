# Fine-tuning Plan & Methodology Notes

Reference document for the central methodological issue in this pipeline and the options for fixing it. Companion to `README.md` (how to run things) and `CLAUDE.md` (code map).

## TL;DR

- The current pipeline asks the model for 25 labels per study but feeds it ~1 slice. The model cannot actually see most of what it's asked about, so ~20 of those 25 predictions are learned from the class prior, not from the image. Metrics that don't account for this look fine.
- Four ways to fix this. **Option 1A — cover all anatomy with 8 slices at 448² — is the recommended path** for "get a working honest fine-tuned model" with the smallest deviation from the current code.
- Demo run on 800–1000 studies: **A100 40 GB community cloud, ~1–2 h wall-clock, ~$2.50 per run, ~$20–30 total project budget** (incl. debug iterations).
- A real Kaggle-competitive model is a different architecture (Option 3, classification heads on frozen vision features), not vanilla VLM SFT.

---

## 1. The label-coverage problem

Lumbar anatomy is 3D and the five spinal levels span ~10–15 cm. No single 2D MRI slice contains all 25 labels' worth of evidence. Each view shows certain things and not others:

| View                             | Shows well                                                  | Does **not** show                       |
| -------------------------------- | ----------------------------------------------------------- | --------------------------------------- |
| Midline sagittal T2              | Central canal stenosis at all 5 levels (cord, CSF, disc)    | Foramina (off-midline), subarticular    |
| Para-sagittal T1 (left or right) | Foraminal narrowing on that side at all 5 levels            | Other side, canal, subarticular         |
| Axial T2 at level X              | Canal + both foramina + both subarticular **at level X**    | Anything at the other 4 levels          |

If the pipeline picks a single sagittal T2 slice, the model can *legitimately* assess up to:

- `spinal_canal_stenosis_*` at the levels visible in that slice (≤ 5 labels)

The other 20 labels — both foraminal × 5 levels, both subarticular × 5 levels — are produced by the model **guessing the prior**. Since the natural RSNA distribution is ~85% Normal/Mild, the model converges to "say Normal/Mild for anything I can't see". The SFT loss decreases monotonically, accuracy looks good, the model is clinically useless.

### Why the current code produces this

Two contributing choices in `data_prep.py`:

1. `SERIES_PRIORITY = ["sagittal_t2", "sagittal_t1", "axial_t2"]` combined with `max_series_per_study=1` means **only sagittal T2 is ever used** — axial T2 is thrown away. Subarticular stenosis is defined on axial views, so we ask the model to predict something we never showed it.
2. `slices_per_series=1` (default) or `=3` (a100_80g profile) from a single series — we never feed left + midline + right para-sagittals, so foraminal labels on at least one side are always guesses.

The token budget (`max_seq_length=5120` for 1 image at 896²) was the original excuse. It is a workable constraint, not a clinical one — see Option 1A.

---

## 2. Fix options at a glance

| Option                                                | Honest 25-label assessment? | Real probabilities? | Deviation from current code | Recommended for                       |
| ----------------------------------------------------- | --------------------------- | ------------------- | --------------------------- | ------------------------------------- |
| Status quo (1 image × 896²)                           | No (~20/25 hallucinated)    | No                  | —                           | Nothing                               |
| **1A — 8 slices × 448², multi-modality**              | Yes                         | No                  | Small (~½ day eng)          | **"Get a working model honestly"**    |
| 1B — 8 slices × 896², multi-modality                  | Yes                         | No                  | Small                       | When you have H100 budget to burn     |
| 2 — Per-finding ROI examples                          | Yes                         | No                  | Large (~1–2 d eng)          | Higher accuracy, willing to refactor  |
| **3 — Frozen encoder + 25 classification heads**      | Yes                         | **Yes**             | Large (~1–2 d eng)          | **Kaggle-competitive, real metrics**  |

---

## 3. Option detail

### Option 1A — Cover all anatomy with 8 slices × 448² (recommended for demo)

Pick slices the way a radiologist scrolls — one per piece of anatomy that has its own label:

| Need to assess          | Slice                                            |
| ----------------------- | ------------------------------------------------ |
| Canal × 5 levels        | 1 midline sagittal T2                             |
| Left foramen × 5        | 1 left para-sagittal T1                          |
| Right foramen × 5       | 1 right para-sagittal T1                         |
| Subarticular × 2 × 5    | 5 axial T2 (one per disc level)                  |

**8 slices total → covers all 25 labels honestly.**

Token cost: 8 × (448/14)² + ~400 text ≈ **8 600 tokens**. Fits inside `max_seq_length=9216` on a 40 GB A100. SigLIP 448² loses some fine detail vs. 896² but is adequate for lumbar pathology — the dominant error in this project is the 25-label hallucination problem, not pixel resolution.

Code-wise this is small: ~½ day of work. Need to:

1. Add a profile to `config.py` (see §6).
2. Extend `data_prep.select_representative_slices` (or wrap it) so the picker can:
   - request a series of a specific modality, not just first-by-priority;
   - return left/midline/right para-sagittals from a sagittal series using slice index;
   - return one axial slice per level using `train_label_coordinates.csv`'s `instance_number` for each level's annotation.
3. Loosen `max_series_per_study` and adjust the per-study loop to gather slices across sagittal T1 + sagittal T2 + axial T2.

### Option 1B — 8 slices × 896²

Same idea as 1A but at native resolution. Tokens/example ≈ **33 000**. Requires `max_seq_length≈33792` and an H100 or A100 80 GB. ~4× more expensive than 1A in compute. The marginal quality gain over 1A is small relative to the cost; **not recommended** unless you've already squeezed everything out of 1A and want to see whether 896² is the bottleneck.

### Option 2 — Per-finding ROI examples

Use `(x, y, instance_number)` from `train_label_coordinates.csv` to **crop a ROI around each annotated finding** and ask the model to grade just that one thing. Reformulates the task:

- One training example per (condition, level) finding instead of one per study.
- ~50 000 examples instead of ~1 800.
- Output is a single severity (`"N"`/`"M"`/`"S"`) instead of a 25-key JSON.

Per-example cost is tiny (~420 tokens), so total training time is dominated by step count, not seq length. Compute is cheap (~$4–5/run on A100 40 GB).

Downside: substantial code change (new ROI cropper, new per-finding data builder, new aggregation at eval to assemble the final 25-vector per study), and inference becomes 25× more forward passes per study. Worthwhile if you want higher accuracy and are willing to refactor the data path.

### Option 3 — Frozen MedGemma encoder + 25 classification heads

Use MedGemma's SigLIP encoder as a feature extractor (frozen), then train 25 small heads (or a single 25 × 3-way head) on pooled features from a stack of slices.

- **Real calibrated probabilities** for free — `rsna_weighted_log_loss` becomes an actual probabilistic log-loss instead of the hard-prediction proxy it is today.
- Trainable parameters are tiny (~10 MB), so training the heads is essentially free.
- One-time feature extraction at ~$1 of compute, then thousands of head-training iterations for cents each.
- This is the architecture Kaggle leaderboard winners use. The leaderboard is not won by generative VLMs.

Downside: ~1–2 days of engineering (extraction script, head architecture — probably attention-pool over slice features, class-weighted CE loss, separate inference path). Phase 3.1 of the repair plan, currently not implemented (`train_head.py` exists only as a TODO note).

---

## 4. Recommended path

| Goal                                                       | Pick      |
| ---------------------------------------------------------- | --------- |
| Demonstrate fine-tuning works honestly, ASAP, minimal eng  | **1A**    |
| Squeeze more accuracy from the VLM-SFT path                | 2         |
| Win on RSNA leaderboard / real probabilities required      | 3         |
| 1A wasn't enough and resolution looks like the bottleneck  | 1B        |

For the current state of this project — pipeline works end-to-end but produces dishonest metrics on the existing 1-slice setup — **Option 1A is the right next step**.

---

## 5. Compute estimates

All prices: RunPod community-cloud (lower than secure-cloud; assumes you tolerate occasional spot pre-emption).

### 5.1 Full-scale runs (~1800 studies, 3 epochs)

| Option                | Per-run hours | Per-run cost     | Realistic project total* | Engineering time |
| --------------------- | ------------- | ---------------- | ------------------------ | ---------------- |
| Status quo (1×896²)   | ~1            | ~$1.20           | ~$10                     | 0                |
| **1A (8×448²)**       | ~2.5          | **~$3–4**        | **~$25–35**              | ~½ day           |
| 1B (8×896²)           | ~10 (A100-80G) | ~$19              | ~$100–150                | ~½ day           |
| 2 (per-finding)       | ~3.5          | ~$4–5            | ~$30–50                  | ~1–2 days        |
| **3 (heads)**         | ~1–2          | **~$1–2**        | **~$10–20**              | ~1–2 days        |

\* Project total assumes ~5–10× per-run cost across debug iterations, hyperparameter sweeps, and final run.

### 5.2 Demo-scale run for Option 1A (800–1000 studies)

| Stage                              | 800 studies     | 1000 studies    |
| ---------------------------------- | --------------- | --------------- |
| Model load + LoRA init             | ~5 min          | ~5 min          |
| Step time (~5 s/step at seq=9216)  | ~25 min         | ~30 min         |
| Eval (every 100 steps)             | ~5–10 min       | ~5–10 min       |
| Checkpoint saves                   | ~5 min          | ~5 min          |
| **Total wall-clock**               | **~45–60 min**  | **~60–90 min**  |

Add `--oversample` (recommended given class imbalance) → +30–50%, so realistically **~1–2 h per run**.

Early stopping (`EarlyStoppingCallback(patience=3)` from Phase 1) often cuts training short by ~30 min on small datasets.

### 5.3 GPU choice for the demo

| GPU                              | $/hr  | Wall-clock | Per-run cost | Notes                                                                  |
| -------------------------------- | ----- | ---------- | ------------ | ---------------------------------------------------------------------- |
| **A100 40 GB (recommended)**     | 1.19  | ~1.5–2 h   | **~$2.50**   | Default pick — bf16 native, headroom for vision LoRA                   |
| A100 80 GB                       | 1.89  | ~1.5 h     | ~$3          | Faster but only marginally; small dataset doesn't saturate the card    |
| L40S 48 GB                       | 0.99  | ~2–2.5 h   | ~$2.50       | Similar to A100 40 GB; sometimes more available                        |
| A40 48 GB                        | 0.39  | ~3.5–4 h   | **~$1.50**   | Cheapest viable — older Ampere, slower steps but works fine            |
| RTX 4090 24 GB                   | 0.69  | ~3 h       | ~$2          | Tight on VRAM — vision LoRA must stay off, OOM risk at seq=9216        |

### 5.4 Demo-scale project budget

| Item                                   | Count | Subtotal     |
| -------------------------------------- | ----- | ------------ |
| One-time DICOM → PNG conversion         | 1     | ~$0.05       |
| Debug runs (OOM, hyperparams, bugs)    | 3–5   | ~$10–15      |
| Hyperparameter tweaks (lr, r, epochs)  | 2–4   | ~$5–10       |
| Final production run                   | 1     | ~$3          |
| **Total**                              |       | **~$20–30**  |

Plus ~½ day of engineering for the profile + slice-picker extension.

### 5.5 Uncertainty bands

These estimates have **±50% uncertainty bands**. Real numbers depend on:

- Spot pre-emption (community cloud) vs. on-demand pricing
- HuggingFace download speed for the base model (~9 GB)
- How many slices actually pass the token-budget guard in `data_prep.py`
- Whether early stopping fires
- Step time variance from FlashAttention version / `torch.compile` status

Confirm against your first real run on RunPod before committing budget for a larger sweep.

---

## 6. What to add in code for Option 1A demo

### 6.1 Profile definition

Add to `PROFILES` in `config.py`:

```python
"demo_a100_40g": {
    "max_seq_length": 9216,
    "image_size": 448,
    "finetune_vision_layers": True,
    "slices_per_series": 3,        # see §6.2 note
    "max_series_per_study": 3,     # sagittal T2 + sagittal T1 + axial T2
    "per_device_train_batch_size": 1,
    "bf16": True,
},
```

Run with `PROFILE=demo_a100_40g`.

### 6.2 Slice picker (the real work)

The existing `data_prep.select_representative_slices` picks N slices spread across the annotated instance numbers of a single series. For honest coverage we need a multi-modality picker:

1. From the **sagittal T2** series: midline slice only (use the median instance number across annotations as a proxy for midline).
2. From the **sagittal T1** series: left and right para-sagittals (use 1st and 3rd quartile of annotated instances).
3. From the **axial T2** series: one slice per level using the per-level `instance_number` in `train_label_coordinates.csv`.

That's the strictly correct version. A cheaper substitute that still beats the current code: just set `max_series_per_study=3` so all three modalities feed the model, and `slices_per_series=3` so each modality contributes ≥3 slices. Total = 9 images. Token cost stays under 9216. Less anatomically precise than the bespoke picker but ~2 h of work instead of ~½ day.

### 6.3 Commands (after profile + picker)

```bash
# One-time data prep
PROFILE=demo_a100_40g python data_prep.py --max_samples 1000 --oversample

# Training
PROFILE=demo_a100_40g python train.py

# Evaluation
PROFILE=demo_a100_40g python evaluate.py \
    --model_path $WORKSPACE/models/medgemma-lumbar/final
```

---

## 7. Open items / future work

- **Option 3 (`train_head.py`)** has been on the roadmap since the Phase 3.1 plan but is not implemented. If real probabilities ever matter, this is the next architectural step after 1A is validated.
- **Token-budget estimator** in `data_prep.py` uses a conservative char→token ratio (3 chars/token). Once you have the real processor handy on RunPod, swap it for the actual tokenizer to avoid dropping borderline examples.
- **Multi-modality slice picker** is the engineering work blocking Option 1A. Worth ~½ day to do properly.
- **Honest probability extraction** — even without going to Option 3, the LM's softmax over the severity tokens (`N`/`M`/`S`) at the right position can be read directly and used as a calibrated probability for `rsna_weighted_log_loss`. Would convert the metric from "proxy" to "real" without architectural change.

---

## 8. Cross-references

- `README.md` — user-facing setup & commands
- `CLAUDE.md` — code map and invariants for future Claude Code sessions
- `config.py` — hardware profiles (`PROFILES`) and label schema
- `data_prep.py` — slice picker, JSON answer template, token-budget filter
