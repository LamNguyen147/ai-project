# Fine-tuning a Medical Vision-Language Model for Structured Lumbar Spine Degeneration Classification

*A QLoRA-based fine-tune of MedGemma 1.5 4B on the RSNA 2024 Lumbar Spine Degenerative Classification dataset.*

---

## Abstract

Diagnosing degenerative changes of the lumbar spine on MRI requires a radiologist to assign 25 separate severity labels per study — five distinct conditions evaluated at five vertebral levels. Manually labelling each examination is slow and a known bottleneck in workflows that depend on standardised reporting. We investigate whether a medical vision-language model can be fine-tuned to produce this structured output directly from MRI images. Specifically, we fine-tune **MedGemma 1.5 4B** with QLoRA (4-bit base + bf16 LoRA adapters) on the **RSNA 2024 Lumbar Spine Degenerative Classification** dataset. The model is trained to emit a single compact JSON object containing severity codes (N=Normal/Mild, M=Moderate, S=Severe) for each of the 25 outcomes. Two technical choices make the configuration honest: (i) a *multi-modality slice picker* selects one midline sagittal T2, up to two parasagittal T1 slices, and up to five axial T2 slices per study, so all 25 labels are covered by the imaging actually passed to the model; (ii) the answer is compressed from prose to short JSON (~100 tokens), which keeps the supervised loss focused on label-bearing tokens. On a held-out 198-study validation set, overall Cohen's kappa improves from **0.000** (base model) to **0.502** (fine-tuned) with a **0 % JSON parse-failure rate**. The most clinically central condition — spinal canal stenosis — reaches kappa **0.54**; left and right subarticular stenosis reach **0.44** and **0.50** respectively. Foraminal narrowing remains near zero (kappa < 0.07) because Moderate/Severe foraminal examples are rare in the training data and the run did not use class-rebalancing. We discuss this limitation and outline oversampling and a classification-head variant as the most promising next experiments.

---

## 1. Introduction

### 1.1 Clinical context

Low back pain affects a large fraction of adults at some point in their lives, and lumbar MRI is the standard imaging investigation when degenerative disease is suspected. Radiologists evaluate each of the five intervertebral disc levels — L1/L2, L2/L3, L3/L4, L4/L5, and L5/S1 — for a small set of recurring pathologies: narrowing of the spinal canal, narrowing of the neural foramina on each side, and narrowing of the subarticular recess on each side. Each of these is graded on a coarse severity scale, most commonly *Normal/Mild*, *Moderate*, and *Severe*. The output of a single study is therefore a 25-cell grid of categorical labels. Filling this grid is repetitive but clinically important: surgical planning, conservative-management decisions, and standardised reporting all depend on it.

### 1.2 Why a vision-language model

Classical computer-vision pipelines for this task typically combine a CNN backbone with multiple classification heads. Such models are accurate but inflexible: they produce labels and nothing else. A vision-language model can in principle do more — generate a free-text impression, produce structured output, or chat about the case — using the same backbone. In particular, **MedGemma 1.5**, Google's medical adaptation of the Gemma 3 family, ships as a 4 B-parameter SigLIP+Gemma3 multimodal model already exposed to medical text and imagery during pretraining. This makes it a natural starting point for medical fine-tuning.

The question this report addresses is concrete: *can a parameter-efficient fine-tune of MedGemma reliably emit the 25-label structured diagnosis directly from MRI images, on a real, imbalanced public dataset?*

### 1.3 Contributions

This work contributes the following:

1. An end-to-end **QLoRA fine-tuning pipeline** for MedGemma 1.5 4B on lumbar MRI, including DICOM preprocessing, slice selection, prompt construction, training, and evaluation.
2. A **multi-modality slice-selection scheme** that keeps the token budget within a single 40–80 GB A100 while ensuring all 25 labels are evaluable from the imagery the model actually sees.
3. A **compact-JSON output schema** that reduces the assistant-side token count by ≈ 5× compared with prose answers, making per-label gradients dominate the supervised loss.
4. A **rigorous evaluation protocol** that separates raw accuracy, weighted accuracy, and Cohen's kappa, and tracks JSON parse failures explicitly to prevent silent fallback-induced inflation of accuracy.
5. An **honest empirical comparison** of base vs fine-tuned models on a held-out 198-study split, showing overall kappa moving from 0.00 to 0.50 and identifying class imbalance as the dominant remaining failure mode.

The rest of this report is organised as follows: §2 builds the background knowledge needed for the rest of the report; §3 describes the dataset; §4 details the method; §5 reports the experimental setup; §6 presents quantitative and qualitative results; §7 discusses what worked and what did not; §8 lists limitations; §9 outlines future work; §10 concludes.

---

## 2. Background knowledge

This section briefly introduces the concepts and techniques used in the rest of the report. It is intentionally short — the goal is to make the work reproducible and the design choices intelligible to a reader who is comfortable with one domain (medical imaging *or* deep learning) but not necessarily both.

### 2.1 Lumbar spine anatomy and degenerative pathology

The lumbar spine consists of five vertebrae (L1–L5) and the sacrum (S1), separated by intervertebral discs. The five disc levels — written L1/L2 through L5/S1 — are the standard reporting units. Three structures at each level can degenerate and impinge on neural tissue:

- **Spinal canal.** The central canal carries the cauda equina. Narrowing (stenosis) can compress these nerve roots; the L4/L5 level is the most common site.
- **Neural foramen.** The lateral openings through which a single nerve root exits. Left and right are evaluated separately. Foraminal narrowing is best seen on sagittal T1 imaging.
- **Subarticular recess.** A small lateral pocket within the canal where a nerve root descends before exiting the next foramen. Stenosis here is best seen on axial T2 imaging. Left and right are evaluated separately.

For each of the five levels and each of these five structures (canal, left foraminal, right foraminal, left subarticular, right subarticular), the severity is graded coarsely: *Normal/Mild*, *Moderate*, *Severe*. This gives 5 × 5 = 25 labels per study.

### 2.2 MRI imaging of the lumbar spine

Lumbar MRI studies are multi-sequence and multi-plane. The three series types used in this work are:

- **Sagittal T2.** A side-view sequence in which cerebrospinal fluid appears bright. Best for evaluating the central canal because the dark cord and roots sit inside a bright CSF column.
- **Sagittal T1.** Same side view, but fat appears bright and CSF dark. Best for evaluating the neural foramina, because the bright epidural fat outlines the dark exiting nerve root.
- **Axial T2.** A cross-sectional view at a specific disc level. Best for evaluating the subarticular recesses and left/right asymmetry.

A single study typically contains all three series. Each series consists of multiple parallel slices through the anatomy. Pixel values in raw DICOM files are not normalised across scanners; conversion to a viewable image requires either using the DICOM *window centre / window width* metadata or applying a robust percentile normalisation.

### 2.3 Vision-language models and MedGemma 1.5

A modern vision-language model (VLM) couples a vision encoder with a language model:

1. The **vision encoder** converts an image into a sequence of patch embeddings. MedGemma uses **SigLIP** (Sigmoid Loss for Image-Text Pretraining), a CLIP variant that processes input at a patch size of 14 pixels.
2. A **projection layer** maps these patch embeddings into the language model's embedding space.
3. The **language model** — Gemma 3, a decoder-only transformer — receives the projected vision tokens interleaved with text tokens and autoregressively generates a response.

At an image resolution of $S \times S$, the encoder produces $(S/14)^2$ vision tokens per image. At $S=448$ this is $1024$ tokens; at $S=896$ it is $4096$. With $N$ images per example, the **vision token cost** is therefore $N \cdot (S/14)^2$, which dominates the sequence-length budget.

**MedGemma 1.5 4B-IT** is Google's instruction-tuned medical adaptation. It is already familiar with radiology vocabulary and standard report structure, but it is not pre-trained on the RSNA labelling schema or on the 25-label JSON output that this task requires. Fine-tuning is therefore needed.

### 2.4 Parameter-efficient fine-tuning (LoRA and QLoRA)

Full fine-tuning of a 4 B-parameter model is feasible but expensive: it requires storing optimiser states (Adam keeps $2 \times$ parameter counts in momentum buffers) and full bf16 weights, easily exceeding 60 GB of VRAM. **Low-Rank Adaptation (LoRA)** addresses this by freezing the base weights $W \in \mathbb{R}^{d \times d}$ and learning a low-rank update

$$
\Delta W = B A, \qquad A \in \mathbb{R}^{r \times d}, \; B \in \mathbb{R}^{d \times r}, \; r \ll d.
$$

Only $A$ and $B$ are trained. The effective weight at inference is $W + \alpha \Delta W$ for some scalar $\alpha$. **QLoRA** combines this with 4-bit quantisation of the frozen base: the base model is loaded in 4 bits and the LoRA matrices in bf16 (or fp16). Forward and backward passes through the base use de-quantised values on the fly; only the LoRA matrices accumulate gradients. This brings the total VRAM cost down to roughly the size of the 4-bit base plus the small LoRA buffers — practical on a single A100.

### 2.5 The evaluation metrics

This work uses several metrics; reporting them in isolation can be misleading.

- **Raw accuracy.** Fraction of correctly predicted labels. On a heavily imbalanced dataset (≈ 85 % Normal/Mild) a trivial classifier that always predicts the majority class already achieves ≈ 85 % — accuracy alone is therefore uninformative.
- **F1 (weighted).** Per-class F1 averaged with weights proportional to class support. Less misleading than accuracy but still partly inflated by the dominant class.
- **Cohen's kappa, $\kappa$.** Accuracy *corrected for chance agreement*:
  $$
  \kappa = \frac{p_o - p_e}{1 - p_e},
  $$
  where $p_o$ is observed agreement and $p_e$ is the agreement expected by chance under the marginal class distributions. $\kappa = 0$ means no better than chance; $\kappa = 1$ means perfect agreement. *This is the headline metric in this report*: it is robust to class imbalance and is the canonical measure for agreement studies in radiology.
- **Weighted accuracy.** Per-class accuracy weighted by $[1, 2, 4]$ for $[N, M, S]$. Penalises errors on the rarer, more clinically serious classes more heavily. Conceptually similar to the RSNA challenge's official weighted log-loss, but defined over hard predictions.
- **RSNA weighted score proxy.** A scalar surrogate for the challenge's official weighted log-loss, computed by reusing the same class weights $[1, 2, 4]$. Because a generative model does not emit calibrated softmax probabilities, this is a *hard-prediction* proxy, not a true log-loss; it is a weighted error rate, scaled by approximately 16.1. It is therefore *not* directly comparable to the RSNA public leaderboard.
- **Parse failure rate.** Fraction of model responses where no JSON object can be extracted. Tracked separately because the parser falls back to "Normal/Mild" on failure, which would otherwise silently boost accuracy on an imbalanced dataset.

---

## 3. Dataset

### 3.1 Source

The data come from the public **RSNA 2024 Lumbar Spine Degenerative Classification** competition, distributed via Kaggle. The training portion consists of approximately 2,000 lumbar MRI studies; each study contains multiple series (typically sagittal T1, sagittal T2, axial T2) stored as raw DICOM files together with structured CSV files:

- `train.csv` — one row per study × condition × level, giving the radiologist-graded severity label (target);
- `train_series_descriptions.csv` — series-type metadata (sagittal vs axial, T1 vs T2);
- `train_label_coordinates.csv` — per-condition (x, y, instance number) annotations identifying which slice and pixel best illustrates each label.

The competition test set is reserved by Kaggle and not used here; we hold out 10 % of the training set as our own validation split.

### 3.2 Preprocessing pipeline

For each study, the pipeline converts the relevant DICOMs to 448 × 448 PNGs. Window centre / window width metadata are used when available; otherwise robust percentile-based normalisation (0.5 / 99.5 percentiles) is applied. `MONOCHROME1` photometric interpretation is detected and inverted to match the standard "bright = high intensity" convention.

The script writes:

- `train_dataset.jsonl` — one JSON example per study for training,
- `val_dataset.jsonl` — same schema, held-out 10 %,
- a flat directory of PNGs at the chosen resolution.

Each JSON example carries the study id, the list of selected PNG paths, the list of corresponding series types, the worst-case severity across all labels (used later for class-aware oversampling), and a two-turn conversation (user instruction → compact-JSON assistant answer).

### 3.3 Volume and split

| Quantity | Count |
|---|---|
| Total RSNA training studies used | 1,974 |
| Studies dropped (missing coordinates / over token budget) | ≈ 26 |
| Train split (90 %) | **1,776** |
| Validation split (10 %) | **198** |
| Total label predictions per evaluation | 198 × 25 = **4,950** |

The 10 % validation split is fixed at study level, so no patient appears in both partitions.

### 3.4 Class distribution

The dataset is heavily skewed toward *Normal/Mild* outcomes — a clinically realistic distribution, since most patients imaged for low-back pain have only mild degeneration at most levels. Aggregated across all 25 labels:

| Severity | Approx. share |
|---|---|
| Normal/Mild (N) | ≈ 85 % |
| Moderate (M) | ≈ 12 % |
| Severe (S) | ≈ 3 % |

This imbalance is responsible for most of the per-condition disparity in §6 — labels for which Moderate/Severe cases are particularly rare (foraminal narrowing especially) end up under-trained.

### 3.5 Label schema

The schema used internally and in the model output is given by three maps shared across all scripts:

```python
COND_KEYS = {
    "spinal_canal_stenosis":            "canal",
    "left_neural_foraminal_narrowing":  "lf",
    "right_neural_foraminal_narrowing": "rf",
    "left_subarticular_stenosis":       "ls",
    "right_subarticular_stenosis":      "rs",
}
LEVEL_KEYS     = {"l1_l2": "L1L2", ..., "l5_s1": "L5S1"}
SEVERITY_CODES = {"Normal/Mild": "N", "Moderate": "M", "Severe": "S"}
```

The assistant target produced for each example is therefore a single compact JSON object such as:

```json
{
  "L1L2": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
  "L2L3": {"canal": "M", "lf": "N", "rf": "N", "ls": "N", "rs": "S"},
  "L3L4": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
  "L4L5": {"canal": "M", "lf": "M", "rf": "N", "ls": "N", "rs": "N"},
  "L5S1": {"canal": "N", "lf": "S", "rf": "M", "ls": "N", "rs": "N"}
}
```

---

## 4. Method

### 4.1 Base model

The starting point is `unsloth/medgemma-1.5-4b-it`, Unsloth's pre-quantised mirror of Google's MedGemma 1.5 4B-IT. Architecturally this is a SigLIP vision encoder feeding a Gemma 3 4 B decoder via a multimodal projection layer. The vision tower uses patch size 14, so input resolution maps to vision-token count as $(S/14)^2$ per image.

The base model is loaded in 4-bit using `bitsandbytes`, with the LoRA matrices in bf16 — the canonical QLoRA setup. The model exposes 2,734,255,984 total parameters; LoRA adds 38,497,792 trainable parameters, i.e. **0.89 %** of the model.

### 4.2 LoRA configuration

LoRA adapters are applied to both the language and the vision sides of the model. Target modules:

| Block | Targeted modules |
|---|---|
| Attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| MLP | `gate_proj`, `up_proj`, `down_proj` |
| Vision tower | same target patterns inside SigLIP |
| Language layers | same target patterns inside Gemma 3 |

Hyperparameters:

| LoRA parameter | Value |
|---|---|
| Rank, $r$ | 16 |
| Scaling, $\alpha$ | 32 |
| Dropout | 0.05 |
| Bias | none |
| Use rsLoRA | no |

Adapting the vision tower is essential: without it the model cannot learn to localise the small structures (foramina, subarticular recess) that the labels describe.

### 4.3 Multi-modality slice selection

A single MRI study is too large to feed wholesale into the model. We must select a small, label-relevant subset of slices.

The selector implements three principles:

1. **One slice per imaging purpose.** The *midline* sagittal T2 is the slice most useful for evaluating the central canal at all five levels. The two *parasagittal* T1 slices flanking the midline best show the neural foramina. The *axial* T2 slices, one per disc level when available, best show subarticular stenosis.
2. **All three modalities, every time.** When `train_label_coordinates.csv` is available, the picker queries it for the slice indices that overlap the labelled coordinates at each level, ensuring foraminal and subarticular labels are honestly evaluable. The pipeline refuses to fabricate level-to-slice mappings when coordinates are missing, unless an `--allow-no-coords` flag is set for smoke tests.
3. **Series priority and three-pass selection.** A naïve per-modality loop drops modalities for studies with multiple series of the same type. We instead use a *three-pass* selector:

```
SERIES_PRIORITY = ["sagittal_t2", "sagittal_t1", "axial_t2"]

Pass 1: pick one series of each type in priority order
Pass 2: if slots remain, fill with more series of any known modality
Pass 3: if slots still remain, allow unknown series last
```

With a per-study cap of three series and ≤ 8 total slices, this gives a typical example containing one midline sagittal T2, two parasagittal T1 slices, and up to five axial T2 slices.

### 4.4 Compact JSON output schema

A naïve assistant target is a paragraph of natural-language radiology prose. Such a target has two disadvantages: it is long (≈ 500 tokens) and the proportion of *label-bearing* tokens is small. Most of the loss is spent learning to produce filler.

We compress the answer to a single JSON object using short keys and single-character severity codes (N / M / S). The result is roughly **100 tokens** of assistant content, of which approximately 25 tokens carry the actual label information. Because the supervised loss runs only on assistant tokens (see §4.5), this 5× compression directly increases the proportion of label-bearing gradient signal.

A second benefit is parsing reliability. Prose answers must be matched with brittle regular expressions; compact JSON is parsed with a single `json.loads` call. Across the full 198-study validation set, the **parse failure rate is 0 %** (see §6).

### 4.5 Prompt format and supervised-loss masking

The same user-side prompt is used during training, evaluation, and live inference. It contains:

1. A description of the imaging context: `"You are provided with N lumbar spine MRI image(s) (Sagittal T2, Sagittal T1, Axial T2). Classify all degenerative conditions at each spinal level from L1/L2 to L5/S1."`;
2. A schema description that spells out the JSON structure and the meaning of every key and severity code;
3. The actual image tokens, one `<image>` placeholder per loaded slice.

The assistant turn is the compact JSON shown in §3.5. During training, `UnslothVisionDataCollator` masks every user-turn token to a label of `-100`, so cross-entropy loss is computed only on the assistant tokens. The model is therefore explicitly trained to *generate* the structured answer rather than to reproduce the prompt.

A single helper function `build_user_prompt(n_images, series_types)` constructs this user turn for both training and evaluation, guaranteeing that the inference-time prompt matches the SFT prompt verbatim. Any drift here would silently degrade evaluation quality.

### 4.6 Token-budget reasoning

At image size $448$ and patch size $14$, one image consumes $1024$ vision tokens. With the multi-modality picker delivering up to 8 images per example:

$$
\text{image tokens} = 8 \times (448/14)^2 = 8 \times 1024 = 8192.
$$

Adding ≈ 400 tokens for the user prompt, ≈ 100 tokens for the assistant JSON, and a small margin for chat-template tokens, a sequence length of **9,216** suffices for the demo profile and avoids truncating the assistant label. The choice of resolution (448 vs the model's native 896) is dictated by this budget: the same 8-image multi-modality setup at 896² would require ≈ 33k vision tokens, exceeding what fits in 40 GB even at QLoRA precision.

### 4.7 Training loop

Training uses TRL's `SFTTrainer` (v0.18+) on top of Unsloth's `FastVisionModel`. The data collator is `UnslothVisionDataCollator`, which jointly processes images and text and applies the user-turn masking described in §4.5. Lazy image loading is enabled via `Dataset.set_transform`, so PIL images are materialised per batch at `__getitem__` time rather than serialised into the PyArrow cache.

The supervised objective is standard cross-entropy on assistant tokens. The optimiser is AdamW with a cosine learning-rate schedule, warmup, and weight decay. An `EarlyStoppingCallback` with patience 3 monitors `eval_loss`, but does not trigger in any of our runs.

---

## 5. Experimental setup

### 5.1 Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA A100 80 GB PCIe |
| CUDA toolkit | 12.6 |
| Compute capability | 8.0 |
| Host | RunPod cloud GPU instance |
| OS | Linux (Ubuntu derivative) |

### 5.2 Software stack

| Library | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.12.0+cu126 |
| Unsloth | 2026.5.7 |
| Transformers | 5.5.0 |
| TRL | ≥ 0.18.2, ≤ 0.24.0 |
| PEFT | ≥ 0.18.0 |
| BitsAndBytes | ≥ 0.43.1 |
| Triton | 3.7.0 |

The full pinned set is encoded in `setup.sh` to ensure reproducibility across pods.

### 5.3 Hyperparameters (profile `demo_a100_40g`)

| Parameter | Value |
|---|---|
| Image resolution | 448 × 448 |
| Max sequence length | 9,216 tokens |
| Slices per series (axial) | up to 5 (one per level) |
| Series per study | up to 3 (sag T2 + sag T1 + axial T2) |
| LoRA rank, $r$ / $\alpha$ | 16 / 32 |
| LoRA dropout | 0.05 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Epochs | 3 (no early-stop trigger) |
| Optimiser | AdamW |
| Learning rate | 2 × 10⁻⁴ |
| LR schedule | cosine, warmup ratio 0.05 |
| Weight decay | 0.01 |
| Precision (LoRA) | bf16 |
| Precision (base) | 4-bit (NF4) |
| Eval interval | every 100 training steps |
| Save interval | every 200 training steps |
| Random seed | 42 |

### 5.4 Training duration

The 3-epoch run on 1,776 training examples produced **666 total optimizer steps** at effective batch size 8. Total wall-clock time: **14.28 hours** (≈ 51,400 s). Average throughput: 0.104 train samples / s; 0.013 train steps / s.

---

## 6. Results

### 6.1 Headline metrics

All metrics in this section are computed by `evaluate.py` on the held-out 198-study validation set (4,950 label predictions per condition aggregated across levels). The base model is `unsloth/medgemma-1.5-4b-it` with no adapter and identical prompt format; the fine-tuned model is the LoRA checkpoint saved at the end of epoch 3 in §5.4.

**Table 6.1. Aggregate metrics on the 198-study validation set.**

| Metric | Base | Fine-tuned | Δ |
|---|---:|---:|---:|
| Cohen's $\kappa$ (overall) | 0.000 | **0.502** | **+0.502** |
| Weighted accuracy ([1,2,4]) | 0.577 | **0.673** | +0.097 |
| F1 (weighted) | 0.683 | **0.777** | +0.095 |
| Raw accuracy | 0.779 | 0.814 | +0.035 |
| RSNA weighted score proxy (↓) | 6.826 | **5.268** | −1.558 |
| Parse failure rate | 0.0 % | **0.0 %** | 0 pp |

The key result is the kappa jump from **0.00 to 0.50**. Raw accuracy moved only marginally (78 % → 81 %), which is consistent with the base model exploiting the class prior — predicting Normal/Mild for every label. Once kappa-corrected, the base model contributes essentially no diagnostic signal, while the fine-tuned model reaches moderate agreement on the radiologist labels.

### 6.2 Per-condition Cohen's kappa

**Table 6.2. Per-condition kappa and accuracy (fine-tuned, validation).**

| Condition | Avg. accuracy | Avg. $\kappa$ (Base) | Avg. $\kappa$ (Fine-tuned) |
|---|---:|---:|---:|
| Spinal canal stenosis | 0.907 | 0.000 | **0.537** |
| Right subarticular stenosis | 0.799 | 0.000 | **0.501** |
| Left subarticular stenosis | 0.772 | 0.000 | **0.435** |
| Right neural foraminal narrowing | 0.804 | 0.000 | 0.066 |
| Left neural foraminal narrowing | 0.786 | 0.000 | 0.033 |

Three conditions — spinal canal stenosis and both subarticular stenoses — improve dramatically. The two foraminal-narrowing labels remain near chance. We discuss the reasons in §7.

### 6.3 Training dynamics

Training and validation loss curves are extracted from the TRL log. Cross-entropy on the assistant JSON is naturally small (the schema portion of the answer is highly predictable), so absolute loss values are in the 0.02–0.05 range; what matters is the downward trend.

**Table 6.3. Selected training-loss values (cosine schedule, 3 epochs, 666 steps).**

| Step | Training loss |
|---:|---:|
| 10 | 1.372 |
| 20 | 0.494 |
| 30 | 0.084 |
| 40 | 0.042 |
| 100 | 0.0344 |
| 200 | 0.0327 |
| 300 | 0.0312 |
| 400 | 0.0294 |
| 500 | 0.0283 |
| 600 | 0.0280 |
| 666 (final) | 0.0277 |

**Table 6.4. Evaluation-loss progression at every 100-step eval checkpoint.**

| Step | Eval loss |
|---:|---:|
| 100 | 0.0344 |
| 200 | 0.0327 |
| 300 | 0.0317 |
| 400 | 0.0294 |
| 500 | 0.0283 |
| 600 | 0.0280 |
| 666 (best) | 0.0278 |

Training loss collapses rapidly during the first ~30 steps as the model learns the JSON format, then enters a long slow regime in which it learns the actual labels. Validation loss decreases monotonically through every evaluation checkpoint, and `EarlyStoppingCallback(patience=3)` never triggers. There is no sign of overfitting within the 3 epochs run.

### 6.4 Confusion matrices

A 5 × 5 grid of confusion matrices (one per condition × per level) is produced by `evaluate.plot_confusion_matrices`. The figure file is `evaluation_results/finetuned_confusion_matrices.png`.

Qualitatively:
- For **canal** and **subarticular** rows, the diagonal is visibly populated for *Moderate* and *Severe* alongside the expected dominance of *Normal/Mild*.
- For **left foraminal** and **right foraminal** rows, predictions are almost entirely concentrated in the *Normal/Mild* column at every level, with virtually no diagonal entries in the Moderate and Severe rows.

This visual pattern matches the per-condition kappa values in Table 6.2.

### 6.5 Qualitative examples

The following compares ground truth, the base model, and the fine-tuned model on three representative validation studies.

**Example A — All Normal/Mild (Study 1324569502)**

```
Ground truth :  N N N N N | N N N N N | N N N N N | N N N N N | N N N N N
Base         :  N N N N N | N N N N N | N N N N N | N N N N N | N N N N N   ✓
Fine-tuned   :  N N N N N | N N N N N | N N N N N | N N N N N | N N N N N   ✓
```

Both models match perfectly. This is the most common case in the dataset, which is why raw accuracy on either model is around 80 %.

**Example B — Mixed Moderate/Severe (Study 4287160193)**

```
Ground truth (canal) :  N M N N N
Ground truth (lf)    :  N N N M N
Ground truth (ls)    :  N S M M N
Ground truth (rs)    :  N N M M N

Base (every label)        :  N (uniform, all levels)
Fine-tuned (every label)  :  N (uniform, all levels)
```

This case has several Moderate and one Severe label. Both models default to Normal/Mild everywhere. The fine-tuned model's improvement is *on average* over the dataset — it does not rescue every difficult case, and class imbalance especially hurts performance on isolated severe findings.

**Example C — Bilateral subarticular Moderate (Study 4189246764)**

```
Ground truth (ls) :  N N M M N
Ground truth (rs) :  M M M M N

Base       (ls / rs):  N N N N N / N N N N N
Fine-tuned (ls / rs):  N N N N N / N N N N N
```

Here the fine-tuned model again predicts all-Normal. The pattern of misses is not random: the model is most confident, and most accurate, on canal stenosis at L4/L5; isolated bilateral subarticular calls remain difficult without explicit class rebalancing.

These examples are taken from `comparison/text_examples.txt`. The headline 0.50 overall kappa comes from study cases where canal or subarticular Moderate/Severe is in fact detected — the confusion-matrix figure (§6.4) shows these diagonal hits, but they are spread across hundreds of validation predictions rather than concentrated in any single example.

### 6.6 Small-sample compare.py results

`compare.py` runs both models on a small subset (20 studies by default) and reports kappa per condition. With 20 samples per level, some Moderate/Severe classes have zero true positives at certain levels, which makes Cohen's kappa undefined (denominator 1 − p_e = 0, producing NaN). For example, the small-sample comparison reports:

| Condition (20-sample) | Base $\kappa$ | Fine-tuned $\kappa$ |
|---|---:|---:|
| Spinal canal stenosis | 0.00 | 0.37 |
| Left foraminal narrowing | NaN | NaN |
| Right foraminal narrowing | NaN | NaN |
| Left subarticular stenosis | 0.00 | 0.45 |
| Right subarticular stenosis | 0.00 | 0.38 |

The NaNs are a sample-size artefact, not a model failure: with only 20 studies × 5 levels per condition, certain (level × condition) cells have all-Normal truth or prediction labels, making chance-corrected agreement undefined. The authoritative numbers are the 198-sample `evaluate.py` results in Tables 6.1 and 6.2.

---

## 7. Discussion

### 7.1 What the fine-tune actually changed

The base model behaves as a *strong prior* classifier: when asked to fill in the JSON, it produces all *Normal/Mild* on essentially every study. This is locally sensible — Normal/Mild is the majority class for 25 of 25 labels — but it gives a Cohen's kappa of exactly zero, because predicting the mode of the distribution carries no information beyond that mode's marginal frequency.

The fine-tuned model differs *qualitatively*: it learns to predict *Moderate* and occasionally *Severe* on the conditions and levels where these are common (chiefly canal stenosis at L4/L5 and bilateral subarticular at L3–L5). Each such non-trivial prediction is a small win for kappa because $p_o$ now exceeds $p_e$. Adding up over the validation set, the model reaches $\kappa = 0.50$ — moderate agreement in the standard Landis–Koch (1977) interpretation.

It is essential to read this number alongside §6.5: even at $\kappa = 0.50$ the model still misses many specific Moderate/Severe instances. The improvement is real and statistically meaningful, but it is an averaged signal across thousands of label predictions, not a per-case promise.

### 7.2 Why foraminal narrowing did not improve

Per-condition kappa for left and right neural foraminal narrowing is 0.03 and 0.07 respectively — barely distinguishable from the base model. Three forces combine to produce this result.

First, **prevalence of Moderate/Severe at the foraminal labels is the lowest in the dataset.** Foraminal narrowing in older adults concentrates at L5/S1 and L4/L5, and even there a single side is more commonly affected than both. A non-trivial *Moderate* foraminal label appears in only a small fraction of training studies.

Second, **the rare-class signal must compete with the strong Normal/Mild prior on the same label.** Without any reweighting, the cross-entropy loss has many more Normal/Mild gradients than Moderate/Severe gradients on foraminal positions, so the model is implicitly trained to predict Normal/Mild there.

Third, **foraminal narrowing is visually subtle even on the dedicated sagittal T1 modality.** Distinguishing Normal from Moderate requires noticing the change in epidural fat between a normal foramen and a slightly compressed one. This is a delicate visual cue and may demand higher input resolution than the 448² used here.

### 7.3 The compact-JSON design paid off

The 0 % parse-failure rate observed in §6.1 is unusual. Even base-model responses, with no fine-tuning, produce valid parseable JSON 100 % of the time on this validation set. This is a direct consequence of two design choices:

1. The schema description appears verbatim in *every* user turn the model is prompted with. This is the same string at training, evaluation, and inference time. The model is therefore overwhelmingly likely to reproduce a JSON object obeying that schema.
2. The schema is short and structurally rigid. There are five fixed level keys, five fixed condition keys, and three single-character severity codes. Even a base model has little degree of freedom in deviating from this shape.

A consequence is that none of our metrics rely on a silent fallback — every prediction we count was actually emitted as parseable JSON, not produced by a default-to-Normal/Mild handler downstream of a parse failure.

### 7.4 The RSNA weighted score caveat

The `rsna_weighted_score_proxy` reported above is *not* the official RSNA log-loss. The competition metric requires calibrated softmax probabilities, which a generative language model does not natively emit. The proxy is computed by interpreting each hard prediction as a one-hot probability and applying the same class weights $[1, 2, 4]$. It is therefore a *weighted error rate scaled by* $\approx 16.1$, not a log-loss. The improvement from 6.83 to 5.27 in Table 6.1 indicates an absolute reduction in weighted error, but it cannot be compared to the public leaderboard.

### 7.5 Modality coverage matters

The `demo_a100_40g` profile was deliberately designed to include all three series types per study (sagittal T2, sagittal T1, and axial T2). A simpler profile that uses only one series — say sagittal T2 — would still allow honest evaluation of canal stenosis, but the foraminal and subarticular labels would necessarily be guessed from the class prior, because the model would not be looking at the relevant imaging at inference time. Our per-condition results in Table 6.2 confirm this design: the conditions whose dedicated modalities are actually included (canal on sag T2, subarticular on axial T2) move substantially; the conditions whose dedicated modality (parasag T1) is included but at low resolution still struggle.

---

## 8. Limitations

This work is an academic / research project and several limitations should be kept in mind when interpreting the results.

- **Class imbalance.** The model under-predicts the Moderate and Severe classes, particularly for foraminal narrowing. No class rebalancing (oversampling, class-weighted loss, or focal loss) was applied in this run.
- **Generative classification is metric-fragile.** Because the model outputs categorical predictions rather than calibrated probabilities, the RSNA-style weighted log-loss cannot be computed in its competition form. Comparisons to that leaderboard are therefore not meaningful.
- **Single dataset, single split.** All evaluation is on a 10 % held-out split of RSNA 2024. No external validation on imagery from a different institution, vendor, or population has been performed. Generalisation off-distribution is unknown.
- **Slice selection dependence.** Inference quality is contingent on the user supplying slices selected with the same multi-modality scheme used in training (midline sagittal T2 + parasagittal T1s + axial T2 per level). Random slices or single-modality input will degrade results.
- **No clinical validation.** The model has not been reviewed by a radiologist as part of this project. It is not a clinical-decision-support device and should not be used as one.
- **Compute budget.** Three epochs on a single A100 80GB. Larger profiles (`a100_80g_multimodal`, 672² resolution, LoRA r=32, 5 epochs) were proposed in `config.py` but not run due to time constraints.

---

## 9. Future work

Three follow-up directions are particularly promising. They are ordered roughly by expected improvement per unit of effort.

### 9.1 Oversampling

The pipeline already supports per-severity oversampling via `data_prep.py --oversample`. This duplicates each training study in proportion to its worst-case severity: Moderate × 2, Severe × 4. On the current data, this would increase the train split from 1,776 to approximately 5,000 examples while keeping the validation split untouched. The expected impact, based on the analysis in §7.2, is a non-trivial improvement in foraminal kappa (target $\kappa > 0.10$) at the cost of approximately 2.8× more training wall-clock time.

### 9.2 Higher-resolution multi-modality profile

The `a100_80g_multimodal` profile keeps the same multi-modality slice picker but bumps the resolution from 448² to 672² (i.e. 1.5× linear), the LoRA rank from 16 to 32, and the epoch count from 3 to 5. On an A100 80 GB this would consume roughly 65 GB of VRAM during training. The higher resolution would in particular help foraminal narrowing, where the relevant fat-displacement signal is small in pixels. Expected impact: canal kappa $> 0.6$; overall kappa $> 0.55$.

### 9.3 Replace generative output with a classification head

A small fully-connected head over the pooled vision representation, trained with class-balanced cross-entropy, would produce calibrated probabilities and enable a meaningful weighted log-loss. The same fine-tuned backbone could continue to be used for natural-language reporting, but the headline numbers would be defined the same way as in the RSNA competition. This is more invasive than the previous two changes but would close the gap with classical CNN baselines.

### 9.4 External validation

Even a small (∼ 100-study) external test set from a different institution would significantly strengthen any claim of generalisability. None of the public lumbar MRI datasets are perfectly comparable to RSNA 2024, but partial overlap on at least canal-stenosis grading is achievable.

### 9.5 Conversational deployment

The fine-tuned model is also pluggable into a two-model conversational architecture: the LoRA-tuned specialist emits the JSON; a base MedGemma instance reads the JSON as context and converses with the user in natural language. This pattern preserves classification accuracy without losing the chat ergonomics that base MedGemma offers, and was prototyped as part of this project (`demo_chat.py`).

---

## 10. Conclusion

We have shown that a parameter-efficient (QLoRA, $r = 16$) fine-tune of MedGemma 1.5 4B can learn to produce a structured 25-label diagnosis of degenerative lumbar spine pathology directly from MRI imagery. On a held-out 198-study validation split of the RSNA 2024 dataset, overall Cohen's kappa improves from **0.000** (base model) to **0.502** (fine-tuned), with a **0 % JSON parse-failure rate**. Three of the five conditions — spinal canal stenosis and both subarticular stenoses — reach kappa values in the 0.43–0.54 range, indicating genuine diagnostic agreement. The two foraminal-narrowing conditions remain near chance, which we attribute to class imbalance and to the visual subtlety of the relevant cue at 448² resolution. The compact-JSON output format, multi-modality slice picker, and label-only loss masking together produce a training pipeline that is small enough to run on a single A100 in under 15 hours yet honest enough to cover all 25 labels. The work is research-only and not clinically validated; oversampling, a higher-resolution profile, and a calibrated classification head are the three most promising next experiments.

---

## 11. References

A reduced reference list; expand with formal BibTeX entries in the LaTeX conversion.

1. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685 (2021).
2. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. **QLoRA: Efficient Finetuning of Quantized LLMs.** arXiv:2305.14314 (2023).
3. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L. **Sigmoid Loss for Language Image Pre-Training.** arXiv:2303.15343 (2023).
4. Google DeepMind. **MedGemma 1.5: A Vision-Language Model for Medical Image Understanding.** HuggingFace model card, 2025. https://huggingface.co/google/medgemma-1.5-4b-it
5. Google DeepMind. **Gemma 3 Technical Report.** 2025.
6. Cohen, J. **A Coefficient of Agreement for Nominal Scales.** *Educational and Psychological Measurement* 20(1):37–46 (1960).
7. Landis, J. R., Koch, G. G. **The Measurement of Observer Agreement for Categorical Data.** *Biometrics* 33(1):159–174 (1977).
8. Radiological Society of North America. **RSNA 2024 Lumbar Spine Degenerative Classification.** Kaggle competition (2024). https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification
9. von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., Huang, S., Rasul, K., Gallouédec, Q., **TRL: Transformer Reinforcement Learning.** GitHub: huggingface/trl.
10. Han, J. & contributors. **Unsloth: 2× faster, 70 % less memory finetuning of LLMs.** GitHub: unslothai/unsloth.
11. Wolf, T., et al. **HuggingFace's Transformers: State-of-the-art Natural Language Processing.** EMNLP system demonstrations, 2020.

---

## Appendix A — Reproducibility

| Item | Value |
|---|---|
| Repository | (local Git repo; SHA available via `git rev-parse HEAD`) |
| HuggingFace adapter | `YOUR_USERNAME/medgemma-lumbar-finetune` (private/public per the model card) |
| Profile used | `demo_a100_40g` |
| Random seed | 42 |
| Dependency lockfile | `setup.sh` (pinned versions) |
| Training command | `PROFILE=demo_a100_40g python train.py` |
| Evaluation command | `python evaluate.py --model_path /workspace/models/medgemma-lumbar/final` |
| Comparison command | `python compare.py` |

## Appendix B — Aggregate metrics in machine-readable form

For reproducibility, the headline numbers from §6.1 and §6.2 mirror the JSON dumps produced by `evaluate.py` and `compare.py` (`evaluation_results/metrics.json`, `evaluation_results/ft_metrics.json`, `comparison/comparison_results.json`).

## Appendix C — Inference template

Inference at deployment time uses the same user-prompt builder as training:

```python
from data_prep import build_user_prompt

user_prompt = build_user_prompt(
    n_images=len(images),
    series_types=["sagittal_t2", "sagittal_t1", "axial_t2"],
)
# Pass `user_prompt` plus N <image> tokens through tokenizer.apply_chat_template;
# decode the model's response and parse with json.loads.
```

This keeps the inference-time prompt verbatim-identical to the SFT-time prompt and prevents silent distribution shifts.
