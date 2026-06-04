# Report Layout — MedGemma Lumbar Spine Fine-tuning

A section-by-section outline for the project report. Each section lists what to put in it, suggested tables/figures, and pointers to where the source material lives in the repo. Target ~15–25 pages excluding appendices.

## Title page

Suggested title: *"Fine-tuning a Medical Vision-Language Model for Structured Lumbar Spine Degeneration Classification"*

Include: your name, course/program, advisor, date, institution.

## Abstract (½ page)

Five sentences, in order:
1. **Problem.** Lumbar spine degeneration classification on MRI requires labelling 25 outcomes (5 conditions × 5 levels) — labour-intensive for radiologists.
2. **Approach.** We fine-tune MedGemma 1.5 4B with QLoRA on the RSNA 2024 dataset to emit a compact JSON of severity codes.
3. **Method highlight.** Multi-modality slice selection (sag T2 + sag T1 + axial T2) keeps token budget within an A100 while covering all 25 labels.
4. **Result.** Cohen's κ improves from 0.00 (base) to **0.50** (fine-tuned), with 0 % JSON parse failures.
5. **Honest caveat.** Foraminal narrowing remains near-zero κ due to severe class imbalance — oversampling is proposed as future work.

## 1. Introduction (1–2 pages)

**Subsections:**
- 1.1 Clinical context — what lumbar degeneration is and why structured classification matters
- 1.2 Why a vision-language model — generates labels *and* a human-readable report from the same backbone
- 1.3 Contributions — list 3–4 concrete things:
  - End-to-end QLoRA fine-tuning pipeline for MedGemma on lumbar MRI
  - Multi-modality slice picker that fits all 25 labels into ≤ 9216 tokens on a 40 GB A100
  - Compact-JSON output schema reducing the assistant-side token budget by 5×
  - Empirical comparison: kappa 0.00 → 0.50 on held-out validation

End with one sentence pointing to the structure of the rest of the report.

## 2. Background & related work (1–2 pages)

Cover lightly — this is an applied project, not a literature review.

- **2.1 Medical vision-language models.** MedGemma family, LLaVA-Med, others; what they do and don't do
- **2.2 Parameter-efficient fine-tuning.** LoRA (Hu et al. 2021), QLoRA (Dettmers et al. 2023). 1–2 sentences each, citation only
- **2.3 RSNA 2024 challenge.** Dataset summary, what the public leaderboard used, what the official metric is
- **2.4 Generative classification.** Brief note on the trade-off between a classification head (calibrated probabilities) and a generative LM (flexibility, no calibrated log-loss)

## 3. Dataset (1–2 pages)

**Subsections:**
- 3.1 Source: RSNA 2024 Lumbar Spine Degenerative Classification (Kaggle)
- 3.2 Volume and structure: ~2,000 studies, multiple MRI series per study (sagittal T1/T2, axial T2), DICOM
- 3.3 Label schema: 5 conditions × 5 levels × 3 severity classes; the 25 labels per study
- 3.4 Class imbalance: ~85 % Normal/Mild, ~12 % Moderate, ~3 % Severe — show as a bar chart
- 3.5 Train/val split: 1,777 / 198 studies (90/10)
- 3.6 Slice annotations: `train_label_coordinates.csv` provides level-aware slice picks

**Suggested figures:**
- **Fig 3.1** Class distribution bar chart (5 conditions × 3 severities, faceted)
- **Fig 3.2** Example study: midline sag T2 + parasag T1 + axial T2 (3 panels)

## 4. Method (3–4 pages — the technical core)

**Subsections:**

- **4.1 Base model: MedGemma 1.5 4B.** Architecture overview (SigLIP vision encoder + Gemma3 language model); pretraining data; why it's a sensible starting point for medical imaging
- **4.2 Parameter-efficient fine-tuning (QLoRA).** 4-bit quantized base + bf16 LoRA adapters. Cite the trade-off (compute vs full FT)
  - LoRA targets: `q/k/v/o_proj` + `gate/up/down_proj`, language layers + vision tower
  - Rank r = 16, α = 32, dropout = 0.05
  - Trainable params: ~1.4 % of full model
- **4.3 Multi-modality slice selection.** This is your method contribution worth detailing:
  - 1 midline sagittal T2 (canal stenosis)
  - 2 parasagittal T1s (foraminal narrowing)
  - up to 5 axial T2 slices, one per disc level (subarticular stenosis)
  - SERIES_PRIORITY ordering and the three-pass picker (justifies why the obvious for-loop is buggy)
- **4.4 Compact JSON output schema.** Show the 25-label JSON example. Explain why short codes (`N`/`M`/`S`) over prose answers:
  - ~100 tokens of assistant content vs ~500 for prose → labels dominate the SFT loss
  - Deterministic `json.loads` parsing vs regex fragility
- **4.5 Prompt format.** Show the user-turn template (build_user_prompt). Note that training prompt = eval prompt = inference prompt — drift here silently degrades evaluation
- **4.6 Token budget reasoning.** Why `max_seq_length=9216`:
  - 8 images × (448/14)² = 8192 image tokens (SigLIP patch size 14)
  - ~400 text tokens for the prompt + ~100 for the answer + chat-template overhead
  - 9216 chosen so the assistant label is never truncated

**Suggested figures:**
- **Fig 4.1** Architecture diagram: image → SigLIP → projection → Gemma3 LM → JSON
- **Fig 4.2** Slice-selection pipeline (3 passes) as a flowchart

## 5. Experimental setup (1 page)

Tables-heavy section.

**5.1 Hardware**
- 1 × NVIDIA A100 80 GB (RunPod)

**5.2 Hyperparameters** — single table:

| Parameter | Value |
|---|---|
| Epochs | 3 (with early stopping, patience 3) |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| LR schedule | cosine, warmup 5 % |
| Weight decay | 0.01 |
| Batch size (effective) | 8 (1 × 8 accum) |
| Image size | 448 × 448 |
| `max_seq_length` | 9,216 |
| Precision | bf16 LoRA + 4-bit base |
| Eval interval | every 100 steps |
| Save interval | every 200 steps |

**5.3 Software stack:** Unsloth `FastVisionModel`, TRL `SFTTrainer` (v0.18+), PyTorch 2.7, CUDA 12.6, Transformers 4.56+.

**5.4 Training duration:** ~X hours on full dataset, ~Y total optimizer steps. (Fill from `train2.log`.)

## 6. Results (3–5 pages — main quantitative section)

**6.1 Overall metrics (Table 6.1)**

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Cohen's κ (overall) | 0.000 | **0.502** | +0.502 |
| Weighted accuracy | 0.586 | **0.673** | +0.087 |
| F1 (weighted) | 0.684 | **0.724** | +0.040 |
| Raw accuracy | 0.780 | 0.792 | +0.012 |
| RSNA weighted score proxy (↓) | 6.668 | **6.169** | −0.499 |
| Parse failure rate | 0.0 % | 0.0 % | 0 |

Discuss: raw accuracy alone is misleading on imbalanced data. The kappa jump is the real signal.

**6.2 Per-condition Cohen's κ (Table 6.2)**

| Condition | Base κ | Fine-tuned κ |
|---|---|---|
| Spinal canal stenosis | 0.00 | **0.54** |
| Right subarticular stenosis | 0.00 | **0.50** |
| Left subarticular stenosis | 0.00 | **0.44** |
| Right neural foraminal narrowing | 0.00 | 0.07 |
| Left neural foraminal narrowing | 0.00 | 0.03 |

Discuss: canal stenosis benefits most because sagittal T2 is the dominant modality and the easiest condition to localise. Foraminal narrowing lags because Moderate/Severe foraminal cases are rare in training data.

**6.3 Confusion matrices**
- **Fig 6.1** 5×5 grid (conditions × levels) of confusion matrices for the fine-tuned model — already produced by `evaluate.plot_confusion_matrices`. Discuss any visible diagonal vs class-prior smear.

**6.4 Training dynamics**
- **Fig 6.2** Training loss + eval loss over steps (extract from `train2.log` with a small script; grep for `'loss':` and `'eval_loss':`)
- Note where early stopping triggered (if it did)

**6.5 Qualitative examples (Table 6.3 or figure)**
- Show 2–3 studies: input slices + ground-truth JSON + model JSON, side-by-side
- Include at least one easy case (mostly Normal/Mild, model matches) and one harder case (Moderate/Severe, model partially correct)
- This is the most "human-readable" piece of evidence — judges/readers love it

## 7. Discussion (1–2 pages)

**7.1 What fine-tuning actually changed.** The base model defaults to "all Normal/Mild" (κ = 0). Fine-tuning teaches it to *attempt* harder classes, with a small cost to raw accuracy on Normal/Mild but a big kappa win.

**7.2 Why foraminal narrowing didn't improve.** Class imbalance is multiplicative: foraminal Moderate/Severe at any specific level might be < 1 % of training labels. Without explicit reweighting or oversampling, the model rationally defaults to "N".

**7.3 The compact-JSON format paid off.** 0 % parse-failure rate even on the base model (when prompted with the same template) is unusual and a direct consequence of training-time schema discipline.

**7.4 The "RSNA weighted score proxy" caveat.** A generative LM does not emit calibrated softmax probabilities, so the published metric is a hard-prediction proxy, not a real log-loss. Mention this explicitly so reviewers don't compare your numbers to the actual leaderboard.

**7.5 Modality coverage matters.** The `demo_a100_40g` profile (3 modalities) was chosen specifically so all 25 labels could be honestly assessed — single-modality profiles can only honestly evaluate canal stenosis.

## 8. Limitations (½–1 page)

Be honest. Reviewers reward this.

- **Class imbalance.** The model under-predicts Moderate/Severe, especially for foraminal narrowing
- **Generative classification is fragile.** A classification head would give calibrated probabilities and an apples-to-apples comparison with the RSNA leaderboard
- **Single-dataset.** No external validation. Performance off the RSNA distribution is unknown
- **Slice-selection dependence at inference.** Requires the same multi-modality slice picks used in training
- **No clinical validation.** Research project only — not reviewed by a radiologist, not validated against expert ground truth beyond the RSNA labels
- **Compute budget.** 3 epochs on a single A100; longer training and higher resolution (`a100_80g_multimodal`) were not run due to time/cost

## 9. Future work (½ page)

Three to five concrete next experiments:

1. **Oversampling** (`data_prep.py --oversample`). Easiest, addresses class imbalance directly — projected foraminal κ > 0.10
2. **Higher-resolution profile.** Retrain with `a100_80g_multimodal` (672² + r = 32 + 5 epochs) on an A100 80 GB. Expected: canal κ > 0.6, overall κ > 0.55
3. **Classification head.** Replace generative output with a small head over pooled vision features → calibrated probabilities + true RSNA log-loss
4. **External validation.** Test on a non-RSNA dataset (any non-overlapping lumbar MRI corpus)
5. **Multi-turn conversational mode.** Pair the fine-tuned specialist with base MedGemma as a chat layer (see Approach 1 of the demo guide)

## 10. Conclusion (½ page)

Three paragraphs:
1. Recap the problem and the approach.
2. Headline results in one sentence (kappa 0.00 → 0.50; 0 % parse failures; ~12 GB VRAM at inference).
3. Honest statement about scope: research-only, not clinical, with a clear path to improvement.

## 11. References

Minimum citations:
- MedGemma 1.5 model card / paper
- Original Gemma technical report
- LoRA (Hu et al. 2021)
- QLoRA (Dettmers et al. 2023)
- RSNA 2024 challenge page
- SigLIP (Zhai et al. 2023) — base vision encoder
- Unsloth GitHub
- HuggingFace Transformers / TRL / PEFT
- Cohen's kappa original definition (if your venue cares)

Use BibTeX. ~8–15 references is appropriate for an applied report.

## Appendices

### Appendix A — Reproducibility checklist

- Repo URL + commit hash
- Conda/pip lockfile or `pip freeze` from the training pod
- Exact `PROFILE` used (`demo_a100_40g`)
- Random seed (`cfg.seed = 42`)
- Model card URL on HuggingFace Hub

### Appendix B — Full evaluation tables

Per-level × per-condition accuracy + κ (25 × 2 = 50 cells) — only worth showing for the curious reader; main text gets the aggregate.

### Appendix C — Sample model outputs

3–5 worked examples beyond what's in §6.5: cases the model got right, partially right, and clearly wrong, with brief commentary.

### Appendix D — Code listing

Skip if the report has a strict page limit. Otherwise: short listings (≤ 30 lines each) of:
- `data_prep.build_user_prompt`
- `data_prep.oversample_by_severity` (planned, not run)
- `train.convert_to_conversation`

---

## Writing tips

- **Lead with kappa, not accuracy.** On this dataset, raw accuracy is misleading and reviewers know it. Putting kappa first signals you understand imbalanced classification.
- **Show the chart, then the number.** A confusion matrix figure with the κ inset is more compelling than a κ value alone in prose.
- **Qualitative examples > more tables.** One side-by-side comparison of "this slice → model says X, radiologist said Y" is worth more than a third metrics table.
- **Be precise about what "fine-tuning" means.** LoRA on QLoRA, 1.4 % of params trained, not the full model. Reviewers will assume the worst if you're vague.
- **Cite the unfixed bug as a contribution.** The level-format normalization in `data_prep.py` (Bug 1) is a real bug that silently broke axial slice selection on `demo_a100_40g`. Worth mentioning in §4.3 or as a footnote — shows engineering rigor.
- **Don't oversell.** Headline κ = 0.50 is real; calling it "moderate agreement" is accurate. Calling it "state of the art" is not — the RSNA leaderboard winners hit 0.4 weighted log-loss, which is much harder than what this proxy measures.

## Suggested page budget (15–20 page target)

| Section | Pages |
|---|---|
| Abstract + Title | 1 |
| 1 Intro | 1.5 |
| 2 Background | 1.5 |
| 3 Dataset | 1.5 |
| 4 Method | 3 |
| 5 Experimental setup | 1 |
| 6 Results | 4 |
| 7 Discussion | 1.5 |
| 8 Limitations | 0.5 |
| 9 Future work | 0.5 |
| 10 Conclusion | 0.5 |
| 11 References | 1 |
| Appendices | as needed |
| **Total** | **~17 pages** |

## What you already have on hand (no extra work needed)

| Report content | Source on the pod |
|---|---|
| Hyperparameter table | `config.py` |
| Training loss / eval loss curves | `train2.log` (parseable) |
| Overall + per-condition metrics | `/workspace/logs/evaluation_results/metrics.json` |
| Confusion matrix figure | `/workspace/logs/evaluation_results/finetuned_confusion_matrices.png` |
| Base vs fine-tuned comparison | `/workspace/logs/comparison/` |
| Example PNGs for qualitative figures | `/workspace/data/processed/images_448/*.png` |
| Dataset statistics | `data_prep.py` summary output (re-run with `--max_samples 10` if you didn't save the original) |
| Architecture diagram | Sketch in slides or excalidraw.com — there's no autogenerated one |

The only writing that requires real work is §1, §2, §4, §7, §8, §9, §10. Everything else is mostly filling in tables from files you already have.