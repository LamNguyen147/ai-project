# Conversational Demo Guide

A two-model interactive demo for the MedGemma lumbar spine fine-tune. The fine-tuned model produces structured JSON findings; the base MedGemma chats about those findings in natural language.

## TL;DR

```bash
cd /workspace/medgemma-finetune
python demo_chat.py
```

In the prompt:
```
You: /sample
You: /analyze
You: What does the L4/L5 finding mean clinically?
```

## What this demo shows

A clean two-stage architecture:

```
┌─────────────────────────┐       ┌──────────────────────────┐
│  Fine-tuned MedGemma    │       │   Base MedGemma 1.5 4B   │
│  (specialist, LoRA)     │       │   (general chat)         │
│                         │       │                          │
│   MRI → JSON findings   │ ───▶  │   findings + question    │
│   (deterministic 25 lbl)│       │   → natural language     │
└─────────────────────────┘       └──────────────────────────┘
```

**Why this is interesting for a research demo:**
- Separates *perception* (vision → structured prediction) from *language* (structured → natural)
- The specialist gives you accuracy on a narrow task without sacrificing chat ability
- It's the same architecture used in modern production AI systems (e.g. specialist model + LLM frontend)
- The fine-tuned model never invents findings — the chat model is grounded in its JSON output

## What you need before the demo runs

The demo needs three things on the pod:

1. **The project code** — `demo_chat.py`, `json_to_report.py`, `config.py`, `data_prep.py`.
2. **The fine-tuned LoRA adapter** at `/workspace/models/medgemma-lumbar/final/` (or an HF repo id).
3. **At least one study's worth of demo images** plus the `val_dataset.jsonl` that the `/sample` command reads. (Or any standalone lumbar-spine PNG you `/load` manually.)

Base MedGemma 1.5 (~4 GB) is downloaded automatically into the HF cache on the
first model load — you don't pre-fetch it.

Total VRAM used: ~12 GB (both models loaded in 4-bit, plus activations).

| GPU | Will it fit? |
|---|---|
| RTX 3090 (24 GB) | ✓ tight but works |
| RTX 4090 (24 GB) | ✓ |
| A40 / L40S / RTX 6000 Ada (48 GB) | ✓ |
| A100 40 / 80 GB | ✓ (overkill) |

Pick a RunPod template with a **CUDA 12.6+ / PyTorch image** (same requirement as
training — see `RUNPOD.md` step 1).

---

## Full setup from a fresh pod

Pick the track that matches your situation:

- **Track A — the network volume from training is reattached.** Everything is
  already on `/workspace`. Jump to [Track A](#track-a--volume-reattached-fast-path).
- **Track B — brand-new pod, nothing on it.** Follow
  [Track B](#track-b--brand-new-pod-full-path) to rebuild from scratch.

### Prerequisites (both tracks)

You need your HuggingFace token to pull the adapter and the base model:

| Variable | Where to get it |
|---|---|
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens → New token (Read) |

Track B also needs Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) to
download the dataset for demo images. Set all of these in the RunPod
**Environment Variables** panel so every terminal sees them, or `export` them
after connecting.

You must also have accepted the Gemma license on the
[MedGemma model page](https://huggingface.co/google/medgemma-1.5-4b-it).

---

### Track A — volume reattached (fast path)

If the training volume is mounted at `/workspace`, the code, adapter, and
processed data already exist. You only need to install the inference deps.

```bash
# 1. Confirm the pieces are present
ls /workspace/medgemma-finetune/demo_chat.py
ls /workspace/models/medgemma-lumbar/final/adapter_model.safetensors
wc -l /workspace/data/processed/val_dataset.jsonl   # >0 lines = /sample will work

# 2. Install inference dependencies (training stack not needed)
export HF_TOKEN=hf_xxx
cd /workspace/medgemma-finetune
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install "transformers>=4.56.1,!=4.57.4,!=4.57.5,!=5.0.0,!=5.1.0,<=5.5.0" \
            "trl>=0.18.2,!=0.19.0,<=0.24.0" \
            "peft>=0.18.0" \
            "bitsandbytes>=0.43.1" \
            "accelerate>=1.0" \
            "pillow" \
            "pydicom"
huggingface-cli login   # paste $HF_TOKEN

# 3. Smoke test, then run the demo (Steps 5 and 6 below)
```

Then skip to [Step 5 — smoke test](#step-5--smoke-test).

---

### Track B — brand-new pod (full path)

Nothing is on the pod. You'll clone the repo, install everything, pull the
adapter from HF Hub, and regenerate a handful of demo images.

#### Step 1 — Clone the repo and run setup

```bash
export HF_TOKEN=hf_xxx
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=xxxxxxxxxxxxxxxx

cd /workspace
git clone <your-repo-url> medgemma-finetune
cd medgemma-finetune

bash setup.sh
```

`setup.sh` installs unsloth + the pinned inference/training stack, logs into
HuggingFace, and writes your Kaggle credentials. ~5–10 minutes. It aborts
immediately if any of the three env vars is missing.

#### Step 2 — Download the fine-tuned adapter

```bash
mkdir -p /workspace/models/medgemma-lumbar/final
huggingface-cli download YOUR_USERNAME/medgemma-lumbar-finetune \
    --local-dir /workspace/models/medgemma-lumbar/final \
    --repo-type model

# Verify the adapter files landed
ls /workspace/models/medgemma-lumbar/final/
# Must include: adapter_config.json  adapter_model.safetensors  + tokenizer files
```

Replace `YOUR_USERNAME` with your HF username (~200 MB download).

#### Step 3 — Get demo images

The `/sample` command reads `/workspace/data/processed/val_dataset.jsonl` and
the PNG files it points to. On a fresh pod you have to regenerate a small slice
of these:

```bash
# 3a. Download just the RSNA dataset (needed for the DICOMs)
mkdir -p /workspace/data/rsna-2024-lumbar-spine
cd /workspace/data/rsna-2024-lumbar-spine
kaggle competitions download \
    -c rsna-2024-lumbar-spine-degenerative-classification
unzip -q rsna-2024-lumbar-spine-degenerative-classification.zip
rm rsna-2024-lumbar-spine-degenerative-classification.zip

# 3b. Build PNGs + JSONL for a handful of studies (fast — minutes, not an hour)
cd /workspace/medgemma-finetune
python data_prep.py --max_samples 50
```

`--max_samples 50` keeps `data_prep.py` to a few minutes and a few hundred PNGs
— plenty for a demo. Drop the flag if you want the full validation set.

> **Shortcut:** if you don't care about real RSNA studies you can `/load` any
> lumbar-spine PNG manually and skip the whole dataset download. But `/sample`
> won't work without `val_dataset.jsonl`, and the model was trained on the
> specific preprocessing in `data_prep.py`, so real samples look best.

After it finishes, confirm:

```bash
wc -l /workspace/data/processed/val_dataset.jsonl   # ~10 lines for max_samples 50
ls /workspace/data/processed/*.png | head           # the PNGs /sample will load
```

---

### Step 5 — Smoke test

Both tracks converge here. Confirm the adapter loads and emits valid JSON
before the live demo:

```bash
cd /workspace/medgemma-finetune
python test_inference.py
```

Takes ~1–2 minutes (longer on the first run while base MedGemma downloads).
A successful run ends with:

```
Parsed JSON (pretty):
{
  "L1L2": { "canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N" },
  ...
}

[OK] Inference smoke test passed.
```

If you see `JSON parse failed`, re-run once (rare stochastic glitch). If it
fails consistently, your `ADAPTER_PATH` is probably pointing at the base model
— recheck Step 2.

---

### Step 6 — Launch the demo

Run inside `tmux` so the session survives an SSH disconnect:

```bash
tmux new -s demo
cd /workspace/medgemma-finetune
python demo_chat.py
```

Detach with `Ctrl+B` then `D`; re-attach later with `tmux attach -t demo`.

The first model load downloads base MedGemma (~4 GB) into the HF cache. After
that, every subsequent demo run reuses the cache and loads in 30–60 seconds.
Once you see the banner, jump to [Commands](#commands) and the
[demo script](#suggested-4-minute-demo-script).

## CLI flags (optional)

`python demo_chat.py` works with no arguments — it defaults to the adapter at
`/workspace/models/medgemma-lumbar/final` and base `unsloth/medgemma-1.5-4b-it`.
Override either default if needed:

```bash
# Point at a custom adapter (e.g. pull straight from an HF repo id)
python demo_chat.py --adapter YOUR_USERNAME/medgemma-lumbar-finetune

# Use a different base chat model
python demo_chat.py --base google/medgemma-1.5-4b-it
```

Initial load takes ~30–60 seconds on a warm cache, longer on a cold one.

## Commands

| Command | What it does |
|---|---|
| `/load <path>` | Add one PNG to the next analysis. Repeat for multiple slices. |
| `/study <id>` | Load **every slice of one study** at once (uses the dataset's recorded image set, or globs `<study_id>_*.png`). The convenient way to load a study you pre-screened with `find_demo_samples.py`. |
| `/sample [sev]` | Load a val study, optionally filtered by severity: `/sample severe`, `/sample moderate`, or `/sample` (any). Prints the ground-truth abnormal labels so you know what the model *should* find. |
| `/analyze` | Run the fine-tuned model on loaded images. Produces JSON + auto-summary. (Optional — asking a question after `/load`/`/sample` auto-runs it.) |
| `/compare` | Run **both** the fine-tuned specialist and the base model on the same images, side by side. Shows what fine-tuning bought you (clean JSON vs base rambling/refusing). |
| `/base <q>` | Ask the **base** MedGemma a free-form question directly about the loaded images (it sees the pixels). Useful for "what would the un-tuned model say?" |
| `/report` | Print the deterministic radiology-style report from the current findings. |
| `/findings` | Dump raw JSON findings (good for showing "this is what the specialist produced"). |
| `/state` | Show what's loaded right now. |
| `/clear` | Reset images, findings, and chat history. |
| `/quit` | Exit. |

Anything not starting with `/` is treated as a chat question about the current findings.

## Picking a sample that shows the model actually learned something

The RSNA dataset is **~85% Normal/Mild**, so a plain `/sample` almost always
loads an all-normal study — the model says "everything's normal" and you've
proven nothing. To demonstrate that the fine-tune *learned to detect pathology*,
you need a study that contains Moderate/Severe labels **and** that the model
predicts correctly.

### Quick path — filter `/sample` by severity

During the demo, just ask for an abnormal study:

```
You: /sample severe
[Loaded 8 image(s) from study 1234567 (filter: severe):]
    - /workspace/data/processed/1234567_..._sagittal_t2_slice4.png
    - ...
  Ground-truth abnormal labels: L4L5 canal=S, L5S1 lf=M
```

`/sample severe` scans `val_dataset.jsonl` for the first study whose
`max_severity` field is `2` (contains a Severe label); `/sample moderate` finds
one with Moderate or worse. The printed "Ground-truth abnormal labels" line
tells you exactly what the model *should* call out — handy for confirming the
prediction is right when you then ask a question.

> This picks the *first* matching study, which may or may not be one the model
> gets right. For a guaranteed-good demo, pre-screen with the script below.

### Best path — pre-screen candidates with `find_demo_samples.py`

Run this **before** the demo (it's a rehearsal tool). It runs the model on every
Severe-containing val study and ranks them by how many abnormal cells the model
predicts correctly, so you walk into the demo with a known-good study:

```bash
cd /workspace/medgemma-finetune
python find_demo_samples.py                 # rank Severe-containing studies
python find_demo_samples.py --severity moderate   # include Moderate cases
python find_demo_samples.py --limit 30      # cap runtime to first 30 candidates
```

Output ends with ready-to-paste `/load` lines for the top studies:

```
================================================================
 Top 5 demo candidates
 (the model correctly flags real pathology in these)
================================================================

Study 1234567 — model got 3/3 abnormal cells exactly right (3/3 flagged abnormal)
  Ground-truth abnormal: L4L5 canal=S, L4L5 ls=M, L5S1 lf=M
  Paste into the demo:
    /load /workspace/data/processed/1234567_..._sagittal_t2_slice4.png
    /load /workspace/data/processed/1234567_..._sagittal_t1_slice3.png
    ...
```

Pick a study near the top (high "exact" score), copy its `/load` lines into a
notes file, and paste them during the demo. Now when you ask "what could this
be?", the model reliably surfaces the real Severe/Moderate findings.

> **Timing:** the script runs the full model on each candidate (~20–40 s each).
> There are usually only a handful of Severe studies in a 198-study val split,
> so it finishes in a few minutes. Use `--limit` if you included Moderate
> (`--severity moderate`) and there are many candidates.

## Suggested 4-minute demo script

This is a recommended flow if you're presenting live. The "Say:" lines are what you can narrate; the `You:` lines are exactly what to type.

### Step 1 — Open with context (~20 s)

> **Say:** "I fine-tuned MedGemma 1.5 4B on the RSNA 2024 Lumbar Spine dataset. The model classifies 5 conditions × 5 vertebral levels — 25 labels total. I'll demo it conversationally, but under the hood the specialist model emits a compact JSON, and a chat layer translates that into natural language."

### Step 2 — Show the inputs (~30 s)

```
You: /sample severe
```

(Or paste the `/load` lines for a study you pre-screened with
`find_demo_samples.py` — see the section above. Using `severe` ensures the study
actually contains pathology rather than being all-normal.)

> **Say:** "These are real preprocessed slices from the validation set — one midline sagittal T2, two parasagittal T1s for foraminal assessment, and axial T2s per level for subarticular evaluation. The fine-tuned model has never seen this study before."

(Optional: open one of the PNGs from `/workspace/data/processed/*.png` in the IDE to show the image — they're stored flat as `<study_id>_<series_id>_<type>_slice<N>.png`.)

### Step 3 — Run the analysis (~30 s)

```
You: /analyze
```

The model produces JSON findings and the chat layer immediately summarises them. While it's generating, narrate:

> **Say:** "The specialist takes ~10 seconds to produce structured findings. The JSON is compact — only ~100 tokens — which keeps the SFT loss focused on labels rather than prose."

### Step 4 — Show both layers (~30 s)

```
You: /findings
You: /report
```

> **Say:** "This is the raw output of the specialist — every level, every condition, every severity. The Python formatter turns it into a radiology-style report. The chat layer can be much more flexible than the formatter."

### Step 4b — Fine-tuned vs base, same input (~40 s, optional but compelling)

This is the clearest way to show what fine-tuning actually changed. Both models
get the *identical* images and prompt:

```
You: /compare
```

```
==============================================================
 Fine-tuned specialist (LoRA) — compact 25-label JSON
==============================================================
{"L1L2":{"canal":"N",...},"L4L5":{"canal":"S",...}, ...}

==============================================================
 Base MedGemma (no fine-tuning) — same input, raw output
==============================================================
I can see a sagittal MRI of the lumbar spine. There appears to be...
(prose, hedging, or a refusal — usually NOT the compact schema)
```

> **Say:** "Same images, same prompt. The fine-tuned model emits a clean,
> parseable 25-label JSON every time — 0% parse-failure on validation. The base
> model is a capable medical VLM, but it narrates in free text and won't reliably
> produce the structured schema we need to score. That structured reliability is
> exactly what the fine-tune bought us."

You can also interrogate the base model directly on the images:

```
You: /base What abnormalities do you see in these MRI images?
```

> **Say:** "The base model can describe images, but it won't give us per-level,
> per-condition severities in a fixed format. That gap is the whole reason we
> fine-tuned."

### Step 5 — Conversational follow-ups (~2 min, the highlight)

Pick 2–3 of these to demonstrate breadth:

```
You: What does the L5/S1 finding mean for the patient?
You: Are there any urgent findings I should flag?
You: Which level has the most disease?
You: Can you write a one-paragraph dictation-style report?
You: Is there anything at L1/L2 worth mentioning?
```

> **Say (after one of the answers):** "Notice the chat model is grounded in the JSON — it never invents findings. If I ask about a condition that's actually normal in the JSON, it'll say so."

### Step 6 — Optional: failure mode honesty (~20 s)

```
You: Could the model have missed anything?
```

The chat model usually responds with something honest. This is a good moment to mention:

> **Say:** "On the held-out validation set the fine-tuned model gets Cohen's kappa of 0.50 overall — moderate agreement. Spinal canal stenosis hits 0.54; foraminal narrowing is still near-zero because Moderate/Severe foraminal cases are rare. Oversampling is the next experiment."

#### Optional: show input-scoped reliability (partial input)

A single MRI slice can only show some anatomy — a midline sagittal T2 shows the
canal, parasagittal T1 shows the foramina, axial T2 shows the subarticular
recesses. The fine-tuned model always emits all 25 labels, but a label is only
*visually grounded* when the modality that shows it is in the input.

The demo handles this in two ways when a modality is missing:
1. It **drops** the un-assessable conditions from the findings *before* the base
   chat model sees them — so the natural-language reply only discusses anatomy
   the slices can actually show (no off-view guesses leak into the conversation).
2. It prints a **coverage note** telling you exactly what was dropped.

To make this explicit, load a partial input on purpose and analyze it:

```
You: /clear
You: /load /workspace/data/processed/<study>_..._sagittal_t2_slice4.png
You: What does this show?
...
Analyzing MRI...
[Findings ready.]
[Coverage note] The loaded images don't cover every modality. Conditions not
visible in this input are dropped before the chat model sees them, so the reply
only covers what the slices can show:
    - no sagittal_t1 slice → neural foraminal narrowing (lf / rf) not assessable
    - no axial_t2 slice → subarticular stenosis (ls / rs) not assessable
    Load the full study (e.g. /sample) for a complete read.

Assistant: From this sagittal T2 slice I can assess the spinal canal. There is
moderate canal stenosis at L4/L5 ... (foraminal and subarticular findings are
not covered because those views weren't provided.)
```

> **Say:** "I only gave it one sagittal slice. The system drops the conditions
> that slice can't show — foramina and subarticular recesses — so the assistant
> only talks about the canal. The model isn't hallucinating about anatomy it
> can't see; it's scoped to its input. Load the full study and all 25 labels
> come back. This is how you'd scope reliability per-input in a real deployment."

`/findings` still shows the raw, unfiltered 25-label JSON (the model's full
output) if you want to contrast what the model *emitted* vs what was *fed to the
chat layer*. The filtering and coverage note apply automatically on any
`/analyze` or auto-analysis whenever a modality is missing.

### Step 7 — Wrap up

```
You: /quit
```

## Sample prompts that work well

These are tuned to show off the chat layer's grounding without straying into hallucination territory.

**Clinical interpretation:**
- "Explain the L4/L5 finding to a referring physician."
- "Which findings would you flag to the surgical team?"
- "Is the canal stenosis severe enough to consider decompression?"

**Cross-level reasoning:**
- "Compare the burden of disease at the upper vs lower lumbar levels."
- "Which level has the most concerning combination of findings?"

**Communication-focused:**
- "Write a one-paragraph radiology report based on these findings."
- "How would you explain this to the patient in plain English?"

**Negative checks (also useful — shows grounding):**
- "Is there evidence of fracture?" (answer: it shouldn't claim there is, because fractures aren't in the schema)
- "What about disc desiccation?" (same — out of scope)

## What to expect from the chat layer

The base MedGemma is a medical VLM, so it's trained on radiology language. Strengths and limits:

**It's good at:**
- Clinical interpretation of the labels you give it
- Plain-language patient explanations
- Compare-and-contrast across levels
- Acknowledging when something isn't in the findings

**It can struggle with:**
- Long multi-turn reasoning over many messages — keep follow-ups focused
- Anything outside the lumbar spine context — it'll either decline or hallucinate
- Numerical scoring (don't ask it to compute Kappa or accuracy)

## Performance / timing

| Action | Approx. time |
|---|---|
| Initial model load (warm HF cache) | 30–60 s |
| `/analyze` (3 images, 100 JSON tokens) | 8–15 s |
| Chat reply (200–400 tokens) | 5–12 s |

On a 24 GB GPU, expect ~10% slower numbers than an A100. Cold HF cache (fresh pod) adds 1–3 min to the first model load while base MedGemma downloads.

## Troubleshooting

| Symptom | Fix |
|---|---|
| OOM on model load | You're on a 16 GB card. Either swap to a 24 GB+ GPU, or run one model at a time (see "Single-model fallback" below). |
| `/analyze` returns invalid JSON | Should never happen on the fine-tuned model — parse failure rate is 0% on validation. If it does, your adapter path is probably pointing at the base model. |
| Chat layer ignores the findings | The system prompt is short by design. If you see hallucinations, check that `chat_reply()` is actually receiving the latest findings — `/findings` should print non-empty JSON before chatting. |
| Cold model load takes > 5 min | First-run download of base MedGemma. Subsequent runs use the HF cache and are fast. |
| `tokenizer has no attribute 'tokenizer'` | Old Unsloth version. Upgrade via `pip install --upgrade unsloth unsloth_zoo`. |

## Single-model fallback (if VRAM is tight)

If you can't fit both models, comment out the chat model load and use the deterministic formatter for the response instead. Less impressive demo but still works on any 16 GB+ GPU:

```python
# In demo_chat.py main(), replace the chat_reply call with:
intro = json_to_report(findings)
```

You lose conversational follow-up but keep the structured-to-natural-language transition.

## Architecture deep-dive (for technical audiences)

If your audience is technical, this is the right level of detail to walk through:

1. **Vision encoder:** MedGemma uses SigLIP at patch size 14. At 448² resolution that's `(448/14)² = 1024` patch tokens per image.
2. **Token budget:** `demo_a100_40g` uses up to 8 images → ~8192 image tokens, leaving ~1024 tokens for the prompt + JSON response within `max_seq_length=9216`.
3. **LoRA targets:** r=16 adapters on `q/k/v/o_proj` + `gate/up/down_proj` in the language layers, and on the SigLIP vision tower too.
4. **SFT format:** chat-template with `<image>` tokens × N + text prompt → compact JSON assistant turn. UnslothVisionDataCollator masks user-turn labels to -100 so the model only learns to predict the assistant response.
5. **Why two models in the demo:** the fine-tune narrows the model's output distribution toward JSON. To recover natural-language ability we use the base model — same architecture, no LoRA. Both share the HF cache so we only download once.

## Files this demo touches

| Path | Read | Written |
|---|---|---|
| `/workspace/models/medgemma-lumbar/final/` | yes (adapter) | no |
| `~/.cache/huggingface/` | yes (base + adapter cache) | yes (first run only) |
| `/workspace/data/processed/val_dataset.jsonl` | yes (for `/sample`) | no |
| `/workspace/data/processed/*.png` | yes (any PNG you `/load`) | no |

No outputs are written by the demo — it's a read-only inference workflow.
