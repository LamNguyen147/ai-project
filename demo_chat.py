"""Conversational demo for the lumbar spine fine-tune.

Architecture:
    [MRI image(s)] → (fine-tuned MedGemma)  → JSON findings  ┐
                                                              ├→ (base MedGemma chat) → reply
                                  user question + history  ───┘

The fine-tuned model is a specialist that only outputs the 25-label JSON.
The base model handles natural conversation about those structured findings.

Usage:
    python demo_chat.py
    # then inside the REPL:
    #   /load <png_path>   add a single image to the next analysis
    #   /study <study_id>  load every slice of one study at once
    #   /sample            auto-load a study from val_dataset.jsonl
    #   /analyze           run the fine-tuned model on loaded images
    #   /compare           run fine-tuned AND base model on the same images
    #   /base <question>   ask the base model directly about the images
    #   /report            print the deterministic radiology-style report
    #   /findings          dump the raw JSON findings
    #   /clear             reset images + chat history
    #   /quit              exit
    #   <anything else>    chat question about the current findings
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from unsloth import FastVisionModel

from config import cfg
from data_prep import build_user_prompt
from json_to_report import json_to_report, findings_to_context_block


# ─── Model loading ──────────────────────────────────────────────────────────

def load_models(adapter_path: str, base_model: str):
    """Load both the fine-tuned specialist and the base chat model."""
    print(f"[1/2] Loading fine-tuned diagnostic model from {adapter_path}...")
    diag_model, diag_tok = FastVisionModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(diag_model)

    print(f"[2/2] Loading base MedGemma chat model from {base_model}...")
    chat_model, chat_tok = FastVisionModel.from_pretrained(
        model_name=base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(chat_model)
    return diag_model, diag_tok, chat_model, chat_tok


# ─── Diagnostic stage ───────────────────────────────────────────────────────

def infer_series_type(path: str) -> str:
    """Pull the series type from the filename, default to sagittal_t2."""
    name = path.lower()
    for s in ("sagittal_t2", "sagittal_t1", "axial_t2"):
        if s in name:
            return s
    return "sagittal_t2"


# Which modality is needed to *visually* assess each condition group. The model
# always emits all 25 labels, but a label is only grounded in the image when the
# modality that shows that anatomy is present in the input.
MODALITY_COVERAGE = {
    "sagittal_t2": "spinal canal stenosis (canal)",
    "sagittal_t1": "neural foraminal narrowing (lf / rf)",
    "axial_t2":    "subarticular stenosis (ls / rs)",
}

# Compact condition keys (see config.COND_KEYS) each modality can assess.
COND_BY_MODALITY = {
    "sagittal_t2": {"canal"},
    "sagittal_t1": {"lf", "rf"},
    "axial_t2":    {"ls", "rs"},
}


def assessable_conditions(image_paths: List[str]) -> set:
    """Condition keys the loaded modalities can actually visualise."""
    present = {infer_series_type(p) for p in image_paths}
    keep = set()
    for m in present:
        keep |= COND_BY_MODALITY.get(m, set())
    return keep


def filter_findings(findings: dict, keep_conds: set) -> dict:
    """Drop condition keys not visually assessable by the loaded modalities, so
    the base model only ever talks about anatomy the input can actually show."""
    return {
        level: {c: s for c, s in conds.items() if c in keep_conds}
        for level, conds in findings.items()
    }


def coverage_warning(image_paths: List[str]) -> Optional[str]:
    """Warn if the loaded images miss a modality, so the off-view labels are
    flagged as not visually supported (rather than looking like hallucination)."""
    present = {infer_series_type(p) for p in image_paths}
    missing = [m for m in MODALITY_COVERAGE if m not in present]
    if not missing:
        return None
    lines = ["[Coverage note] The loaded images don't cover every modality. "
             "Conditions not visible in this input are dropped before the chat "
             "model sees them, so the reply only covers what the slices can show:"]
    for m in missing:
        lines.append(f"    - no {m} slice → {MODALITY_COVERAGE[m]} not assessable")
    lines.append("    Load the full study (e.g. /sample) for a complete read.")
    return "\n".join(lines)


def run_diagnosis(diag_model, diag_tok, image_paths: List[str]) -> dict:
    """Run the fine-tuned model on the images and return parsed JSON findings."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    series_types = [infer_series_type(p) for p in image_paths]

    user_prompt = build_user_prompt(len(images), series_types=series_types)
    messages = [{"role": "user", "content": [
        *[{"type": "image"} for _ in images],
        {"type": "text", "text": user_prompt},
    ]}]
    text = diag_tok.apply_chat_template(messages, add_generation_prompt=True)
    inputs = diag_tok(images, text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = diag_model.generate(
            **inputs, max_new_tokens=200, do_sample=False, use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    response = diag_tok.batch_decode(out[:, input_len:], skip_special_tokens=True)[0]
    response = response[response.find("{") : response.rfind("}") + 1]
    return json.loads(response)


def run_base_vision(chat_model, chat_tok, image_paths: List[str],
                    prompt: Optional[str] = None, max_new_tokens: int = 300) -> str:
    """Run the BASE (un-fine-tuned) model on the same images, returning its raw
    text. Used for side-by-side comparison: the base model usually won't emit
    the compact 25-label JSON — it narrates, hedges, or refuses — which is the
    whole point of showing what fine-tuning bought us. `prompt=None` reuses the
    exact diagnostic prompt the fine-tuned model sees; pass a string to ask the
    base model a free-form question about the images instead."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    series_types = [infer_series_type(p) for p in image_paths]
    if prompt is None:
        prompt = build_user_prompt(len(images), series_types=series_types)

    messages = [{"role": "user", "content": [
        *[{"type": "image"} for _ in images],
        {"type": "text", "text": prompt},
    ]}]
    text = chat_tok.apply_chat_template(messages, add_generation_prompt=True)
    inputs = chat_tok(images, text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = chat_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    return chat_tok.batch_decode(out[:, input_len:], skip_special_tokens=True)[0].strip()


# ─── Chat stage ─────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a clinical radiology assistant explaining MRI findings to a physician.

A specialist vision model has analyzed the patient's lumbar spine MRI and produced the structured findings below. Use ONLY these findings when answering questions — do not invent additional findings. If asked about something that is not in the findings, say so.

Severity scale: normal or mild → moderate → severe.

Findings:
{findings_block}
"""


def chat_reply(chat_model, chat_tok, findings: dict, history: list, question: str,
               image_paths: Optional[List[str]] = None) -> str:
    """Generate a natural-language reply using the base model + the findings as
    context. If `image_paths` is given, the findings are first filtered down to
    the conditions those modalities can actually assess, so the chat model never
    discusses anatomy the input can't show."""
    if image_paths is not None:
        findings = filter_findings(findings, assessable_conditions(image_paths))
    system_text = SYSTEM_TEMPLATE.format(
        findings_block=findings_to_context_block(findings),
    )

    # Gemma/MedGemma chat templates have NO system role and require strictly
    # alternating user/assistant turns. So we can't add the findings context as
    # its own turn (that would be two user turns in a row when history is empty
    # → "roles must alternate"). Instead, fold the context into the first user
    # turn's text. We rebuild messages fresh each call and only replace the
    # local copy of messages[0], so `history` itself is never mutated.
    messages = list(history)
    messages.append({"role": "user", "content": [{"type": "text", "text": question}]})

    first_text = messages[0]["content"][0]["text"]
    messages[0] = {"role": "user",
                   "content": [{"type": "text", "text": f"{system_text}\n\n{first_text}"}]}

    text = chat_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Bypass the vision branch for pure text chat. processor.tokenizer is the
    # plain text tokenizer; the processor itself expects an image arg.
    tok = chat_tok.tokenizer if hasattr(chat_tok, "tokenizer") else chat_tok
    inputs = tok(text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = chat_model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    reply = tok.decode(out[0, input_len:], skip_special_tokens=True).strip()
    return reply


# ─── Helpers ────────────────────────────────────────────────────────────────

SEVERITY_FILTERS = {"any": 0, "moderate": 1, "severe": 2}


def load_sample_from_val(severity: str = "any"):
    """Pick a val study, optionally one that contains Moderate/Severe labels.

    The dataset is ~85% Normal/Mild, so the first row is almost always all-normal
    and proves nothing in a demo. Each val row carries a `max_severity` field
    (0=Normal/Mild only, 1=has Moderate, 2=has Severe); we scan for the first row
    at or above the requested level. Returns (image_paths, study_id, gt_findings).
    All of the study's images are loaded (not just 3) so the model gets the same
    multi-modality input it was trained on — important for an honest prediction.
    """
    target = SEVERITY_FILTERS[severity]
    val_path = Path(cfg.processed_data_dir) / "val_dataset.jsonl"
    with open(val_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("max_severity", 0) >= target:
                gt = json.loads(row["messages"][-1]["content"])
                return row["image_paths"], row["study_id"], gt
    raise ValueError(f"No val study found with severity >= {severity!r}.")


def load_study_images(study_id: str):
    """Load every slice for one study. Prefers the exact image_paths recorded in
    the dataset (val first, then train) so the set matches what the model trained
    on; falls back to globbing `<study_id>_*.png` in the processed dir if the
    study isn't in either JSONL. Returns (image_paths, gt_findings_or_None)."""
    proc = Path(cfg.processed_data_dir)
    for name in ("val_dataset.jsonl", "train_dataset.jsonl"):
        jsonl = proc / name
        if not jsonl.exists():
            continue
        with open(jsonl) as f:
            for line in f:
                row = json.loads(line)
                if str(row["study_id"]) == str(study_id):
                    gt = json.loads(row["messages"][-1]["content"])
                    return row["image_paths"], gt
    # Fallback: glob the flat PNG layout (<study_id>_<series>_<type>_sliceN.png)
    paths = sorted(str(p) for p in proc.glob(f"{study_id}_*.png"))
    if not paths:
        raise ValueError(f"No images found for study {study_id!r} in {proc}.")
    return paths, None


def print_banner() -> None:
    print()
    print("=" * 62)
    print(" Lumbar Spine MRI Assistant — interactive demo")
    print("=" * 62)
    print("Commands:")
    print("  /load <path>   add a single MRI image (PNG)")
    print("  /study <id>    load every slice of one study at once")
    print("  /sample [sev]  load a sample study from the val set")
    print("                 sev = any | moderate | severe (default any)")
    print("  /analyze       run the diagnostic model on loaded images")
    print("  /compare       run fine-tuned AND base model on the same images")
    print("  /base <q>      ask the base model directly about the images")
    print("  /report        show the deterministic radiology report")
    print("  /findings      show the raw JSON findings")
    print("  /state         show what's currently loaded")
    print("  /clear         reset images + chat history")
    print("  /quit          exit")
    print("Anything else is a chat question about the current findings.")
    print()


# ─── Main REPL ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="/workspace/models/medgemma-lumbar/final",
                        help="LoRA adapter path or HF repo id")
    parser.add_argument("--base", default="unsloth/medgemma-1.5-4b-it",
                        help="Base chat model")
    args = parser.parse_args()

    diag_model, diag_tok, chat_model, chat_tok = load_models(args.adapter, args.base)
    print_banner()

    image_paths: List[str] = []
    findings: Optional[dict] = None
    history: list = []

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        if user == "/quit":
            break

        if user == "/clear":
            image_paths.clear()
            findings = None
            history = []
            print("[Cleared images + chat history.]\n")
            continue

        if user == "/state":
            print(f"  images loaded:  {len(image_paths)}")
            for p in image_paths:
                print(f"    - {p}")
            print(f"  findings ready: {bool(findings)}")
            print(f"  chat turns:     {len(history) // 2}\n")
            continue

        if user.startswith("/load "):
            p = user[6:].strip()
            if not Path(p).exists():
                print(f"[Error: {p} not found.]\n")
                continue
            image_paths.append(p)
            print(f"[Loaded image #{len(image_paths)}: {Path(p).name} "
                  f"(series: {infer_series_type(p)})]\n")
            continue

        if user.split()[0] == "/sample":
            parts = user.split()
            sev = parts[1].lower() if len(parts) > 1 else "any"
            if sev not in SEVERITY_FILTERS:
                print(f"[Usage: /sample [{' | '.join(SEVERITY_FILTERS)}]]\n")
                continue
            try:
                samples, study_id, gt = load_sample_from_val(severity=sev)
            except Exception as e:
                print(f"[Could not load sample: {e}]\n")
                continue
            image_paths.extend(samples)
            print(f"[Loaded {len(samples)} image(s) from study {study_id} "
                  f"(filter: {sev}):]")
            for s in samples:
                print(f"    - {s}")
            # Show ground-truth Moderate/Severe labels so you know what the model
            # *should* find. Handy for rehearsal; ignore it during the live show.
            gt_ms = [f"{lvl} {cond}={s2}"
                     for lvl, conds in gt.items()
                     for cond, s2 in conds.items() if s2 in ("M", "S")]
            if gt_ms:
                print(f"  Ground-truth abnormal labels: {', '.join(gt_ms)}")
            else:
                print("  Ground truth: all Normal/Mild.")
            print()
            continue

        if user.startswith("/study"):
            parts = user.split()
            if len(parts) < 2:
                print("[Usage: /study <study_id>]\n")
                continue
            study_id = parts[1]
            try:
                study_imgs, gt = load_study_images(study_id)
            except Exception as e:
                print(f"[Could not load study: {e}]\n")
                continue
            image_paths.extend(study_imgs)
            mods = sorted({infer_series_type(p) for p in study_imgs})
            print(f"[Loaded {len(study_imgs)} image(s) for study {study_id} "
                  f"— modalities: {', '.join(mods)}:]")
            for s in study_imgs:
                print(f"    - {s}")
            if gt is not None:
                gt_ms = [f"{lvl} {cond}={s2}"
                         for lvl, conds in gt.items()
                         for cond, s2 in conds.items() if s2 in ("M", "S")]
                print(f"  Ground-truth abnormal labels: "
                      f"{', '.join(gt_ms) if gt_ms else 'all Normal/Mild'}")
            print()
            continue

        if user == "/analyze":
            if not image_paths:
                print("[No images. Use /load <path> or /sample first.]\n")
                continue
            print("Analyzing MRI...")
            try:
                findings = run_diagnosis(diag_model, diag_tok, image_paths)
            except Exception as e:
                print(f"[Diagnosis failed: {e}]\n")
                continue
            print("[Findings ready.]\n")
            # Coverage note hidden for now — findings are still filtered to the
            # assessable conditions in chat_reply(); re-enable to surface it:
            #   warn = coverage_warning(image_paths)
            #   if warn:
            #       print(warn + "\n")
            # Kick off the conversation with an auto-summary
            print("Generating summary...")
            intro = chat_reply(chat_model, chat_tok, findings, [],
                               "Briefly summarise the key findings for a referring physician.",
                               image_paths=image_paths)
            print(f"\nAssistant: {intro}\n")
            history.append({"role": "user",
                            "content": [{"type": "text",
                                         "text": "Briefly summarise the key findings for a referring physician."}]})
            history.append({"role": "assistant",
                            "content": [{"type": "text", "text": intro}]})
            continue

        if user == "/report":
            if findings is None:
                print("[Run /analyze first.]\n")
                continue
            print()
            print(json_to_report(findings))
            print()
            continue

        if user == "/findings":
            if findings is None:
                print("[Run /analyze first.]\n")
                continue
            print(json.dumps(findings, indent=2))
            print()
            continue

        if user == "/compare":
            if not image_paths:
                print("[No images. Use /load <path> or /sample first.]\n")
                continue
            print("Running fine-tuned specialist on the images...")
            try:
                ft = run_diagnosis(diag_model, diag_tok, image_paths)
                ft_out = json.dumps(ft)
            except Exception as e:
                ft_out = f"[no parseable JSON: {e}]"
            print("Running base MedGemma on the SAME images + prompt...")
            try:
                base_out = run_base_vision(chat_model, chat_tok, image_paths)
            except Exception as e:
                base_out = f"[generation failed: {e}]"
            print("\n" + "=" * 62)
            print(" Fine-tuned specialist (LoRA) — compact 25-label JSON")
            print("=" * 62)
            print(ft_out)
            print("\n" + "=" * 62)
            print(" Base MedGemma (no fine-tuning) — same input, raw output")
            print("=" * 62)
            print(base_out)
            print()
            continue

        if user.startswith("/base "):
            if not image_paths:
                print("[No images. Use /load <path> or /sample first.]\n")
                continue
            question = user[len("/base "):].strip()
            if not question:
                print("[Usage: /base <question about the images>]\n")
                continue
            print("Asking base MedGemma directly (it sees the images)...")
            try:
                base_out = run_base_vision(chat_model, chat_tok, image_paths,
                                           prompt=question, max_new_tokens=400)
            except Exception as e:
                base_out = f"[generation failed: {e}]"
            print(f"\nBase model: {base_out}\n")
            continue

        # Free-form chat. If images are loaded but not yet analyzed, run the
        # diagnosis automatically so the demo flows naturally: load an image,
        # then just ask "what could this be?" without a separate /analyze step.
        if findings is None:
            if not image_paths:
                print("[No image yet. Use /load <path> or /sample first, "
                      "then ask your question.]\n")
                continue
            print("Analyzing MRI...")
            try:
                findings = run_diagnosis(diag_model, diag_tok, image_paths)
            except Exception as e:
                print(f"[Diagnosis failed: {e}]\n")
                continue
            print("[Findings ready.]\n")
            # Coverage note hidden for now — findings are still filtered to the
            # assessable conditions in chat_reply(); re-enable to surface it:
            #   warn = coverage_warning(image_paths)
            #   if warn:
            #       print(warn + "\n")

        try:
            reply = chat_reply(chat_model, chat_tok, findings, history, user,
                               image_paths=image_paths)
        except Exception as e:
            print(f"[Chat failed: {e}]\n")
            continue
        print(f"\nAssistant: {reply}\n")
        history.append({"role": "user",
                        "content": [{"type": "text", "text": user}]})
        history.append({"role": "assistant",
                        "content": [{"type": "text", "text": reply}]})

    print("Goodbye.")


if __name__ == "__main__":
    main()
