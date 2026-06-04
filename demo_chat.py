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
    #   /load <png_path>   add an image to the next analysis
    #   /sample            auto-load 3 images from val_dataset.jsonl
    #   /analyze           run the fine-tuned model on loaded images
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


# ─── Chat stage ─────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a clinical radiology assistant explaining MRI findings to a physician.

A specialist vision model has analyzed the patient's lumbar spine MRI and produced the structured findings below. Use ONLY these findings when answering questions — do not invent additional findings. If asked about something that is not in the findings, say so.

Severity scale: normal or mild → moderate → severe.

Findings:
{findings_block}
"""


def chat_reply(chat_model, chat_tok, findings: dict, history: list, question: str) -> str:
    """Generate a natural-language reply using the base model + the findings as context."""
    system_text = SYSTEM_TEMPLATE.format(
        findings_block=findings_to_context_block(findings),
    )

    # Build a flat text-only conversation. MedGemma's chat template handles
    # role tags; we just hand it system + history + new question.
    messages = [{"role": "user", "content": [{"type": "text", "text": system_text}]}]
    for turn in history:
        messages.append(turn)
    messages.append({"role": "user", "content": [{"type": "text", "text": question}]})

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

def load_sample_from_val(n: int = 3) -> List[str]:
    """Pull the image paths from the first val example as a quick demo set."""
    val_path = Path(cfg.processed_data_dir) / "val_dataset.jsonl"
    with open(val_path) as f:
        sample = json.loads(f.readline())
    return sample["image_paths"][:n]


def print_banner() -> None:
    print()
    print("=" * 62)
    print(" Lumbar Spine MRI Assistant — interactive demo")
    print("=" * 62)
    print("Commands:")
    print("  /load <path>   add an MRI image (PNG)")
    print("  /sample        load a sample study from the val set")
    print("  /analyze       run the diagnostic model on loaded images")
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

        if user == "/sample":
            try:
                samples = load_sample_from_val(n=3)
            except Exception as e:
                print(f"[Could not load sample: {e}]\n")
                continue
            image_paths.extend(samples)
            print(f"[Loaded {len(samples)} sample image(s) from val set:]")
            for s in samples:
                print(f"    - {s}")
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
            # Kick off the conversation with an auto-summary
            print("Generating summary...")
            intro = chat_reply(chat_model, chat_tok, findings, [],
                               "Briefly summarise the key findings for a referring physician.")
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

        # Free-form chat
        if findings is None:
            print("[Run /analyze first so I have findings to talk about.]\n")
            continue

        try:
            reply = chat_reply(chat_model, chat_tok, findings, history, user)
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
