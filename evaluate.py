# evaluate.py
"""
Evaluates fine-tuned MedGemma on RSNA 2024 validation set.
"""

import os
import re
import json
import functools
from typing import Optional
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for RunPod
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    cfg,
    COND_KEYS,
    LEVEL_KEYS,
    SEVERITY_FROM_CODE,
)
from data_prep import build_user_prompt


def load_model(model_path: str, use_finetuned: bool = True):
    """Load model for evaluation."""

    if use_finetuned:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_path,
            max_seq_length=cfg.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
            token=os.environ.get("HF_TOKEN"),
        )
        FastVisionModel.for_inference(model)
    else:
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
            token=os.environ.get("HF_TOKEN"),
        )
        FastVisionModel.for_inference(model)
    
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_path if use_finetuned else cfg.model_name,
        token=os.environ.get("HF_TOKEN")
    )
    
    return model, tokenizer, processor


def run_inference(model, tokenizer, processor, image_paths, prompt: str) -> str:
    """
    Run inference on one or more images + a text prompt.

    BUG FIX v3.0: The previous signature accepted a single image_path (str).
    The dataset stores image_paths (list), one path per image token in the
    user message. This function now accepts either a str or a list[str].
    """
    # Normalise to list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    images = []
    for p in image_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"  [WARN] Could not load image {p}: {e}")

    if not images:
        return ""

    # Build message with one {"type": "image"} entry per loaded image
    user_content = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        images=images,
        text=text,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,   # BUG FIX v3.0: removed temperature=0.1 which
                               # is ignored (and misleading) with do_sample=False
        )

    # Decode only the new tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


@functools.lru_cache(maxsize=4096)
def _extract_label_json(response: str) -> Optional[dict]:
    """
    Pull the severity matrix out of a model response. The training template
    asks the model to emit a single JSON object; we tolerate optional code
    fences and minor surrounding chatter. lru_cache amortises parsing across
    the 25 condition×level lookups per sample.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    # Also try the first balanced {...} block in the raw response.
    start = response.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(response)):
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(response[start:i + 1])
                    break
    for blob in candidates:
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def parse_severity_from_response(response: str, condition: str, level: str) -> str:
    """
    Extract predicted severity for one (condition, level) pair from a JSON
    response. Falls back to "Normal/Mild" (majority class) when the response
    is unparseable or missing the requested key.
    """
    parsed = _extract_label_json(response)
    if not parsed:
        return "Normal/Mild"

    level_key = LEVEL_KEYS.get(level)
    cond_key = COND_KEYS.get(condition)
    if level_key is None or cond_key is None:
        return "Normal/Mild"

    level_data = parsed.get(level_key)
    if not isinstance(level_data, dict):
        return "Normal/Mild"

    code = level_data.get(cond_key)
    if code is None:
        return "Normal/Mild"

    return SEVERITY_FROM_CODE.get(str(code).strip().upper(), "Normal/Mild")


def _load_val_dataset() -> list:
    """
    Load the validation set, preferring JSONL over legacy JSON.
    """
    base = Path(cfg.processed_data_dir)
    jsonl = base / "val_dataset.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            return [json.loads(line) for line in f if line.strip()]
    legacy = base / "val_dataset.json"
    with open(legacy) as f:
        return json.load(f)


def evaluate_model(model_path: str, use_finetuned: bool = True,
                   max_samples: int = None) -> tuple:
    """Run full evaluation on validation set."""
    
    model, tokenizer, processor = load_model(model_path, use_finetuned)

    val_data = _load_val_dataset()

    if max_samples:
        val_data = val_data[:max_samples]

    print(f"Evaluating on {len(val_data)} samples...")

    results = []

    for sample in tqdm(val_data):
        # BUG FIX v3.0: dataset stores image_paths (list), not image_path (str)
        image_paths = sample.get("image_paths", sample.get("image_path", []))
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Get ground truth from messages
        gt_response = sample["messages"][-1]["content"]

        # Use the same prompt the model was trained on. Paraphrasing here
        # silently degrades eval quality, so we reconstruct via the shared
        # helper instead of hardcoding the wording.
        eval_prompt = build_user_prompt(
            n_images=len(image_paths),
            series_types=sample.get("series_types"),
        )

        # Run inference
        pred_response = run_inference(
            model, tokenizer, processor, image_paths, eval_prompt
        )
        
        # Parse predictions for each condition+level
        sample_results = {
            "study_id": sample["study_id"],
            "gt_response": gt_response,
            "pred_response": pred_response,
            "predictions": {},
            "ground_truth": {}
        }
        
        for cond in cfg.conditions:
            for level in cfg.levels:
                key = f"{cond}_{level}"
                
                # Parse ground truth
                gt_sev = parse_severity_from_response(gt_response, cond, level)
                pred_sev = parse_severity_from_response(pred_response, cond, level)
                
                sample_results["ground_truth"][key] = gt_sev
                sample_results["predictions"][key] = pred_sev
        
        results.append(sample_results)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    return metrics, results


def rsna_weighted_score_proxy(
    all_gt: list,
    all_pred_probs: list,
    weights: list = None,
) -> float:
    """
    Compute a *proxy* for the RSNA 2024 weighted log-loss.

    The real competition metric is a probabilistic log-loss with class
    weights [1, 2, 4]. A generative LM does not expose calibrated soft
    probabilities, so we currently feed hard one-hot pseudo-probabilities
    (see `severity_to_one_hot`). After clipping to ε that means each row
    contributes either ≈ -log(1-ε) ≈ 0 (correct) or -log(ε) ≈ 16.1
    (incorrect) — i.e. this number is fundamentally a weighted error rate
    rescaled by 16.1, not a probabilistic log-loss.

    Treat it as a proxy. To get the real metric, expose per-severity-token
    logits from the LM and pass real softmax probabilities here.

    Args:
        all_gt:        list of int ground-truth class indices (0/1/2)
        all_pred_probs: list of [p_normal, p_moderate, p_severe] arrays.
        weights:       per-class weights; defaults to cfg.rsna_loss_weights

    Returns:
        scalar weighted-error proxy (lower is better)
    """
    if weights is None:
        weights = cfg.rsna_loss_weights  # [1.0, 2.0, 4.0]

    eps = 1e-7
    total_loss = 0.0
    total_weight = 0.0

    for gt_idx, pred_probs in zip(all_gt, all_pred_probs):
        pred_probs = np.array(pred_probs, dtype=np.float64)
        pred_probs = np.clip(pred_probs, eps, 1 - eps)
        pred_probs /= pred_probs.sum()

        row_loss = -np.log(pred_probs[gt_idx])
        row_weight = weights[gt_idx]

        total_loss += row_weight * row_loss
        total_weight += row_weight

    return total_loss / total_weight if total_weight > 0 else float("inf")


# Backwards-compatible alias so existing imports / dashboards keep working.
rsna_weighted_log_loss = rsna_weighted_score_proxy


def weighted_accuracy(all_gt: list, all_pred: list, weights: list = None) -> float:
    """
    Class-weighted accuracy — a more honest summary of a hard-prediction
    setup than the log-loss proxy above. Each correct prediction is worth
    weights[gt_class]; the result is sum(correct·weight) / sum(weight).
    """
    if weights is None:
        weights = cfg.rsna_loss_weights
    num, den = 0.0, 0.0
    for gt, pred in zip(all_gt, all_pred):
        w = weights[gt]
        den += w
        if gt == pred:
            num += w
    return num / den if den > 0 else 0.0


def severity_to_one_hot(severity_str: str) -> list:
    """
    Convert a predicted severity string to a hard probability vector.

    For a generative model that outputs text, we treat the parsed label as
    a hard prediction (prob=1 for predicted class, 0 for others). This allows
    using rsna_weighted_log_loss as an approximation; for calibrated soft
    probabilities a dedicated classification head would be needed.
    """
    mapping = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    idx = mapping.get(severity_str, 0)
    probs = [0.0, 0.0, 0.0]
    probs[idx] = 1.0
    return probs


def compute_metrics(results: list) -> dict:
    """Compute comprehensive evaluation metrics."""
    metrics = {}
    
    all_keys = [f"{c}_{l}" for c in cfg.conditions for l in cfg.levels]
    
    overall_gt = []
    overall_pred = []
    
    condition_metrics = {}
    
    for key in all_keys:
        gt_list = [r["ground_truth"].get(key, "Normal/Mild") for r in results]
        pred_list = [r["predictions"].get(key, "Normal/Mild") for r in results]
        
        gt_encoded = [cfg.severity_labels.index(g) for g in gt_list]
        pred_encoded = [cfg.severity_labels.index(p) for p in pred_list]
        
        overall_gt.extend(gt_encoded)
        overall_pred.extend(pred_encoded)
        
        condition_metrics[key] = {
            "accuracy": accuracy_score(gt_encoded, pred_encoded),
            "f1_weighted": f1_score(gt_encoded, pred_encoded, average="weighted", zero_division=0),
            "kappa": cohen_kappa_score(gt_encoded, pred_encoded, weights="quadratic"),
        }
    
    # Overall metrics
    metrics["overall_accuracy"] = accuracy_score(overall_gt, overall_pred)
    metrics["overall_f1_weighted"] = f1_score(overall_gt, overall_pred, average="weighted", zero_division=0)
    metrics["overall_kappa"] = cohen_kappa_score(overall_gt, overall_pred, weights="quadratic")

    # Hard-prediction proxy for the RSNA weighted log-loss. The "log_loss"
    # variant is kept as an alias for any downstream dashboards still using
    # that key, but `rsna_weighted_score_proxy` is the honest name.
    all_pred_probs = [severity_to_one_hot(cfg.severity_labels[p]) for p in overall_pred]
    metrics["rsna_weighted_score_proxy"] = rsna_weighted_score_proxy(
        overall_gt, all_pred_probs, weights=cfg.rsna_loss_weights
    )
    metrics["rsna_weighted_log_loss"] = metrics["rsna_weighted_score_proxy"]
    metrics["weighted_accuracy"] = weighted_accuracy(
        overall_gt, overall_pred, weights=cfg.rsna_loss_weights
    )

    metrics["condition_metrics"] = condition_metrics

    # Per-condition summary
    for cond in cfg.conditions:
        cond_keys = [f"{cond}_{l}" for l in cfg.levels]
        metrics[f"{cond}_avg_accuracy"] = np.mean([condition_metrics[k]["accuracy"] for k in cond_keys])
        metrics[f"{cond}_avg_kappa"] = np.mean([condition_metrics[k]["kappa"] for k in cond_keys])

    # Parse-failure rate: fraction of samples where the predicted response
    # could not be parsed as JSON. `parse_severity_from_response` silently
    # falls back to "Normal/Mild" for these, so without this counter an
    # unparseable base-model run looks accidentally accurate by class prior.
    n_parse_fail = 0
    for r in results:
        pred_text = r.get("pred_response", "")
        if _extract_label_json(pred_text) is None:
            n_parse_fail += 1
    metrics["parse_failure_rate"] = (
        n_parse_fail / len(results) if results else 0.0
    )
    metrics["parse_failure_count"] = n_parse_fail

    return metrics


def plot_confusion_matrices(results: list, output_dir: str, prefix: str = ""):
    """Generate confusion matrices for each condition-level pair."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(5, 5, figsize=(25, 25))
    
    for i, cond in enumerate(cfg.conditions):
        for j, level in enumerate(cfg.levels):
            key = f"{cond}_{level}"
            
            gt_list = [r["ground_truth"].get(key, "Normal/Mild") for r in results]
            pred_list = [r["predictions"].get(key, "Normal/Mild") for r in results]
            
            cm = confusion_matrix(gt_list, pred_list, labels=cfg.severity_labels)
            
            ax = axes[i][j]
            sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                       xticklabels=["N/M", "Mod", "Sev"],
                       yticklabels=["N/M", "Mod", "Sev"],
                       cmap="Blues")
            
            level_name = level.upper().replace("_", "/")
            cond_short = cond.replace("_", "\n")
            ax.set_title(f"{level_name}\n{cond_short}", fontsize=8)
            ax.set_xlabel("Predicted", fontsize=7)
            ax.set_ylabel("True", fontsize=7)
    
    plt.suptitle(f"{prefix}Confusion Matrices — All Conditions × Levels", 
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    save_path = output_path / f"{prefix}confusion_matrices.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Confusion matrices saved to: {save_path}")


def print_metrics_summary(metrics: dict, title: str = "Evaluation Results"):
    """Print formatted metrics summary."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"Overall Accuracy:          {metrics['overall_accuracy']:.4f}")
    print(f"Overall F1 (weighted):     {metrics['overall_f1_weighted']:.4f}")
    print(f"Overall Kappa (QW):        {metrics['overall_kappa']:.4f}")
    print(f"Weighted Accuracy:         {metrics.get('weighted_accuracy', float('nan')):.4f}  (class-weighted, higher is better)")
    print(f"RSNA Weighted Score Proxy: {metrics.get('rsna_weighted_score_proxy', float('nan')):.4f}  (proxy for log-loss, lower is better)")
    print(f"Parse Failure Rate:        {metrics.get('parse_failure_rate', 0.0):.4f}  "
          f"({metrics.get('parse_failure_count', 0)} unparseable responses → fell back to Normal/Mild)")
    print(f"\nPer-Condition Summary:")
    for cond in cfg.conditions:
        print(f"  {cond.replace('_', ' ').title():<40} "
              f"Acc: {metrics[f'{cond}_avg_accuracy']:.3f}  "
              f"Kappa: {metrics[f'{cond}_avg_kappa']:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=cfg.output_dir + "/final")
    # BooleanOptionalAction so `--no-finetuned` actually works. The previous
    # `action="store_true", default=True` left the flag permanently True with
    # no way to evaluate the base model from CLI.
    parser.add_argument(
        "--finetuned",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.environ.get("WORKSPACE", "/workspace"), "logs", "evaluation_results"),
    )
    args = parser.parse_args()
    
    print(f"\nEvaluating {'fine-tuned' if args.finetuned else 'base'} model...")
    
    metrics, results = evaluate_model(
        model_path=args.model_path,
        use_finetuned=args.finetuned,
        max_samples=args.max_samples
    )
    
    print_metrics_summary(metrics, title="Fine-tuned Model Evaluation")
    plot_confusion_matrices(results, args.output_dir, prefix="finetuned_")
    
    # Save detailed results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "metrics.json", "w") as f:
        # Filter out non-serializable
        clean_metrics = {k: v for k, v in metrics.items() 
                        if k != "condition_metrics"}
        json.dump(clean_metrics, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")