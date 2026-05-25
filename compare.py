# compare.py
"""
Generates side-by-side comparison of base MedGemma vs fine-tuned model.
"""

import os
import json
from typing import Optional
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import cfg
from data_prep import build_user_prompt
from evaluate import load_model, run_inference, parse_severity_from_response, compute_metrics


def compare_models(
    num_samples: int = 20,
    output_dir: Optional[str] = None,
):
    """Run both models on the same samples and compare outputs."""
    if output_dir is None:
        output_dir = os.path.join(
            os.environ.get("WORKSPACE", "/workspace"), "logs", "comparison"
        )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load validation data (prefers .jsonl, falls back to legacy .json)
    base = Path(cfg.processed_data_dir)
    val_jsonl = base / "val_dataset.jsonl"
    if val_jsonl.exists():
        with open(val_jsonl) as f:
            val_data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(base / "val_dataset.json") as f:
            val_data = json.load(f)
    
    # Select diverse samples
    samples = val_data[:num_samples]

    # --- Run BASE model ---
    print("Loading BASE model (MedGemma without fine-tuning)...")
    base_model, base_tokenizer, base_processor = load_model(
        model_path=cfg.model_name, use_finetuned=False
    )

    base_results = []
    print("Running base model inference...")
    for sample in tqdm(samples):
        # BUG FIX v3.0: use image_paths (list), not image_path (str)
        image_paths = sample.get("image_paths", sample.get("image_path", []))
        # Reconstruct the exact training prompt for this sample.
        eval_prompt = build_user_prompt(
            n_images=len(image_paths) if not isinstance(image_paths, str) else 1,
            series_types=sample.get("series_types"),
        )
        pred = run_inference(base_model, base_tokenizer, base_processor,
                           image_paths, eval_prompt)

        gt = sample["messages"][-1]["content"]

        result = {
            "study_id": sample["study_id"],
            "image_paths": image_paths,
            "ground_truth": {},
            "base_predictions": {}
        }
        
        for cond in cfg.conditions:
            for level in cfg.levels:
                key = f"{cond}_{level}"
                result["ground_truth"][key] = parse_severity_from_response(gt, cond, level)
                result["base_predictions"][key] = parse_severity_from_response(pred, cond, level)
        
        result["base_response"] = pred
        result["gt_response"] = gt
        base_results.append(result)
    
    # Free base model memory
    del base_model, base_tokenizer, base_processor
    torch.cuda.empty_cache()
    
    # --- Run FINE-TUNED model ---
    print("\nLoading FINE-TUNED model...")
    ft_model, ft_tokenizer, ft_processor = load_model(
        model_path=cfg.output_dir + "/final", use_finetuned=True
    )
    
    print("Running fine-tuned model inference...")
    for i, sample in enumerate(tqdm(samples)):
        # BUG FIX v3.0: use image_paths (list)
        image_paths = sample.get("image_paths", sample.get("image_path", []))
        eval_prompt = build_user_prompt(
            n_images=len(image_paths) if not isinstance(image_paths, str) else 1,
            series_types=sample.get("series_types"),
        )
        pred = run_inference(ft_model, ft_tokenizer, ft_processor,
                            image_paths, eval_prompt)
        
        base_results[i]["ft_predictions"] = {}
        for cond in cfg.conditions:
            for level in cfg.levels:
                key = f"{cond}_{level}"
                base_results[i]["ft_predictions"][key] = \
                    parse_severity_from_response(pred, cond, level)
        
        base_results[i]["ft_response"] = pred
    
    del ft_model, ft_tokenizer, ft_processor
    torch.cuda.empty_cache()
    
    # --- Compute metrics for both ---
    # Reformat for compute_metrics compatibility. `pred_response` is passed
    # through so compute_metrics can compute parse_failure_rate honestly:
    # unparseable base-model output silently maps to "Normal/Mild" and would
    # otherwise inflate the base baseline by class prior.
    base_formatted = [{"ground_truth": r["ground_truth"],
                       "predictions": r["base_predictions"],
                       "pred_response": r["base_response"],
                       "study_id": r["study_id"]} for r in base_results]
    ft_formatted = [{"ground_truth": r["ground_truth"],
                    "predictions": r["ft_predictions"],
                    "pred_response": r["ft_response"],
                    "study_id": r["study_id"]} for r in base_results]
    
    base_metrics = compute_metrics(base_formatted)
    ft_metrics = compute_metrics(ft_formatted)
    
    # --- Generate comparison report ---
    print_comparison_report(base_metrics, ft_metrics)
    plot_comparison_chart(base_metrics, ft_metrics, output_path)
    save_text_examples(base_results, output_path)
    
    # Save full results
    with open(output_path / "comparison_results.json", "w") as f:
        json.dump({
            "base_metrics": {k: v for k, v in base_metrics.items() 
                           if k != "condition_metrics"},
            "ft_metrics": {k: v for k, v in ft_metrics.items() 
                          if k != "condition_metrics"},
            "samples": base_results
        }, f, indent=2)
    
    print(f"\nComparison results saved to: {output_dir}")
    return base_metrics, ft_metrics


def print_comparison_report(base_metrics: dict, ft_metrics: dict):
    """Print side-by-side metrics comparison."""
    print("\n" + "="*70)
    print("  BASE vs FINE-TUNED MODEL COMPARISON")
    print("="*70)
    
    metrics_to_compare = [
        ("Overall Accuracy",          "overall_accuracy",          "up"),
        ("Overall F1 (weighted)",     "overall_f1_weighted",       "up"),
        ("Overall Kappa (QW)",        "overall_kappa",             "up"),
        ("Weighted Accuracy",         "weighted_accuracy",         "up"),
        ("RSNA Weighted Score Proxy", "rsna_weighted_score_proxy", "down"),
        ("Parse Failure Rate",        "parse_failure_rate",        "down"),
    ]

    for name, key, direction in metrics_to_compare:
        base_val = base_metrics.get(key, 0)
        ft_val = ft_metrics.get(key, 0)
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        marker = " (↑ better)" if direction == "up" else " (↓ better)"
        print(f"  {name:<35} Base: {base_val:.4f}  FT: {ft_val:.4f}  "
              f"({sign}{delta:.4f}){marker}")
    
    print(f"\n  Per-Condition Kappa:")
    for cond in cfg.conditions:
        key = f"{cond}_avg_kappa"
        base_val = base_metrics.get(key, 0)
        ft_val = ft_metrics.get(key, 0)
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        cond_name = cond.replace("_", " ").title()
        print(f"    {cond_name:<40} Base: {base_val:.3f}  FT: {ft_val:.3f}  "
              f"({sign}{delta:.3f})")


def plot_comparison_chart(base_metrics: dict, ft_metrics: dict, output_path: Path):
    """Generate bar chart comparing base vs fine-tuned performance."""
    
    conditions = cfg.conditions
    base_kappas = [base_metrics.get(f"{c}_avg_kappa", 0) for c in conditions]
    ft_kappas = [ft_metrics.get(f"{c}_avg_kappa", 0) for c in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Per-condition kappa
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, base_kappas, width, label="Base MedGemma", 
                    color="#2196F3", alpha=0.8)
    bars2 = ax1.bar(x + width/2, ft_kappas, width, label="Fine-tuned", 
                    color="#4CAF50", alpha=0.8)
    
    ax1.set_xlabel("Condition")
    ax1.set_ylabel("Quadratic Weighted Kappa")
    ax1.set_title("Per-Condition Performance\n(Quadratic Weighted Kappa)")
    ax1.set_xticks(x)
    cond_labels = [c.replace("_", "\n") for c in conditions]
    ax1.set_xticklabels(cond_labels, fontsize=8)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.6, color="orange", linestyle="--", alpha=0.5, label="Good threshold")
    ax1.grid(axis="y", alpha=0.3)
    
    # Overall metrics
    ax2 = axes[1]
    overall_metrics = ["Accuracy", "F1 (weighted)", "Kappa (QW)"]
    base_overall = [
        base_metrics["overall_accuracy"],
        base_metrics["overall_f1_weighted"],
        base_metrics["overall_kappa"]
    ]
    ft_overall = [
        ft_metrics["overall_accuracy"],
        ft_metrics["overall_f1_weighted"],
        ft_metrics["overall_kappa"]
    ]
    
    x2 = np.arange(len(overall_metrics))
    ax2.bar(x2 - width/2, base_overall, width, label="Base MedGemma", 
            color="#2196F3", alpha=0.8)
    ax2.bar(x2 + width/2, ft_overall, width, label="Fine-tuned", 
            color="#4CAF50", alpha=0.8)
    
    ax2.set_xlabel("Metric")
    ax2.set_ylabel("Score")
    ax2.set_title("Overall Performance Comparison")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(overall_metrics)
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    
    plt.suptitle("MedGemma: Base vs Fine-tuned on RSNA 2024 Lumbar Spine", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    save_path = output_path / "comparison_chart.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved: {save_path}")


def save_text_examples(results: list, output_path: Path, num_examples: int = 5):
    """Save text examples showing base vs fine-tuned responses."""
    
    with open(output_path / "text_examples.txt", "w") as f:
        f.write("BASE vs FINE-TUNED: Response Examples\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(results[:num_examples]):
            f.write(f"Example {i+1} — Study ID: {result['study_id']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"GROUND TRUTH:\n{result['gt_response']}\n\n")
            f.write(f"BASE MODEL:\n{result['base_response']}\n\n")
            f.write(f"FINE-TUNED:\n{result['ft_response']}\n\n")
            f.write("=" * 80 + "\n\n")
    
    print(f"Text examples saved: {output_path / 'text_examples.txt'}")


if __name__ == "__main__":
    compare_models(num_samples=20)