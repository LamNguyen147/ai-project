"""Find the best validation studies for a live demo.

The RSNA dataset is ~85% Normal/Mild, so a random study shows nothing. This
script scans the val set for studies that *contain* Moderate/Severe labels, runs
the fine-tuned model on each, and ranks them by how many of those abnormal cells
the model predicts correctly. The top studies are the ones to use in the demo:
real pathology that the model demonstrably detects.

Run on the pod (needs the adapter + val_dataset.jsonl):
    python find_demo_samples.py                      # rank Severe-containing studies
    python find_demo_samples.py --severity moderate  # include Moderate too
    python find_demo_samples.py --limit 30           # only test the first 30 candidates
    python find_demo_samples.py --top 8              # print 8 best candidates

For each top candidate it prints ready-to-paste `/load` lines and the exact
ground-truth abnormal labels so you know what the model should call out.
"""
import argparse
import json
from pathlib import Path

from unsloth import FastVisionModel

from config import cfg
from demo_chat import run_diagnosis  # reuse the exact demo inference path

ADAPTER_PATH = "/workspace/models/medgemma-lumbar/final"
VAL_JSONL = Path(cfg.processed_data_dir) / "val_dataset.jsonl"


def abnormal_cells(findings: dict) -> dict:
    """Return {(level, cond): severity_code} for every Moderate/Severe cell."""
    return {
        (level, cond): sev
        for level, conds in findings.items()
        for cond, sev in conds.items()
        if sev in ("M", "S")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", choices=["moderate", "severe"], default="severe",
                    help="minimum ground-truth severity a study must contain")
    ap.add_argument("--adapter", default=ADAPTER_PATH)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap how many candidate studies to run the model on")
    ap.add_argument("--top", type=int, default=5, help="how many winners to print")
    args = ap.parse_args()

    target = 2 if args.severity == "severe" else 1
    label = "Severe" if target == 2 else "Moderate/Severe"

    # 1. Collect candidate studies from the val set using the max_severity field.
    candidates = []
    with open(VAL_JSONL) as f:
        for line in f:
            row = json.loads(line)
            if row.get("max_severity", 0) >= target:
                candidates.append(row)
    print(f"{len(candidates)} val studies contain at least one {label} label.")
    if args.limit:
        candidates = candidates[: args.limit]
        print(f"Testing the first {len(candidates)} (--limit).")

    if not candidates:
        print("Nothing to test. Try --severity moderate.")
        return

    # 2. Load the diagnostic model once (4-bit, ~6 GB).
    print(f"Loading model from {args.adapter}...")
    model, tok = FastVisionModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)

    # 3. Run the model on each candidate and score the abnormal-cell hit rate.
    results = []
    for i, row in enumerate(candidates, 1):
        gt = json.loads(row["messages"][-1]["content"])
        gt_abn = abnormal_cells(gt)
        try:
            pred = run_diagnosis(model, tok, row["image_paths"])
        except Exception as e:
            print(f"  [{i}/{len(candidates)}] study {row['study_id']}: FAILED ({e})")
            continue
        # exact severity match on the GT-abnormal cells
        hits = sum(1 for cell, sev in gt_abn.items()
                   if pred.get(cell[0], {}).get(cell[1]) == sev)
        # partial credit: model called it abnormal but got the grade wrong
        flagged = sum(1 for cell in gt_abn
                      if pred.get(cell[0], {}).get(cell[1]) in ("M", "S"))
        results.append({
            "study_id": row["study_id"],
            "n_abnormal": len(gt_abn),
            "exact_hits": hits,
            "flagged": flagged,
            "gt_abnormal": gt_abn,
            "image_paths": row["image_paths"],
        })
        print(f"  [{i}/{len(candidates)}] study {row['study_id']}: "
              f"exact {hits}/{len(gt_abn)}, flagged-abnormal {flagged}/{len(gt_abn)}")

    if not results:
        print("No studies produced parseable predictions.")
        return

    # 4. Rank: most exact hits first, then most flagged-abnormal, then most disease.
    results.sort(key=lambda r: (r["exact_hits"], r["flagged"], r["n_abnormal"]),
                 reverse=True)

    print("\n" + "=" * 64)
    print(f" Top {min(args.top, len(results))} demo candidates")
    print(" (the model correctly flags real pathology in these)")
    print("=" * 64)
    for r in results[: args.top]:
        print(f"\nStudy {r['study_id']} — model got {r['exact_hits']}/{r['n_abnormal']} "
              f"abnormal cells exactly right ({r['flagged']}/{r['n_abnormal']} flagged abnormal)")
        gt_str = ", ".join(f"{lvl} {cond}={sev}"
                           for (lvl, cond), sev in r["gt_abnormal"].items())
        print(f"  Ground-truth abnormal: {gt_str}")
        print("  Paste into the demo:")
        for p in r["image_paths"]:
            print(f"    /load {p}")


if __name__ == "__main__":
    main()
