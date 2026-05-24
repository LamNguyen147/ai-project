# data_prep.py
"""
Converts RSNA 2024 DICOM images to PNG + builds instruction-tuning dataset.

Strategy:
  1. Identify series by description (Sagittal T1, Sagittal T2, Axial T2)
  2. Use label coordinates to select slices that best cover each vertebral level
  3. Export at 896x896 to match MedGemma 1.5's native SigLIP resolution
  4. Build multi-image instruction examples where possible
"""

import os
import json
import random
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
from config import cfg


# ── Series type detection ────────────────────────────────────────────────────

# Keywords used to classify a series description into one of three types.
# The RSNA dataset uses free-text descriptions, so we do substring matching.
SERIES_TYPE_KEYWORDS = {
    "sagittal_t2": ["sagittal t2", "sag t2", "t2 sag", "stir", "t2/stir"],
    "sagittal_t1": ["sagittal t1", "sag t1", "t1 sag"],
    "axial_t2":    ["axial t2", "ax t2", "t2 ax", "axial"],
}

# Priority order: sagittal T2 is most informative for canal/foraminal stenosis
SERIES_PRIORITY = ["sagittal_t2", "sagittal_t1", "axial_t2"]

class NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalar types to native Python before JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def classify_series(description: str) -> Optional[str]:
    """Map a free-text series description to a canonical series type."""
    desc_lower = description.lower().strip()
    for series_type, keywords in SERIES_TYPE_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return series_type
    return None  # Unknown series type — we'll still include it as fallback


def get_series_map(study_id: int, series_desc_df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Returns a dict mapping series_type -> [series_id, ...] for a study.
    series_desc_df must have columns: study_id, series_id, series_description
    """
    study_rows = series_desc_df[series_desc_df["study_id"] == study_id]
    result: Dict[str, List[int]] = {}
    for _, row in study_rows.iterrows():
        stype = classify_series(str(row["series_description"]))
        if stype is None:
            stype = "unknown"
        result.setdefault(stype, []).append(int(row["series_id"]))
    return result


# ── Slice selection ───────────────────────────────────────────────────────────

def select_representative_slices(
    dcm_files: List[Path],
    coord_rows: pd.DataFrame,
    n_slices: int = 3,
) -> List[Path]:
    """
    Select N slices that best cover the annotated vertebral levels.

    coord_rows: rows from train_label_coordinates.csv for this study+series.
    Falls back to evenly-spaced slices if no coordinates are available.
    """
    total = len(dcm_files)
    if total == 0:
        return []

    if coord_rows is not None and len(coord_rows) > 0:
        # Use the instance_number column (1-indexed) to identify annotated slices
        if "instance_number" in coord_rows.columns:
            instance_nums = coord_rows["instance_number"].dropna().astype(int).unique()
            # Map instance numbers to file indices (instance numbers are 1-indexed)
            # Sort DICOM files by InstanceNumber tag if available
            indexed = _sort_dcm_by_instance(dcm_files)
            selected_indices = []
            for inst in sorted(instance_nums):
                idx = inst - 1  # convert 1-indexed → 0-indexed
                if 0 <= idx < len(indexed):
                    selected_indices.append(idx)

            # Spread selected to cover range; if fewer than n_slices, pad with neighbors
            if len(selected_indices) >= n_slices:
                # Pick n_slices spread across the range
                step = len(selected_indices) // n_slices
                chosen = [selected_indices[i * step] for i in range(n_slices)]
            else:
                chosen = selected_indices
                # Pad with evenly spaced fallback
                while len(chosen) < n_slices:
                    mid = total // 2
                    if mid not in chosen:
                        chosen.append(mid)
                    else:
                        chosen.append(min(chosen) - 1 if min(chosen) > 0 else max(chosen) + 1)
                chosen = sorted(set(max(0, min(c, total - 1)) for c in chosen))[:n_slices]

            return [indexed[i] for i in chosen]

    # Fallback: evenly spaced slices (skipping first/last 10% which are often blank)
    margin = max(1, total // 10)
    usable = dcm_files[margin: total - margin]
    if len(usable) == 0:
        usable = dcm_files
    step = max(1, len(usable) // n_slices)
    return [usable[i * step] for i in range(min(n_slices, len(usable)))]


def _sort_dcm_by_instance(dcm_files: List[Path]) -> List[Path]:
    """Sort DICOM files by InstanceNumber tag; fall back to filename sort."""
    try:
        tagged = []
        for f in dcm_files:
            try:
                dcm = pydicom.dcmread(str(f), stop_before_pixels=True)
                inst = int(getattr(dcm, "InstanceNumber", 9999))
            except Exception:
                inst = 9999
            tagged.append((inst, f))
        tagged.sort(key=lambda x: x[0])
        return [f for _, f in tagged]
    except Exception:
        return sorted(dcm_files)


# ── DICOM → PNG conversion ────────────────────────────────────────────────────

def dicom_to_png(dcm_path: str, output_path: str,
                 size: int = cfg.image_size) -> bool:
    """
    Convert a DICOM file to a normalized PNG at `size x size` pixels.

    Resolution: 896x896 — matches MedGemma 1.5's SigLIP vision encoder.
    Downsampling to 512x512 or smaller WILL lose fine detail in small
    structures like neural foramina and subarticular recesses.

    Normalization: percentile window (p1–p99) then LANCZOS resize.
    """
    try:
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)

        # Apply DICOM modality LUT (Rescale Slope / Intercept)
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            img = img * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

        # Apply VOI LUT (Window Center / Width) if available, else use percentile
        if hasattr(dcm, "WindowCenter") and hasattr(dcm, "WindowWidth"):
            wc = float(dcm.WindowCenter[0] if isinstance(dcm.WindowCenter, pydicom.multival.MultiValue)
                       else dcm.WindowCenter)
            ww = float(dcm.WindowWidth[0] if isinstance(dcm.WindowWidth, pydicom.multival.MultiValue)
                       else dcm.WindowWidth)
            lo, hi = wc - ww / 2, wc + ww / 2
        else:
            lo, hi = np.percentile(img, [1, 99])

        img = np.clip(img, lo, hi)
        img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

        # Resize to 896x896 using high-quality LANCZOS downsampling
        pil_img = Image.fromarray(img).convert("RGB")
        pil_img = pil_img.resize((size, size), Image.LANCZOS)
        pil_img.save(output_path, optimize=False)
        return True
    except Exception as e:
        print(f"  [WARN] Error converting {dcm_path}: {e}")
        return False


# ── Instruction building ──────────────────────────────────────────────────────

def build_instruction(
    study_id: int,
    image_paths: List[str],
    series_types: List[str],
    labels: Dict[str, str],
) -> dict:
    """
    Build a single multi-image instruction-tuning example.

    image_paths: list of exported PNG paths for this study
    series_types: corresponding modality label per image (e.g. 'sagittal_t2')
    labels: dict mapping 'condition_level' -> severity string
    """
    # Format the label descriptions
    findings = []
    for level in cfg.levels:
        level_findings = []
        for cond in cfg.conditions:
            sev = labels.get(f"{cond}_{level}")
            if sev:
                cond_name = cond.replace("_", " ").title()
                level_findings.append(f"- {cond_name}: {sev}")
        if level_findings:
            level_name = level.upper().replace("_", "/")
            findings.append(f"\n**{level_name}:**\n" + "\n".join(level_findings))

    findings_text = "\n".join(findings)

    # Describe which views are provided (helps the model orient itself)
    view_desc = ", ".join(
        t.replace("_", " ").title() for t in series_types
    ) if series_types else "MRI"

    prompts = [
        f"You are provided with {len(image_paths)} lumbar spine MRI image(s) "
        f"({view_desc}). Analyze them and classify all degenerative conditions "
        f"at each spinal level from L1/L2 to L5/S1.",

        f"These lumbar spine MRI images ({view_desc}) show degenerative changes. "
        f"For each level (L1/L2 through L5/S1), assess: Spinal Canal Stenosis, "
        f"Left/Right Neural Foraminal Narrowing, and Left/Right Subarticular Stenosis "
        f"as Normal/Mild, Moderate, or Severe.",

        f"Review the provided {view_desc} MRI images of the lumbar spine and provide "
        f"a structured severity classification for all five degenerative conditions "
        f"at each of the five intervertebral levels.",
    ]
    user_prompt = random.choice(prompts)

    # Build message content with multiple images
    user_content = []
    for _ in image_paths:
        user_content.append({"type": "image"})  # one image token per image
    user_content.append({"type": "text", "text": user_prompt})

    assistant_response = (
        f"Based on my analysis of the provided {view_desc} MRI images, "
        f"here are the lumbar spine findings:\n"
        f"{findings_text}\n\n"
        f"Classification follows the standard three-tier severity grading: "
        f"Normal/Mild, Moderate, and Severe. All five lumbar intervertebral "
        f"levels (L1/L2 to L5/S1) were assessed."
    )

    return {
        "study_id": study_id,
        "image_paths": image_paths,        # list — multi-image support
        "series_types": series_types,
        "messages": [
            {"role": "user",    "content": user_content},
            {"role": "assistant", "content": assistant_response},
        ],
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def prepare_dataset(max_samples: Optional[int] = None):
    """
    Main function to prepare the training dataset.

    Steps:
      1. Load label CSV and series description CSV
      2. Load coordinate CSV (for slice selection)
      3. For each study: identify series by type, extract representative slices
      4. Export slices as 896x896 PNGs
      5. Build instruction-tuning JSON
    """
    print("Loading CSVs...")
    df_labels = pd.read_csv(cfg.train_csv)
    data_dir = Path(cfg.data_dir)

    # Series descriptions: study_id, series_id, series_description
    desc_path = data_dir / "train_series_descriptions.csv"
    if desc_path.exists():
        df_desc = pd.read_csv(desc_path)
        print(f"  Loaded series descriptions: {len(df_desc)} rows")
    else:
        df_desc = None
        print("  [WARN] train_series_descriptions.csv not found — will use directory order fallback")

    # Label coordinates: study_id, series_id, instance_number, condition, level, x, y
    coord_path = data_dir / "train_label_coordinates.csv"
    if coord_path.exists():
        df_coords = pd.read_csv(coord_path)
        print(f"  Loaded label coordinates: {len(df_coords)} rows")
    else:
        df_coords = None
        print("  [WARN] train_label_coordinates.csv not found — will use evenly-spaced slice fallback")

    # Create output directory for PNGs
    png_dir = Path(cfg.processed_data_dir) / "images_896"
    png_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    study_ids = df_labels["study_id"].unique()
    if max_samples:
        study_ids = study_ids[:max_samples]

    print(f"\nProcessing {len(study_ids)} studies...")

    for study_id in tqdm(study_ids, desc="Studies"):
        study_path = data_dir / "train_images" / str(study_id)
        if not study_path.exists():
            continue

        # ── Build labels dict for this study ─────────────────────────────
        study_row = df_labels[df_labels["study_id"] == study_id]
        if len(study_row) == 0:
            continue
        study_row = study_row.iloc[0]

        labels = {}
        for cond in cfg.conditions:
            for level in cfg.levels:
                col = f"{cond}_{level}"
                if col in study_row.index and not pd.isna(study_row[col]):
                    val = study_row[col]
                    # BUG FIX v3.0: RSNA 2024 train.csv stores severity as
                    # strings ("Normal/Mild", "Moderate", "Severe"), NOT
                    # integers. Calling int() on these would crash. We accept
                    # both formats for robustness.
                    if isinstance(val, str):
                        # Normalise capitalisation differences
                        val_norm = val.strip()
                        if val_norm in cfg.severity_labels:
                            labels[col] = val_norm
                        elif val_norm.lower() in ("normal/mild", "normal", "mild"):
                            labels[col] = "Normal/Mild"
                        elif val_norm.lower() == "moderate":
                            labels[col] = "Moderate"
                        elif val_norm.lower() == "severe":
                            labels[col] = "Severe"
                        # else: unknown string — skip
                    else:
                        # Integer encoding (0/1/2) — kept for compatibility
                        idx = int(val)
                        if 0 <= idx < len(cfg.severity_labels):
                            labels[col] = cfg.severity_labels[idx]

        if not labels:
            continue

        # ── Identify series directories ───────────────────────────────────
        if df_desc is not None:
            series_map = get_series_map(study_id, df_desc)
        else:
            # Fallback: treat all directories as unknown type in filesystem order
            series_dirs = sorted(study_path.iterdir())
            series_map = {"unknown": [int(d.name) for d in series_dirs if d.is_dir()]}

        # ── Select series to include (up to max_series_per_study) ────────
        selected_series: List[Tuple[str, int]] = []  # (series_type, series_id)
        for stype in SERIES_PRIORITY:
            if stype in series_map:
                for sid in series_map[stype]:
                    selected_series.append((stype, sid))
        # Add any unknown series not yet included
        for sid in series_map.get("unknown", []):
            selected_series.append(("unknown", sid))

        selected_series = selected_series[: cfg.max_series_per_study]

        if not selected_series:
            continue

        # ── Process each selected series ──────────────────────────────────
        all_image_paths = []
        all_series_types = []

        for series_type, series_id in selected_series:
            series_path = study_path / str(series_id)
            if not series_path.exists():
                continue

            dcm_files = sorted(series_path.glob("*.dcm"))
            if not dcm_files:
                continue

            # Get coordinate rows for this series (for slice selection)
            if df_coords is not None:
                coord_rows = df_coords[
                    (df_coords["study_id"] == study_id) &
                    (df_coords["series_id"] == series_id)
                ]
            else:
                coord_rows = None

            # Select representative slices
            chosen_slices = select_representative_slices(
                dcm_files, coord_rows, n_slices=cfg.slices_per_series
            )

            for i, dcm_file in enumerate(chosen_slices):
                out_name = f"{study_id}_{series_id}_{series_type}_slice{i}.png"
                out_path = png_dir / out_name
                if not out_path.exists():
                    ok = dicom_to_png(str(dcm_file), str(out_path), size=cfg.image_size)
                    if not ok:
                        continue
                all_image_paths.append(str(out_path))
                all_series_types.append(series_type)

        if not all_image_paths:
            continue

        # ── Build training example ────────────────────────────────────────
        example = build_instruction(study_id, all_image_paths, all_series_types, labels)
        dataset.append(example)

    # ── Train / val split ─────────────────────────────────────────────────
    random.seed(cfg.seed)
    random.shuffle(dataset)
    split = int(0.9 * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]

    out_dir = Path(cfg.processed_data_dir)
    with open(out_dir / "train_dataset.json", "w") as f:
        json.dump(train_data, f, indent=2, cls=NumpyEncoder)
    with open(out_dir / "val_dataset.json", "w") as f:
        json.dump(val_data, f, indent=2, cls=NumpyEncoder)

    print(f"\nDataset created:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")
    avg_imgs = np.mean([len(d["image_paths"]) for d in dataset])
    print(f"  Avg images per example: {avg_imgs:.1f}")
    print(f"  Image resolution: {cfg.image_size}×{cfg.image_size}")
    print(f"  Saved to: {out_dir}")

    return train_data, val_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of studies (for quick sanity checks)")
    args = parser.parse_args()
    prepare_dataset(max_samples=args.max_samples)