# data_prep.py
"""
Converts RSNA 2024 DICOM images to PNG + builds instruction-tuning dataset.

Strategy:
  1. Identify series by description (Sagittal T1, Sagittal T2, Axial T2)
  2. Use label coordinates to select slices that best cover each vertebral level
  3. Export at cfg.image_size (square) to match MedGemma 1.5's SigLIP resolution
  4. Emit a compact JSON answer template so the SFT loss is dominated by
     label-bearing tokens, not prose boilerplate.
  5. Write JSONL (one example per line) for streaming-friendly loading.
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
from config import (
    cfg,
    set_seed,
    COND_KEYS,
    LEVEL_KEYS,
    SEVERITY_CODES,
)

# SigLIP patch size for MedGemma 1.5. Used to estimate vision-token cost
# per image so we can drop examples that exceed cfg.max_seq_length before
# they reach training.
SIGLIP_PATCH_SIZE = 14
# Safety margin (tokens) reserved for the chat-template wrappers (BOS/EOS,
# role headers, etc.). Image and answer tokens must fit in
#   cfg.max_seq_length - TOKEN_SAFETY_MARGIN.
TOKEN_SAFETY_MARGIN = 256


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
    allow_no_coords: bool = False,
) -> List[Path]:
    """
    Select N slices that best cover the annotated vertebral levels.

    coord_rows: rows from train_label_coordinates.csv for this study+series.
    If `allow_no_coords` is False and no coordinates exist, returns []
    so the caller can skip the study — we refuse to fabricate level→slice
    mappings, which would teach the model to assert anatomy it cannot see.
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

            # Spread the annotated slices evenly across the available range.
            if len(selected_indices) >= n_slices:
                if n_slices == 1:
                    chosen = [selected_indices[len(selected_indices) // 2]]
                else:
                    positions = np.linspace(
                        0, len(selected_indices) - 1, n_slices
                    ).round().astype(int)
                    chosen = [selected_indices[p] for p in positions]
            else:
                # Fewer annotations than requested slices: keep them all,
                # then pad with the middle of the un-annotated range.
                chosen = list(selected_indices)
                mid = total // 2
                k = 0
                while len(chosen) < n_slices:
                    cand = mid + (k if k % 2 == 0 else -k)
                    if 0 <= cand < total and cand not in chosen:
                        chosen.append(cand)
                    k += 1
                    if k > 2 * total:
                        break
                chosen = sorted(set(chosen))[:n_slices]

            return [indexed[i] for i in chosen]

    if not allow_no_coords:
        # Refuse to invent a level-to-slice mapping.
        return []

    # Opt-in fallback for smoke tests only: evenly spaced slices (skipping
    # first/last 10% which are often blank).
    margin = max(1, total // 10)
    usable = dcm_files[margin: total - margin]
    if len(usable) == 0:
        usable = dcm_files
    if n_slices == 1:
        return [usable[len(usable) // 2]]
    positions = np.linspace(0, len(usable) - 1, n_slices).round().astype(int)
    return [usable[p] for p in positions]


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

# ── Token-budget estimator ────────────────────────────────────────────────────
#
# We cannot import the real processor here (data_prep runs without GPU deps),
# but we can estimate tightly: SigLIP contributes (image_size/patch)² tokens
# per image, and BPE text averages ~4 chars/token. Examples that exceed
# cfg.max_seq_length - TOKEN_SAFETY_MARGIN are dropped before training.

def estimate_token_cost(n_images: int, text: str, image_size: int) -> int:
    """Conservative upper bound on the per-example token count."""
    image_tokens = n_images * (image_size // SIGLIP_PATCH_SIZE) ** 2
    text_tokens = len(text) // 3 + 16   # 3 chars/token (conservative)
    return image_tokens + text_tokens


# ── Instruction building (JSON answer template) ───────────────────────────────

# Single description of the answer schema, embedded in every user prompt so the
# model learns the contract from the data. Compact codes keep the assistant
# output near ~100 tokens (vs ~500 for the old prose template), raising the
# proportion of label-bearing tokens in the SFT loss.
SCHEMA_DESCRIPTION = (
    "Respond with one JSON object and nothing else. "
    "Keys are spinal levels: L1L2, L2L3, L3L4, L4L5, L5S1. "
    "Each value is an object with five condition keys: "
    "canal=Spinal Canal Stenosis, lf=Left Neural Foraminal Narrowing, "
    "rf=Right Neural Foraminal Narrowing, ls=Left Subarticular Stenosis, "
    "rs=Right Subarticular Stenosis. "
    "Each condition value is N (Normal/Mild), M (Moderate), or S (Severe). "
    'Example: {"L1L2":{"canal":"N","lf":"N","rf":"N","ls":"N","rs":"N"},...}'
)


def labels_to_json(labels: Dict[str, str]) -> str:
    """
    Encode the 25-way severity matrix as a compact JSON string matching the
    schema embedded in SCHEMA_DESCRIPTION.
    """
    matrix: Dict[str, Dict[str, str]] = {}
    for level in cfg.levels:
        level_key = LEVEL_KEYS[level]
        level_data: Dict[str, str] = {}
        for cond in cfg.conditions:
            cond_key = COND_KEYS[cond]
            sev = labels.get(f"{cond}_{level}")
            if sev:
                level_data[cond_key] = SEVERITY_CODES[sev]
        if level_data:
            matrix[level_key] = level_data
    return json.dumps(matrix, separators=(",", ":"))


def max_severity_rank(labels: Dict[str, str]) -> int:
    """0 if every label is Normal/Mild, 1 if any Moderate, 2 if any Severe."""
    rank = 0
    for sev in labels.values():
        if sev == "Severe":
            return 2
        if sev == "Moderate":
            rank = max(rank, 1)
    return rank


def build_instruction(
    study_id: int,
    image_paths: List[str],
    series_types: List[str],
    labels: Dict[str, str],
) -> dict:
    """
    Build a single multi-image instruction-tuning example with a JSON answer.

    The user prompt embeds the full schema description; the assistant answer
    is a single compact JSON object so downstream parsing is `json.loads`
    rather than regex against multi-line prose.
    """
    view_desc = (
        ", ".join(t.replace("_", " ").title() for t in series_types)
        if series_types else "MRI"
    )

    user_prompt = (
        f"You are provided with {len(image_paths)} lumbar spine MRI image(s) "
        f"({view_desc}). Classify all degenerative conditions at each spinal "
        f"level from L1/L2 to L5/S1.\n\n{SCHEMA_DESCRIPTION}"
    )

    user_content: List[dict] = [{"type": "image"} for _ in image_paths]
    user_content.append({"type": "text", "text": user_prompt})

    assistant_response = labels_to_json(labels)

    return {
        "study_id": int(study_id),
        "image_paths": image_paths,
        "series_types": series_types,
        "max_severity": max_severity_rank(labels),
        "messages": [
            {"role": "user",    "content": user_content},
            {"role": "assistant", "content": assistant_response},
        ],
    }


def oversample_by_severity(
    dataset: List[dict],
    weights: Optional[List[float]] = None,
) -> List[dict]:
    """
    Duplicate examples whose worst label is Moderate (×2) or Severe (×4),
    using `cfg.rsna_loss_weights` as duplication factors. This is the cheapest
    way to counter the natural ~85% Normal/Mild imbalance for an SFT setup
    that has no per-token loss weighting hook.
    """
    if weights is None:
        weights = cfg.rsna_loss_weights
    out: List[dict] = []
    for ex in dataset:
        k = int(round(weights[ex.get("max_severity", 0)]))
        out.extend([ex] * max(1, k))
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def prepare_dataset(
    max_samples: Optional[int] = None,
    allow_no_coords: bool = False,
    oversample: bool = False,
):
    """
    Main function to prepare the training dataset.

    Steps:
      1. Load label CSV and series description CSV
      2. Load coordinate CSV (for slice selection)
      3. For each study: identify series by type, extract representative slices
      4. Export slices as cfg.image_size² PNGs
      5. Build instruction-tuning JSONL
      6. Drop examples that exceed the token budget
      7. Optionally oversample Moderate/Severe studies

    `allow_no_coords` keeps the legacy fallback (evenly spaced slices with no
    real level→slice mapping) — only use it for smoke tests.
    """
    set_seed(cfg.seed)

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
        if not allow_no_coords:
            raise FileNotFoundError(
                f"{coord_path} not found and allow_no_coords=False. "
                "Pass --allow-no-coords to fall back to evenly-spaced slices "
                "(NOT recommended outside smoke tests — fabricates level→slice "
                "mappings and teaches the model to hallucinate anatomy)."
            )
        print("  [WARN] train_label_coordinates.csv not found — fabricated slice mapping enabled by --allow-no-coords")

    # Create output directory for PNGs (named by image_size for clarity).
    png_dir = Path(cfg.processed_data_dir) / f"images_{cfg.image_size}"
    png_dir.mkdir(parents=True, exist_ok=True)

    dataset: List[dict] = []
    dropped_no_coords = 0
    dropped_oversize = 0
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
                dcm_files, coord_rows,
                n_slices=cfg.slices_per_series,
                allow_no_coords=allow_no_coords,
            )

            if not chosen_slices:
                continue  # study had no coordinates; skip (see flag above)

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
            dropped_no_coords += 1
            continue

        # ── Build training example ────────────────────────────────────────
        example = build_instruction(study_id, all_image_paths, all_series_types, labels)

        # Token-budget guard: drop examples that wouldn't fit in
        # cfg.max_seq_length once vision + chat-template overhead is added.
        prompt_text = ""
        for item in example["messages"][0]["content"]:
            if item.get("type") == "text":
                prompt_text = item["text"]
                break
        answer_text = example["messages"][1]["content"]
        est = estimate_token_cost(
            n_images=len(all_image_paths),
            text=prompt_text + answer_text,
            image_size=cfg.image_size,
        )
        if est + TOKEN_SAFETY_MARGIN > cfg.max_seq_length:
            dropped_oversize += 1
            continue

        dataset.append(example)

    # ── Train / val split ─────────────────────────────────────────────────
    random.shuffle(dataset)
    split = int(0.9 * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]

    if oversample:
        n_before = len(train_data)
        train_data = oversample_by_severity(train_data)
        print(f"  Oversampled train set: {n_before} → {len(train_data)} examples")

    out_dir = Path(cfg.processed_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_dataset.jsonl"
    val_path   = out_dir / "val_dataset.jsonl"
    _write_jsonl(train_path, train_data)
    _write_jsonl(val_path, val_data)

    print(f"\nDataset created:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")
    if dataset:
        avg_imgs = float(np.mean([len(d["image_paths"]) for d in dataset]))
        print(f"  Avg images per example: {avg_imgs:.1f}")
    print(f"  Image resolution: {cfg.image_size}×{cfg.image_size}")
    print(f"  Dropped (no coords): {dropped_no_coords}")
    print(f"  Dropped (over token budget @ max_seq_length={cfg.max_seq_length}): {dropped_oversize}")
    print(f"  Saved to: {out_dir}")

    return train_data, val_data


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    """Write one JSON object per line. Tighter than indented JSON and lets
    train/eval stream the dataset instead of loading hundreds of MB at once."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), cls=NumpyEncoder) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of studies (for quick sanity checks)")
    parser.add_argument("--allow-no-coords", action="store_true",
                        help="Fall back to evenly-spaced slices when "
                             "train_label_coordinates.csv is missing. Smoke tests only — "
                             "the level→slice mapping is fabricated.")
    parser.add_argument("--oversample", action="store_true",
                        help="Duplicate Moderate (×2) and Severe (×4) studies in the "
                             "training split to counter the natural class imbalance.")
    args = parser.parse_args()
    prepare_dataset(
        max_samples=args.max_samples,
        allow_no_coords=args.allow_no_coords,
        oversample=args.oversample,
    )