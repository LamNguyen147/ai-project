# config.py
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Workspace root. Override with `WORKSPACE=/some/path` so the repo is no longer
# tied to RunPod's /workspace convention.
WORKSPACE = os.environ.get("WORKSPACE", "/workspace")


# ── Label schema (shared across data_prep, train, evaluate, compare) ─────────
# A compact JSON schema replaces the previous prose answer template. Compact
# keys cut the assistant token count from ~500 → ~100, raising the proportion
# of label-bearing tokens in the SFT loss.
#
# Example assistant response:
#   {"L1L2":{"canal":"N","lf":"N","rf":"N","ls":"N","rs":"N"},
#    "L2L3":{"canal":"M","lf":"N","rf":"N","ls":"N","rs":"S"}, ...}

COND_KEYS: Dict[str, str] = {
    "spinal_canal_stenosis":             "canal",
    "left_neural_foraminal_narrowing":   "lf",
    "right_neural_foraminal_narrowing":  "rf",
    "left_subarticular_stenosis":        "ls",
    "right_subarticular_stenosis":       "rs",
}
COND_FROM_KEY: Dict[str, str] = {v: k for k, v in COND_KEYS.items()}

LEVEL_KEYS: Dict[str, str] = {
    "l1_l2": "L1L2", "l2_l3": "L2L3", "l3_l4": "L3L4",
    "l4_l5": "L4L5", "l5_s1": "L5S1",
}
LEVEL_FROM_KEY: Dict[str, str] = {v: k for k, v in LEVEL_KEYS.items()}

SEVERITY_CODES: Dict[str, str] = {"Normal/Mild": "N", "Moderate": "M", "Severe": "S"}
SEVERITY_FROM_CODE: Dict[str, str] = {v: k for k, v in SEVERITY_CODES.items()}


# Hardware-specific overlays. Pick one with the PROFILE env var, e.g.:
#   PROFILE=rtx_4090 python train.py
# Anything left out of a profile inherits the dataclass default below.
PROFILES = {
    "a100_80g": {
        # 3 sagittal slices × 4096 image tokens + ~1024 for prompt/answer
        "max_seq_length": 13312,
        "image_size": 896,
        "finetune_vision_layers": True,
        "slices_per_series": 3,
        "max_series_per_study": 1,
        "per_device_train_batch_size": 1,
        "bf16": True,
    },
    "a100_40g": {
        # Single slice profile that fits on a 40 GB A100
        "max_seq_length": 5120,
        "image_size": 896,
        "finetune_vision_layers": True,
        "slices_per_series": 1,
        "max_series_per_study": 1,
        "per_device_train_batch_size": 1,
        "bf16": True,
    },
    "rtx_4090": {
        # 24 GB consumer card: drop image res so vision tokens fit, freeze
        # the vision encoder, and shrink the LM context accordingly.
        "max_seq_length": 2048,
        "image_size": 448,
        "finetune_vision_layers": False,
        "slices_per_series": 1,
        "max_series_per_study": 1,
        "per_device_train_batch_size": 1,
        "bf16": True,
    },
}


def set_seed(seed: int) -> None:
    """
    Seed every RNG we touch. Imports of numpy/torch are deferred so this module
    stays importable in environments that don't have them (e.g. static lint).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class Config:
    # Model — use Unsloth's pre-optimized MedGemma 1.5 checkpoint
    # Falls back to Google's official repo if Unsloth version is unavailable
    model_name: str = "unsloth/medgemma-1.5-4b-it"
    # Alternative: "google/medgemma-1.5-4b-it"
    output_dir: str = f"{WORKSPACE}/models/medgemma-lumbar"

    # Data
    data_dir: str = f"{WORKSPACE}/data/rsna-2024-lumbar-spine"
    train_csv: str = f"{WORKSPACE}/data/rsna-2024-lumbar-spine/train.csv"
    series_desc_csv: str = f"{WORKSPACE}/data/rsna-2024-lumbar-spine/train_series_descriptions.csv"
    processed_data_dir: str = f"{WORKSPACE}/data/processed"
    
    # Image resolution — MedGemma 1.5 natively processes 896x896.
    # Do NOT downsample below this; upsampling from smaller sizes loses detail.
    image_size: int = 896
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training
    # NOTE: at 896×896 a single image consumes ~4096 SigLIP patch tokens.
    # max_seq_length must be > image_tokens + prompt_tokens + answer_tokens,
    # otherwise the assistant label is truncated and the model trains on
    # nothing. 5120 leaves ~1024 tokens for prompt + structured answer.
    # See PROFILES above for hardware-specific overrides (e.g. RTX 4090
    # uses 2048 with a 448² image).
    max_seq_length: int = 5120
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # Which LoRA target stacks to attach. Vision fine-tuning roughly doubles
    # peak VRAM; turn it off on 24 GB cards via the rtx_4090 profile.
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    fp16: bool = False
    bf16: bool = True
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Multi-series data strategy
    # BUG FIX v3.0: With 896×896 images and SigLIP patch_size=14, each image
    # produces ~4096 image tokens. max_seq_length=4096 means even a SINGLE
    # image fills the entire context, leaving almost no room for text.
    # Practical limits:
    #   - 1 image  → ~4096 image tokens + ~200 text tokens → needs seq ≥ 4300
    #   - 3 images → ~12288 image tokens → needs seq ≥ 12500 (A100 80GB only)
    # Default config: 1 slice per study from the best sagittal_t2 series.
    # Increase slices_per_series/max_series_per_study only if you have A100 80GB
    # and set max_seq_length accordingly (8192 for 2 images, 16384 for 4 images).
    slices_per_series: int = 1      # 1 slice = ~4096 image tokens; safe for all GPUs
    max_series_per_study: int = 1   # Use only the highest-priority series by default
    
    # Misc
    seed: int = 42
    hf_token: Optional[str] = None
    wandb_project: str = "medgemma-lumbar-spine"
    
    # Condition labels
    conditions: List[str] = field(default_factory=lambda: [
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis"
    ])
    levels: List[str] = field(default_factory=lambda: [
        "l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"
    ])
    severity_labels: List[str] = field(default_factory=lambda: [
        "Normal/Mild", "Moderate", "Severe"
    ])
    
    # RSNA competition weighted log-loss class weights
    # Severe misclassification is penalized more heavily
    rsna_loss_weights: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])

    def __post_init__(self):
        # Apply a hardware profile if requested. Unknown PROFILE values are
        # ignored loudly so a typo doesn't silently fall back to A100 defaults.
        profile_name = os.environ.get("PROFILE")
        if profile_name:
            if profile_name not in PROFILES:
                raise ValueError(
                    f"PROFILE={profile_name!r} is not one of {list(PROFILES)}"
                )
            for key, value in PROFILES[profile_name].items():
                if not hasattr(self, key):
                    raise AttributeError(
                        f"Profile {profile_name!r} sets unknown Config field {key!r}"
                    )
                setattr(self, key, value)

cfg = Config()