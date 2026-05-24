# config.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    # Model — use Unsloth's pre-optimized MedGemma 1.5 checkpoint
    # Falls back to Google's official repo if Unsloth version is unavailable
    model_name: str = "unsloth/medgemma-1.5-4b-it"
    # Alternative: "google/medgemma-1.5-4b-it"
    output_dir: str = "/workspace/models/medgemma-lumbar"
    
    # Data
    data_dir: str = "/workspace/data/rsna-2024-lumbar-spine"
    train_csv: str = "/workspace/data/rsna-2024-lumbar-spine/train.csv"
    series_desc_csv: str = "/workspace/data/rsna-2024-lumbar-spine/train_series_descriptions.csv"
    processed_data_dir: str = "/workspace/data/processed"
    
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
    # NOTE: max_seq_length increased to 4096 to accommodate 896x896 visual tokens
    # which produce ~3136 image tokens (vs ~256 at 224x224).
    # On RTX 4090 (24GB): reduce to 2048 and set finetune_vision_layers=False
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
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

cfg = Config()