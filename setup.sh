#!/bin/bash
# save as /workspace/setup.sh and run: bash /workspace/setup.sh
#
# Target environment:
#   Python 3.10 or 3.11
#   CUDA 12.6+ with a bf16-capable GPU (Ampere or newer)
#   PyTorch 2.7+ (upgraded by this script if the base image is older)
#   Tested on RunPod A100 80GB PCIe/SXM with cu126 and cu128 images
#
# Required environment variables (the script aborts if any are unset):
#   HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY

set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN before running setup.sh (HuggingFace access token).}"
: "${KAGGLE_USERNAME:?Set KAGGLE_USERNAME before running setup.sh.}"
: "${KAGGLE_KEY:?Set KAGGLE_KEY before running setup.sh.}"

WORKSPACE="${WORKSPACE:-/workspace}"

echo "=== Setting up MedGemma Fine-tuning Environment ==="
echo "Workspace: $WORKSPACE"

# Create directory structure
mkdir -p "$WORKSPACE"/{medgemma-finetune,data,models,logs}
cd "$WORKSPACE/medgemma-finetune"

# Install system utilities missing from some RunPod images
apt-get update -qq && apt-get install -y -qq unzip tmux

# Install dependencies
pip install --upgrade pip

# unsloth-zoo 2026.x requires torchao>=0.13.0 which requires torch>=2.7.
# If the base image already ships torch>=2.7, skip the install to avoid
# downgrading. Otherwise try cu128 → cu126 → cu124 in order.
python -c "
import torch, sys
major, minor = (int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
sys.exit(0 if (major, minor) >= (2, 7) else 1)
" 2>/dev/null || {
    pip install "torch>=2.7.0" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 || \
    pip install "torch>=2.7.0" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 || \
    pip install "torch>=2.6.0" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
}

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# IMPORTANT: every spec is quoted so bash does NOT interpret '>=' as a
# redirect operator (which would silently drop everything after the first
# package on each line and create stray files named '=4.45.0' etc.).
pip install \
    "transformers>=4.56.1,!=4.57.4,!=4.57.5,!=5.0.0,!=5.1.0,<=5.5.0" \
    "trl>=0.18.2,!=0.19.0,<=0.24.0" \
    "datasets>=2.20.0" \
    "pillow>=10.0.0" \
    "pydicom>=2.4.0" \
    "scikit-learn>=1.5.0" \
    "kaggle" \
    "accelerate>=1.0" \
    "bitsandbytes>=0.43.1" \
    "peft>=0.18.0" \
    "scipy" \
    "matplotlib" \
    "seaborn" \
    "opencv-python-headless" \
    "huggingface_hub"

# Login to HuggingFace
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Setup Kaggle
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << EOF
{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

echo "=== Setup complete! ==="