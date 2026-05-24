#!/bin/bash
# save as /workspace/setup.sh and run: bash /workspace/setup.sh
#
# Target environment:
#   Python 3.10 or 3.11
#   CUDA 12.1+ with a bf16-capable GPU (Ampere or newer)
#   Tested on RunPod A100 80GB and A100 40GB images
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

# Install dependencies
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# IMPORTANT: every spec is quoted so bash does NOT interpret '>=' as a
# redirect operator (which would silently drop everything after the first
# package on each line and create stray files named '=4.45.0' etc.).
pip install \
    "transformers>=4.45.0,<4.50" \
    "trl>=0.11.0,<0.12" \
    "datasets>=2.20.0" \
    "pillow>=10.0.0" \
    "pydicom>=2.4.0" \
    "scikit-learn>=1.5.0" \
    "kaggle" \
    "accelerate" \
    "bitsandbytes" \
    "peft" \
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