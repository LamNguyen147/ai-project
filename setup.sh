#!/bin/bash
# save as /workspace/setup.sh and run: bash /workspace/setup.sh

set -e

echo "=== Setting up MedGemma Fine-tuning Environment ==="

# Create directory structure
mkdir -p /workspace/{medgemma-finetune,data,models,logs}
cd /workspace/medgemma-finetune

# Install dependencies
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install \
    transformers>=4.45.0 \
    trl>=0.11.0 \
    datasets>=2.20.0 \
    pillow>=10.0.0 \
    pydicom>=2.4.0 \
    scikit-learn>=1.5.0 \
    kaggle \
    accelerate \
    bitsandbytes \
    peft \
    scipy \
    matplotlib \
    seaborn \
    opencv-python-headless \
    huggingface_hub

# Login to HuggingFace
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Setup Kaggle
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << EOF
{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

echo "=== Setup complete! ==="