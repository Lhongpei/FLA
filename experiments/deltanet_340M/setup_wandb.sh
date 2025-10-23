#!/bin/bash

# WandB 配置脚本
# 运行这个脚本来配置你的 WandB 账户

echo "============================================"
echo "WandB Configuration Setup"
echo "============================================"
echo ""

# Activate environment
source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate fla

echo "Step 1: Get your WandB API key"
echo "  1. Visit: https://wandb.ai/authorize"
echo "  2. Login or create an account"
echo "  3. Copy your API key"
echo ""

# Check if already logged in
if wandb status &>/dev/null; then
    echo "✓ You are already logged in to WandB!"
    wandb status
    echo ""
    read -p "Do you want to re-login? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing login."
        exit 0
    fi
fi

echo "Step 2: Login to WandB"
echo ""
echo "Please paste your API key when prompted:"
wandb login

echo ""
echo "============================================"
echo "✓ WandB Configuration Complete!"
echo "============================================"
echo ""
echo "Verify your login:"
wandb status
echo ""
echo "You can now customize your WandB settings by editing:"
echo "  - train_baseline.sh"
echo "  - train_osla.sh"
echo ""
echo "Key parameters to customize:"
echo "  --wandb-project <your-project-name>"
echo "  --wandb-run-name <your-run-name>"
echo ""
echo "Or set environment variables:"
echo "  export WANDB_PROJECT='my-project'"
echo "  export WANDB_ENTITY='my-username'  # Optional: your WandB username/team"
echo "============================================"




