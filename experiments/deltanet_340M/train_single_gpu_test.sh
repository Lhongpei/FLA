#!/bin/bash

# Single GPU test script
set -e

# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

# Training parameters
BATCH_SIZE=2
GRAD_ACCUM=4
SEQ_LEN=2048
MAX_STEPS=100
WARMUP_STEPS=10
LR=3e-4
MIN_LR=3e-5

# Dataset
DATASET_NAME="fla-hub/pg19"

# Output
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/deltanet_340M.json"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/test_single_gpu"
WANDB_PROJECT="fla-deltanet-test"
RUN_NAME="single-gpu-test"

echo "============================================"
echo "Single GPU Test - DeltaNet 340M"
echo "============================================"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Activate conda environment
source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate fla

# Run on single GPU (no distributed training)
python ${SCRIPT_DIR}/train_deltanet.py \
    --config ${CONFIG_PATH} \
    --dataset-name ${DATASET_NAME} \
    --seq-len ${SEQ_LEN} \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRAD_ACCUM} \
    --learning-rate ${LR} \
    --min-lr ${MIN_LR} \
    --warmup-steps ${WARMUP_STEPS} \
    --max-steps ${MAX_STEPS} \
    --output-dir ${OUTPUT_DIR} \
    --log-interval 5 \
    --save-interval 50 \
    --bf16 \
    --wandb \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name ${RUN_NAME}

echo "Test completed!"




