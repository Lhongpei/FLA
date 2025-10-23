#!/bin/bash

# Training script for DeltaNet 340M (baseline without OSLA)
# This script launches distributed training on 8 GPUs

set -e

# Configuration
# Use 2 truly contiguous GPUs (0,1) to avoid NCCL issues
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2

# Training parameters  
BATCH_SIZE=2
GRAD_ACCUM=16
SEQ_LEN=2048
MAX_STEPS=20000
WARMUP_STEPS=1000
LR=3e-4
MIN_LR=3e-5

# Dataset
DATASET_NAME="fla-hub/pg19"
DATASET_PATH="/data1/la_group/.cache/huggingface/datasets/fla-hub___pg19/default/0.0.0/217f9837c7bc0f95e57984ffbfead40939abc451"

# Output
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/deltanet_340M.json"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/deltanet_340M_baseline"
WANDB_PROJECT="fla-deltanet-osla-comparison"
RUN_NAME="deltanet-340M-baseline-pg19"

echo "============================================"
echo "Training DeltaNet 340M (Baseline)"
echo "============================================"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET_NAME}"
echo "Batch size: ${BATCH_SIZE} x ${GRAD_ACCUM} x 2 GPUs = $(($BATCH_SIZE * $GRAD_ACCUM * 2))"
echo "Sequence length: ${SEQ_LEN}"
echo "Max steps: ${MAX_STEPS}"
echo "============================================"

# Activate conda environment
source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate fla

# Launch training with torchrun
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    ${SCRIPT_DIR}/train_deltanet.py \
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
    --log-interval 10 \
    --save-interval 2000 \
    --bf16 \
    --wandb \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name ${RUN_NAME}

echo "Training completed!"

