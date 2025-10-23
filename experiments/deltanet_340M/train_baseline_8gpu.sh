#!/bin/bash
# Training script for DeltaNet 340M Baseline on 8 GPUs
# This script launches distributed training on 8 H100 GPUs

set -e

# Configuration - Use all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8

# NCCL settings for multi-GPU training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=lo
export NCCL_NVLS_ENABLE=0

# Training parameters
BATCH_SIZE=2          # Per GPU batch size
GRAD_ACCUM=4          # Gradient accumulation steps
SEQ_LEN=2048
MAX_STEPS=20000       # Total training steps
WARMUP_STEPS=1000
LR=3e-4
MIN_LR=3e-5

# Dataset
DATASET_NAME="fla-hub/pg19"

# Output
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/deltanet_340M.json"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/deltanet_340M_baseline_8gpu"
WANDB_PROJECT="fla-deltanet-baseline"
RUN_NAME="deltanet-340M-baseline-8gpu-pg19"

echo "============================================"
echo "Training DeltaNet 340M (Baseline) on 8 GPUs"
echo "============================================"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET_NAME}"
echo "Batch size: ${BATCH_SIZE} x ${GRAD_ACCUM} x 8 GPUs = $(($BATCH_SIZE * $GRAD_ACCUM * 8))"
echo "Sequence length: ${SEQ_LEN}"
echo "Max steps: ${MAX_STEPS}"
echo "Global tokens per step: $((${BATCH_SIZE} * ${GRAD_ACCUM} * 8 * ${SEQ_LEN}))"
echo "============================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Activate conda environment
source /home/datagen/anaconda3/etc/profile.d/conda.sh
conda activate fla

# Launch training with torchrun
torchrun \
    --nproc_per_node=8 \
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

echo "============================================"
echo "Training completed!"
echo "============================================"
