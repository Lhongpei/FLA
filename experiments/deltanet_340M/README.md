# DeltaNet 340M Training Experiments

This directory contains training scripts and configurations for comparing DeltaNet baseline and DeltaNet with OSLA on 340M parameter models.

## Environment Setup

The training environment uses a conda environment named `fla`:

```bash
conda activate fla
```

All dependencies are already installed including PyTorch, Transformers, Datasets, etc.

## Model Configurations

Two model configurations are provided:

1. **deltanet_340M.json**: Baseline DeltaNet (use_osla=false)
2. **deltanet_340M_osla.json**: DeltaNet with OSLA (use_osla=true)

Both configurations have:
- Hidden size: 1024
- Number of layers: 24
- Number of heads: 8
- Vocabulary size: 32000
- Total parameters: ~340M

## Training Scripts

### Baseline DeltaNet

```bash
bash train_baseline.sh
```

This trains the baseline DeltaNet model without OSLA.

### DeltaNet with OSLA

```bash
bash train_osla.sh
```

This trains the DeltaNet model with OSLA enabled.

## Training Configuration

Both scripts use the same hyperparameters for fair comparison:

- **GPUs**: 8x H100
- **Batch size**: 4 per GPU
- **Gradient accumulation**: 8 steps
- **Global batch size**: 256 (4 × 8 × 8)
- **Sequence length**: 2048
- **Total tokens per step**: 524,288 (~0.5M)
- **Learning rate**: 3e-4 (peak)
- **Min learning rate**: 3e-5
- **LR schedule**: Cosine with linear warmup
- **Warmup steps**: 1000
- **Max steps**: 20000
- **Total tokens**: ~10.5B tokens
- **Precision**: BF16
- **Weight decay**: 0.1
- **Gradient clipping**: 1.0

## Dataset

Training uses the **PG19** dataset from HuggingFace:
- Dataset: `fla-hub/pg19`
- Cached at: `/data1/la_group/.cache/huggingface/datasets/fla-hub___pg19/`
- Sequence length: 2048 tokens

## Outputs

Training outputs are saved to:
- Baseline: `outputs/deltanet_340M_baseline/`
- OSLA: `outputs/deltanet_340M_osla/`

Each output directory contains:
- Training arguments (`train_args.json`)
- Checkpoints every 2000 steps
- Final model checkpoint

## Monitoring

Training metrics are logged to Weights & Biases:
- Project: `deltanet-340M`
- Runs: 
  - `deltanet-340M-baseline`
  - `deltanet-340M-osla`

Logged metrics include:
- Training loss
- Learning rate
- Tokens per second
- Step number

## Quick Start

To start training both models sequentially:

```bash
# Train baseline first
bash train_baseline.sh

# Then train OSLA version
bash train_osla.sh
```

Or run them in parallel on different nodes/tmux sessions if you have multiple sets of 8 GPUs.

## Expected Results

The OSLA version should demonstrate:
- Comparable or better perplexity/loss
- Similar or improved training stability
- Efficient training with the chunk-based attention mechanism

Monitor the W&B dashboard to compare training curves between the two approaches.



