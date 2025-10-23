#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for DeltaNet 340M models (with/without OSLA).
This script supports multi-GPU distributed training and real datasets.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from fla.models import DeltaNetConfig, DeltaNetForCausalLM


class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized data from HuggingFace datasets."""

    def __init__(
        self,
        dataset,
        seq_len: int = 2048,
        tokenizer=None,
    ):
        self.dataset = dataset
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Handle different dataset formats
        if 'input_ids' in item:
            tokens = item['input_ids']
        elif 'text' in item:
            # Tokenize on-the-fly if needed
            tokens = self.tokenizer.encode(item['text'], add_special_tokens=True)
        else:
            raise ValueError(f"Dataset item must contain 'input_ids' or 'text', got: {item.keys()}")
        
        # Ensure we have enough tokens
        if len(tokens) < self.seq_len + 1:
            # Pad if necessary
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        else:
            # Truncate to seq_len + 1
            tokens = tokens[:self.seq_len + 1]
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeltaNet 340M model.")
    
    # Model configuration
    parser.add_argument("--config", type=str, required=True, help="Path to model config JSON file.")
    parser.add_argument("--use-osla", action="store_true", help="Enable OSLA training mode.")
    
    # Data configuration
    parser.add_argument("--dataset-name", type=str, default="fla-hub/pg19", help="HuggingFace dataset name.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to pre-loaded dataset cache.")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length.")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for dataset.")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Linear warmup steps.")
    parser.add_argument("--max-steps", type=int, default=20000, help="Maximum training steps.")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping.")
    
    # Logging and checkpointing
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints.")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval in steps.")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint saving interval.")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval.")
    
    # System configuration
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for optimization.")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="deltanet-340M", help="W&B project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name.")
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple:
    """Initialize distributed training."""
    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        # Single GPU mode
        local_rank = 0
        world_size = 1
        torch.cuda.set_device(local_rank)
    
    return local_rank, world_size


def get_cosine_schedule_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    elif step >= max_steps:
        return min_lr
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cosine_decay


def load_model_config(config_path: str) -> DeltaNetConfig:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return DeltaNetConfig(**config_dict)


def prepare_dataloader(args: argparse.Namespace, tokenizer, world_size: int) -> DataLoader:
    """Prepare training dataloader."""
    # Load dataset
    if args.dataset_path:
        print(f"Loading dataset from disk: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)
    else:
        print(f"Loading dataset from HuggingFace: {args.dataset_name}")
        dataset = load_dataset(
            args.dataset_name,
            split=args.dataset_split,
            streaming=args.streaming,
        )
    
    # Create dataset wrapper
    train_dataset = PreTokenizedDataset(
        dataset=dataset,
        seq_len=args.seq_len,
        tokenizer=tokenizer,
    )
    
    # Create sampler for distributed training (or regular shuffle for single GPU)
    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader, sampler


def main():
    # Parse arguments
    args = parse_args()
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup distributed training
    local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    is_main_process = local_rank == 0
    
    # Set random seed
    set_seed(args.seed + local_rank)
    
    # Create output directory
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save args
        with open(os.path.join(args.output_dir, "train_args.json"), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Initialize Weights & Biases
    if args.wandb and is_main_process:
        import wandb
        run_name = args.wandb_run_name or f"deltanet-340M-{'osla' if args.use_osla else 'baseline'}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
    
    # Load model configuration
    config = load_model_config(args.config)
    if args.use_osla:
        config.use_osla = True
    
    if is_main_process:
        print(f"Model configuration:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Num heads: {config.num_heads}")
        print(f"  Use OSLA: {config.use_osla}")
        print(f"  Vocab size: {config.vocab_size}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # Create model
    if is_main_process:
        print("Initializing model...")
    
    model = DeltaNetForCausalLM(config)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if is_main_process:
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Wrap model with DDP (only in distributed mode)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
        )
    
    # Optionally compile model
    if args.compile:
        if is_main_process:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Prepare dataloader
    if is_main_process:
        print("Preparing dataloader...")
    
    dataloader, sampler = prepare_dataloader(args, tokenizer, world_size)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Training loop
    if is_main_process:
        print(f"\nStarting training for {args.max_steps} steps...")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Global batch size: {args.batch_size * args.gradient_accumulation_steps * world_size}")
        print(f"Total tokens per step: {args.batch_size * args.gradient_accumulation_steps * world_size * args.seq_len / 1e6:.2f}M")
    
    model.train()
    data_iter = iter(dataloader)
    epoch = 0
    
    # Training metrics
    total_loss = 0.0
    total_tokens = 0
    
    for step in range(args.max_steps):
        # Update learning rate
        lr = get_cosine_schedule_with_warmup(
            step,
            args.warmup_steps,
            args.max_steps,
            args.learning_rate,
            args.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Accumulate gradients
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float32):
                if args.use_osla:
                    outputs = model(input_ids=input_ids, labels=labels, use_osla=True)
                else:
                    outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Track metrics
            total_loss += loss.item() * args.gradient_accumulation_steps
            total_tokens += input_ids.numel()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if (step + 1) % args.log_interval == 0 and is_main_process:
            avg_loss = total_loss / args.log_interval
            tokens_per_sec = total_tokens / args.log_interval
            
            log_msg = (
                f"[Step {step + 1:05d}/{args.max_steps}] "
                f"loss={avg_loss:.4f} | "
                f"lr={lr:.6f} | "
                f"tokens/s={tokens_per_sec:.0f}"
            )
            print(log_msg, flush=True)
            
            if args.wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": step + 1,
                })
            
            total_loss = 0.0
            total_tokens = 0
        
        # Save checkpoint
        if (step + 1) % args.save_interval == 0 and is_main_process:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step + 1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model (handle both DDP and non-DDP cases)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(checkpoint_path)
            
            # Save optimizer state
            torch.save({
                'optimizer': optimizer.state_dict(),
                'step': step + 1,
                'epoch': epoch,
            }, os.path.join(checkpoint_path, "optimizer.pt"))
            
            print(f"Checkpoint saved to {checkpoint_path}", flush=True)
    
    # Final checkpoint
    if is_main_process:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        # Handle both DDP and non-DDP models
        model_to_save = model.module if world_size > 1 else model
        model_to_save.save_pretrained(final_path)
        print(f"Final model saved to {final_path}", flush=True)
    
    # Cleanup
    if args.wandb and is_main_process:
        wandb.finish()
    
    # Only destroy process group if we initialized it (distributed mode)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
