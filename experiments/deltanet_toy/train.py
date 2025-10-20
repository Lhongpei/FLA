#!/usr/bin/env python

# -*- coding: utf-8 -*-

import argparse
import itertools
import os
import random
from typing import Dict, Iterator, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from fla.models import DeltaNetConfig, DeltaNetForCausalLM


class SyntheticDataset(Dataset):
    """Synthetic dataset that generates random token sequences."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy DeltaNet training on synthetic data.")
    parser.add_argument("--steps", type=int, default=50, help="Total training steps.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size.")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length for synthetic data.")
    parser.add_argument("--hidden-size", type=int, default=512, help="Model hidden size.")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of DeltaNet blocks.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Linear warmup steps.")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Tokenizer vocabulary size.")
    parser.add_argument("--num-samples", type=int, default=2048, help="Total synthetic samples.")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval in steps.")
    parser.add_argument("--output-dir", type=str, default="experiments/deltanet_toy/checkpoints", help="Checkpoint directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> Tuple[int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def build_dataloader(args: argparse.Namespace) -> Tuple[DataLoader, DistributedSampler]:
    dataset = SyntheticDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    return loader, sampler


def prepare_model(args: argparse.Namespace, device: torch.device) -> torch.nn.parallel.DistributedDataParallel:
    config = DeltaNetConfig(
        attn_mode="chunk",
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_short_conv=True,
        use_gate=False,
        use_osla=True,  # Enable OSLA training mode
        initializer_range=0.02,
    )
    model = DeltaNetForCausalLM(config)
    model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        gradient_as_bucket_view=True,
    )
    return ddp_model


def linear_warmup_lr(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    return base_lr * float(step + 1) / float(max(1, warmup_steps))


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()

    local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed + dist.get_rank())

    loader, sampler = build_dataloader(args)
    model = prepare_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    data_iter: Iterator[Dict[str, torch.Tensor]] = iter(loader)
    epoch = 0

    for step in range(args.steps):
        try:
            batch_dict = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch_dict = next(data_iter)

        input_ids = batch_dict["input_ids"].to(device, non_blocking=True)
        labels = batch_dict["labels"].to(device, non_blocking=True)

        for param_group in optimizer.param_groups:
            param_group["lr"] = linear_warmup_lr(step, args.learning_rate, args.warmup_steps)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels, use_osla=True)
            loss = outputs.loss

        loss.backward()
        optimizer.step()

        if (step + 1) % args.log_interval == 0 and local_rank == 0:
            tokens_per_gpu = args.batch_size * args.seq_len
            total_tokens = tokens_per_gpu * world_size * args.log_interval
            print(
                f"[step {step + 1:04d}] loss={loss.item():.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6f} | "
                f"tokens/interval={total_tokens}",
                flush=True,
            )

    dist.barrier()
    if local_rank == 0:
        ckpt_path = os.path.join(args.output_dir, "deltanet_toy.pt")
        torch.save(model.module.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
