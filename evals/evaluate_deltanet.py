#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeltaNet Model Evaluation Script
Standard evaluation of DeltaNet models using lm-evaluation-harness
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure FLA library is properly imported and registered
sys.path.insert(0, str(Path(__file__).parent.parent))

import fla
from fla.models import DeltaNetConfig, DeltaNetForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register('delta_net', DeltaNetConfig)
AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM)

def main():
    parser = argparse.ArgumentParser(description='DeltaNet Model Evaluation')
    parser.add_argument('--model', type=str, default='fla-hub/delta_net-1.3B-100B',
                       help='DeltaNet model name or path')
    parser.add_argument('--tasks', type=str, 
                       default='wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa',
                       help='Evaluation task list, comma-separated')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_fewshot', type=int, default=0,
                       help='Number of few-shot examples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device type (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       help='Data type (float16/bfloat16/float32)')
    parser.add_argument('--output_path', type=str, default='./results',
                       help='Output path for results')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Whether to use multi-GPU evaluation')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    # Build evaluation command
    if args.multi_gpu:
        cmd = f"""accelerate launch -m evals.harness \\
    --output_path {args.output_path} \\
    --model hf \\
    --model_args pretrained={args.model},dtype={args.dtype},max_length={args.max_length},trust_remote_code=True \\
    --tasks {args.tasks} \\
    --batch_size {args.batch_size} \\
    --num_fewshot {args.num_fewshot} \\
    --device {args.device} \\
    --show_config \\
    --trust_remote_code"""
    else:
        cmd = f"""python -m evals.harness \\
    --output_path {args.output_path} \\
    --model hf \\
    --model_args pretrained={args.model},dtype={args.dtype},max_length={args.max_length},trust_remote_code=True \\
    --tasks {args.tasks} \\
    --batch_size {args.batch_size} \\
    --num_fewshot {args.num_fewshot} \\
    --device {args.device} \\
    --show_config \\
    --trust_remote_code"""
    
    print("=" * 80)
    print("DeltaNet Model Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Data type: {args.dtype}")
    print(f"Max length: {args.max_length}")
    print(f"Multi-GPU: {args.multi_gpu}")
    print(f"Output path: {args.output_path}")
    print("=" * 80)
    print("Executing command:")
    print(cmd)
    print("=" * 80)
    
    # Execute evaluation
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("Evaluation completed! Results saved in:", args.output_path)
    else:
        print("Evaluation failed, exit code:", exit_code)
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
