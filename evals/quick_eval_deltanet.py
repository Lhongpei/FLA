#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeltaNet Quick Evaluation Script
Run basic language model evaluation tasks
"""

import argparse
import os
import sys

# Ensure FLA library is properly imported and registered
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fla
from fla.models import DeltaNetConfig, DeltaNetForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register('delta_net', DeltaNetConfig)
AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM)

def main():
    parser = argparse.ArgumentParser(description='DeltaNet Quick Evaluation')
    parser.add_argument('--model', type=str, default='fla-hub/delta_net-1.3B-100B',
                       help='DeltaNet model name')
    parser.add_argument('--tasks', type=str, default='wikitext,lambada_openai',
                       help='Evaluation tasks (default: wikitext,lambada_openai)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device type')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DeltaNet Quick Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Build evaluation command
    cmd = f"""python -m evals.harness \\
    --model hf \\
    --model_args pretrained={args.model},dtype=bfloat16,trust_remote_code=True \\
    --tasks {args.tasks} \\
    --batch_size {args.batch_size} \\
    --num_fewshot 0 \\
    --device {args.device} \\
    --show_config \\
    --trust_remote_code"""
    
    print("Executing command:")
    print(cmd)
    print("=" * 80)
    
    # Execute evaluation
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("Evaluation completed!")
    else:
        print("Evaluation failed, exit code:", exit_code)
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
