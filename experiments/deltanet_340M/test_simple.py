#!/usr/bin/env python
# Simple test to isolate the issue

import torch
import json
from fla.models import DeltaNetConfig, DeltaNetForCausalLM

print("Loading config...")
with open("configs/deltanet_340M.json", 'r') as f:
    config_dict = json.load(f)

config = DeltaNetConfig(**config_dict)
print(f"Config loaded: {config.hidden_size}, {config.num_hidden_layers} layers")

print("\nInitializing model...")
model = DeltaNetForCausalLM(config)
model = model.cuda()
print(f"Model on cuda: {next(model.parameters()).device}")

# Test forward pass
print("\nTesting forward pass...")
batch_size = 1
seq_len = 16
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda')

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids)
        
print(f"✓ Forward pass successful!")
print(f"Output shape: {outputs.logits.shape}")
print(f"Loss: N/A (no labels provided)")

