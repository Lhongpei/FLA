#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeltaNet Simple Evaluation Script
Direct testing of model's basic inference capabilities
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Ensure FLA library is properly imported and registered
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fla
from fla.models import DeltaNetConfig, DeltaNetForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register('delta_net', DeltaNetConfig)
AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM)

def evaluate_deltanet(model_name='fla-hub/delta_net-1.3B-100B'):
    """Simple evaluation of DeltaNet model"""
    
    print(f"Evaluating model: {model_name}")
    print("=" * 60)
    
    try:
        # Load tokenizer
        print("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("   ✓ Tokenizer loaded successfully")
        
        # Load model
        print("2. Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("   ✓ Model loaded successfully")
        
        # Test text generation
        print("3. Testing text generation...")
        test_prompts = [
            "The future of artificial intelligence is",
            "In the realm of machine learning,",
            "Deep learning has revolutionized"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test {i}: '{prompt}'")
            
            # Encode input
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate text
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            generation_time = time.time() - start_time
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   Generated result: {generated_text}")
            print(f"   Generation time: {generation_time:.2f} seconds")
        
        print("\n✓ Evaluation completed! DeltaNet model works normally")
        
        # Display model information
        print(f"\nModel information:")
        print(f"- Parameters: ~1.3B")
        print(f"- Hidden size: {model.config.hidden_size}")
        print(f"- Number of layers: {model.config.num_hidden_layers}")
        print(f"- Number of attention heads: {model.config.num_heads}")
        print(f"- Vocabulary size: {model.config.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("DeltaNet Simple Evaluation")
    print("=" * 60)
    
    # Evaluate default model
    success = evaluate_deltanet()
    
    if success:
        print("\n" + "=" * 60)
        print("Evaluation successful! DeltaNet model can be used normally")
        print("\nAvailable DeltaNet models:")
        print("  - fla-hub/delta_net-1.3B-100B (1.3B parameters)")
        print("  - fla-hub/delta_net-2.7B-100B (2.7B parameters)")
        print("  - fla-hub/delta_net-1.3B-8K-100B (1.3B parameters, 8K context)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Evaluation failed, please check environment configuration")
        print("=" * 60)

if __name__ == "__main__":
    main()
