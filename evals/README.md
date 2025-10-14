# DeltaNet Evaluation Toolkit

This directory contains all DeltaNet model evaluation related scripts and tools.

## File Structure

```
evals/
├── deltanet_eval.py          # Unified evaluation entry script
├── harness.py                # lm-evaluation-harness wrapper
├── ppl.py                    # Perplexity evaluation script
├── test_deltanet.py          # Model testing script
├── simple_eval_deltanet.py   # Simple text generation evaluation
├── quick_eval_deltanet.py    # Quick standard evaluation
├── evaluate_deltanet.py      # Full standard evaluation
├── evaluation_results.md     # 📊 Evaluation results report
├── evaluation_results.json   # 📊 Raw evaluation results (JSON)
├── README.md                 # Detailed documentation
├── DeltaNet_Evaluation_Guide.md  # Complete usage guide
└── QUICK_START.md            # Quick start guide
```

## 📊 Latest Evaluation Results

**DeltaNet 1.3B Model Performance on WikiText-2**:

| Metric | Value | Description |
|--------|-------|-------------|
| Word Perplexity | **16.71** | Word-level perplexity, lower is better |
| Byte Perplexity | **1.69** | Byte-level perplexity, lower is better |
| Bits per Byte | **0.76** | Compression efficiency, lower is better |

✅ **Evaluation Status**: Successfully completed  
📅 **Evaluation Date**: 2025-10-14  
🔧 **Model**: fla-hub/delta_net-1.3B-100B  
📋 **Task**: WikiText-2  

For detailed results, please see [evaluation_results.md](evaluation_results.md)

## Quick Start

### 1. Unified Entry Script (Recommended)

```bash
# Test model loading
python deltanet_eval.py test

# Simple text generation evaluation
python deltanet_eval.py simple

# Quick standard evaluation
python deltanet_eval.py quick --tasks wikitext,lambada_openai

# Full standard evaluation
python deltanet_eval.py full --model fla-hub/delta_net-1.3B-100B

# Perplexity evaluation
python deltanet_eval.py perplexity --data fla-hub/pg19
```

### 2. Direct Use of Individual Scripts

```bash
# Test model
python test_deltanet.py

# Simple evaluation
python simple_eval_deltanet.py

# Quick evaluation
python quick_eval_deltanet.py --tasks wikitext --batch_size 16

# Full evaluation
python evaluate_deltanet.py --model fla-hub/delta_net-1.3B-100B

# Perplexity evaluation
python ppl.py --path fla-hub/delta_net-1.3B-100B --data fla-hub/pg19
```

## Available DeltaNet Models

- `fla-hub/delta_net-1.3B-100B` (1.3B parameters) - Recommended for testing
- `fla-hub/delta_net-2.7B-100B` (2.7B parameters) - Larger model
- `fla-hub/delta_net-1.3B-8K-100B` (1.3B parameters, 8K context) - Long context version

## Evaluation Types

### 1. Test Evaluation (`test`)
- Verify model can be loaded normally
- Check basic configuration information
- No actual inference performed

### 2. Simple Evaluation (`simple`)
- Basic text generation testing
- Test generation effects with multiple prompts
- Display generation time and results

### 3. Quick Evaluation (`quick`)
- Standard evaluation using lm-evaluation-harness
- Run few tasks (e.g., wikitext, lambada_openai)
- Suitable for quick model performance verification

### 4. Full Evaluation (`full`)
- Run complete standard evaluation suite
- Includes multiple tasks: wikitext, lambada_openai, piqa, hellaswag, winogrande, arc_easy, arc_challenge, boolq, sciq, copa, openbookqa
- Generate detailed evaluation reports

### 5. Perplexity Evaluation (`perplexity`)
- Specifically for evaluating language model perplexity
- Supports long sequence evaluation
- Provides block-level perplexity analysis

## Environment Requirements

Ensure the `fla` conda environment is activated:

```bash
conda activate fla
```

## Notes

1. **GPU Memory**: 1.3B model requires ~3-4GB GPU memory, 2.7B model requires ~6-8GB
2. **Evaluation Time**: Full evaluation may take longer, recommend using quick evaluation first for verification
3. **Compatibility**: Fixed transformers 4.57.0 compatibility issues

## Troubleshooting

If you encounter issues, please refer to the detailed instructions in `DeltaNet_Evaluation_Guide.md`.
