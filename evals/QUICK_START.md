# DeltaNet Evaluation Toolkit - Quick Start Guide

## 🚀 Quick Start

### 1. Activate Environment
```bash
conda activate fla
cd /home/cyzhou/FLA/evals
```

### 2. Test Model (Recommended First Step)
```bash
python deltanet_eval.py test
```

### 3. Simple Text Generation Evaluation
```bash
python deltanet_eval.py simple
```

### 4. Quick Standard Evaluation
```bash
python deltanet_eval.py quick --tasks wikitext,lambada_openai
```

### 5. Full Standard Evaluation
```bash
python deltanet_eval.py full --model fla-hub/delta_net-1.3B-100B
```

### 6. Perplexity Evaluation
```bash
python deltanet_eval.py perplexity --data fla-hub/pg19
```

## 📋 Available Models

- `fla-hub/delta_net-1.3B-100B` (1.3B parameters) - Recommended
- `fla-hub/delta_net-2.7B-100B` (2.7B parameters) - Larger model
- `fla-hub/delta_net-1.3B-8K-100B` (1.3B parameters, 8K context) - Long context

## 🔧 Common Parameters

- `--model`: Specify model name
- `--tasks`: Specify evaluation tasks
- `--batch_size`: Batch size
- `--device`: Device type (cuda/cpu)
- `--dtype`: Data type (bfloat16/float16/float32)

## 📊 Evaluation Types

| Command | Purpose | Time | Memory |
|---------|---------|------|--------|
| `test` | Test model loading | < 1 min | Low |
| `simple` | Text generation test | 1-2 min | 3-4GB |
| `quick` | Quick standard evaluation | 5-10 min | 3-4GB |
| `full` | Full standard evaluation | 30-60 min | 3-4GB |
| `perplexity` | Perplexity evaluation | 10-30 min | 3-4GB |

## ❓ Help

View all available options:
```bash
python deltanet_eval.py --help
python deltanet_eval.py <command> --help
```

## 📁 File Descriptions

- `deltanet_eval.py` - Unified entry script (recommended)
- `test_deltanet.py` - Model testing
- `simple_eval_deltanet.py` - Simple evaluation
- `quick_eval_deltanet.py` - Quick evaluation
- `evaluate_deltanet.py` - Full evaluation
- `ppl.py` - Perplexity evaluation
- `harness.py` - lm-evaluation-harness wrapper
- `README.md` - Detailed documentation
- `DeltaNet_Evaluation_Guide.md` - Complete usage guide
