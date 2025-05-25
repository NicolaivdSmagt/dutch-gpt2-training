# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a pipeline for training GPT-2 language models from scratch on Dutch text data using the `BramVanroy/wikipedia_culturax_dutch` dataset from Hugging Face. The project is designed to be modular, simple to use, and optimized for fast iteration on an AWS g5.12xlarge instance with 4x A10G GPUs.

## Core Components

The project consists of three main scripts:

1. **preprocess.py**: Downloads and processes the Dutch Wikipedia dataset
2. **training.py**: Trains a GPT-2 model on the preprocessed data
3. **inference.py**: Generates text using the trained model

## Common Commands

### Data Preprocessing

```bash
# Process a small test dataset (10k tokens) - good for testing
python preprocess.py --dataset-size 10k

# Process a small dataset (1M tokens)
python preprocess.py --dataset-size 1M

# Process a larger dataset (100M tokens)
python preprocess.py --dataset-size 100M --num-proc 16

# Process with custom parameters
python preprocess.py --dataset-size 10M --chunk-size 1024 --test-split 0.002 --output-dir custom_data
```

### Model Training

```bash
# Train a small model on test dataset (10k tokens) - good for testing
python training.py --dataset-path processed_data/chunked_10k --model-size small

# Train a small model on 1M tokens
python training.py --dataset-path processed_data/chunked_1M --model-size small

# Train a medium model on 10M tokens with wandb logging
python training.py --dataset-path processed_data/chunked_10M --model-size medium --use-wandb

# Train with custom parameters
python training.py --dataset-path processed_data/chunked_100M --model-size small --epochs 3 --batch-size 16 --gradient-accumulation-steps 4
```

### Text Generation

```bash
# Generate text in interactive mode
python inference.py --model-path trained_models/gpt2-dutch-small-latest/final_model --interactive

# Generate text with default prompts
python inference.py --model-path trained_models/gpt2-dutch-small-latest/final_model

# Run benchmarking
python inference.py --model-path trained_models/gpt2-dutch-small-latest/final_model --benchmark
```

## Architecture Details

### Data Processing Pipeline

The preprocessing script follows this workflow:
1. Load dataset from Hugging Face (configurable size: 10k, 1M, 10M, 100M, 1B, 10B tokens)
2. Create train/test split
3. Tokenize using GPT-2 tokenizer
4. Concatenate all texts and create fixed-size chunks (default: 1024 tokens)
5. Save processed dataset to disk

**Note**: The chunking uses a concatenation-based approach that combines all texts before splitting into chunks, which is more efficient and stable than per-document chunking.

### Training Pipeline

The training script implements:
1. Configurable model sizes (small, medium, large)
2. Distributed training across available GPUs
3. Mixed-precision training
4. Learning rate scheduling
5. Integration with Weights & Biases (optional)

### Inference System

The inference script provides:
1. Interactive and batch text generation
2. Configurable generation parameters (temperature, top-k, top-p)
3. Performance benchmarking

## Performance Expectations

- **Preprocessing**: 
  - 10k tokens: ~3-5 seconds
  - 1M tokens: ~1 minute
  - 100M tokens: ~20-30 minutes
  - 10B tokens: ~24+ hours

- **Training** (on g5.12xlarge with 4x A10G GPUs):
  - Small model, 10k tokens: ~30 seconds (for testing)
  - Small model, 1M tokens: ~10-15 minutes
  - Small model, 100M tokens: ~10-12 hours
  - Medium model, 10M tokens: ~4-6 hours

- **Inference**:
  - Small model: ~10-20 tokens per second per GPU
  - Medium model: ~5-10 tokens per second per GPU
  - Large model: ~2-5 tokens per second per GPU

## Troubleshooting

### Common Issues and Solutions

**Arrow data batching error during preprocessing:**
- **Issue**: `ArrowInvalid: Column 2 named input_ids expected length X but got length Y`
- **Solution**: Fixed in the latest version by using concatenation-based chunking instead of per-document chunking

**Training fails with "evaluation_strategy" parameter error:**
- **Issue**: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`
- **Solution**: Fixed in the latest version by updating to `eval_strategy` parameter for compatibility with newer transformers versions

**Accelerate version compatibility:**
- **Issue**: `ImportError: Using the 'Trainer' with 'PyTorch' requires 'accelerate>=0.26.0'`
- **Solution**: Run `pip install 'accelerate>=0.26.0'` to update the package

**Poor text generation quality:**
- **Issue**: Generated text contains random words or nonsensical output
- **Solution**: This is expected when training on small datasets (like 10k tokens). Use larger datasets (1M+ tokens) for better results