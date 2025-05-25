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
# Process a small dataset (1M tokens)
python preprocess.py --dataset-size 1M

# Process a larger dataset (100M tokens)
python preprocess.py --dataset-size 100M --num-proc 16

# Process with custom parameters
python preprocess.py --dataset-size 10M --chunk-size 1024 --test-split 0.002 --output-dir custom_data
```

### Model Training

```bash
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
1. Load dataset from Hugging Face (configurable size)
2. Create train/test split
3. Tokenize using GPT-2 tokenizer
4. Create fixed-size chunks (default: 1024 tokens)
5. Save processed dataset to disk

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
  - 1M tokens: ~1 minute
  - 100M tokens: ~20-30 minutes
  - 10B tokens: ~24+ hours

- **Training** (on g5.12xlarge with 4x A10G GPUs):
  - Small model, 1M tokens: ~10-15 minutes
  - Small model, 100M tokens: ~10-12 hours
  - Medium model, 10M tokens: ~4-6 hours

- **Inference**:
  - Small model: ~10-20 tokens per second per GPU
  - Medium model: ~5-10 tokens per second per GPU
  - Large model: ~2-5 tokens per second per GPU