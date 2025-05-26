# GPT-2 Dutch Language Model Training

This repository contains scripts for training a GPT-2 language model from scratch on Dutch text data. The project uses the `BramVanroy/wikipedia_culturax_dutch` dataset from Hugging Face and is designed to be modular, easy to use, and fast to iterate on.

## Overview

The project is divided into three main components:

1. **Data Preprocessing**: Download and prepare the Dutch dataset for training
2. **Model Training**: Train a GPT-2 model from scratch with configurable parameters
3. **Inference**: Generate text with the trained model

Each component is implemented as a separate script to allow for easy experimentation and iteration.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.12+
- Accelerate 0.26+
- CUDA-compatible GPU (at least one A10G for small models)

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate wandb numpy tqdm
```

## Hardware Setup

This project is optimized for an AWS g5.12xlarge instance with 4x A10G GPUs. The scripts automatically detect and use all available GPUs for distributed training.

## Scripts

### 1. Data Preprocessing (`preprocess.py`)

This script downloads and preprocesses the Dutch Wikipedia dataset for GPT-2 training.

```bash
python preprocess.py --dataset-size 1M --chunk-size 1024 --test-split 0.002
```

Key parameters:
- `--dataset-size`: Size of the dataset to use (1M, 10M, 100M, 1B, 10B tokens)
- `--chunk-size`: Size of text chunks for training (default: 1024)
- `--test-split`: Fraction of data to use for testing (default: 0.002)
- `--output-dir`: Directory to save processed data (default: "processed_data")
- `--num-proc`: Number of processes for dataset processing (default: 8)

Expected runtime:
- 1M tokens: ~1 minute
- 10M tokens: ~5 minutes
- 100M tokens: ~20-30 minutes
- 1B tokens: ~3-4 hours
- 10B tokens: ~24+ hours

### 2. Model Training (`training.py`)

This script trains a GPT-2 model on the preprocessed data using distributed training across multiple GPUs.

**IMPORTANT**: Always use `accelerate launch` for multi-GPU training to avoid memory issues:

```bash
accelerate launch training.py --dataset-path processed_data/chunked_1M --model-size small
```

Key parameters:
- `--dataset-path`: Path to preprocessed dataset (required)
- `--model-size`: Size of GPT-2 model to train (small, medium, large)
- `--output-dir`: Directory to save trained model (default: "trained_models")
- `--epochs`: Number of training epochs (default: 1)
- `--batch-size`: Batch size per GPU for training (default: 8)
- `--gradient-accumulation-steps`: Number of updates steps to accumulate before backward pass (default: 1)
- `--learning-rate`: Initial learning rate (default: 5e-5)
- `--use-wandb`: Use Weights & Biases for logging

Model sizes and approximate parameters:
- Small: ~124M parameters (12 layers, 768 embedding dim, 12 attention heads)
- Medium: ~355M parameters (24 layers, 1024 embedding dim, 16 attention heads)
- Large: ~774M parameters (36 layers, 1280 embedding dim, 20 attention heads)

Expected training time on g5.12xlarge (4x A10G):
- Small model, 1M tokens: ~10-15 minutes
- Small model, 10M tokens: ~1-2 hours
- Small model, 100M tokens: ~10-12 hours
- Medium model, 1M tokens: ~30-45 minutes
- Medium model, 10M tokens: ~4-6 hours
- Large model, 1M tokens: ~1-2 hours

### 3. Inference (`inference.py`)

This script generates text using the trained model.

```bash
python inference.py --model-path trained_models/gpt2-dutch-small-latest/final_model
```

Key parameters:
- `--model-path`: Path to the trained model (required)
- `--prompt`: Text prompt for generation
- `--prompts-file`: Path to a file containing prompts (one per line)
- `--max-length`: Maximum length of generated text (default: 100)
- `--temperature`: Sampling temperature (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 50)
- `--top-p`: Top-p (nucleus) sampling parameter (default: 0.95)
- `--interactive`: Run in interactive mode, prompting for user input
- `--benchmark`: Run benchmarking on generation speed

Expected generation speed:
- Small model: ~10-20 tokens per second on a single A10G GPU
- Medium model: ~5-10 tokens per second on a single A10G GPU
- Large model: ~2-5 tokens per second on a single A10G GPU

## Training Process

A full training workflow typically follows these steps:

1. **Preprocess the data**: Start with a small dataset (1M tokens) for quick experimentation
   ```bash
   python preprocess.py --dataset-size 1M
   ```

2. **Train a small model**: Start with a small model to verify the pipeline
   ```bash
   accelerate launch training.py --dataset-path processed_data/chunked_1M --model-size small --epochs 3
   ```

3. **Test the model**: Generate text to evaluate model quality
   ```bash
   python inference.py --model-path trained_models/gpt2-dutch-small-latest/final_model --interactive
   ```

4. **Scale up**: Increase dataset size and model size for better results
   ```bash
   python preprocess.py --dataset-size 10M
   accelerate launch training.py --dataset-path processed_data/chunked_10M --model-size medium
   ```

## Performance Optimization

### Multi-GPU Training

The training script uses Hugging Face Accelerate for distributed training across all available GPUs. **Always use `accelerate launch`** instead of running the script directly.

**Setup (first time only):**
```bash
accelerate config
```
Or use the automatic default configuration for 4-GPU training.

**Training commands:**
```bash
# For small datasets (reduce batch size to avoid OOM)
accelerate launch training.py --dataset-path processed_data/chunked_1M --model-size small --batch-size 4

# For larger datasets
accelerate launch training.py --dataset-path processed_data/chunked_100M --model-size small --batch-size 4 --gradient-accumulation-steps 4
```

**Performance tips:**
- Use smaller batch sizes (4-8) per GPU to avoid memory issues
- Enable gradient accumulation for larger effective batch sizes (e.g., `--gradient-accumulation-steps 4`)
- Mixed precision training is enabled by default

### Dataset Processing

For large datasets:

- Increase the number of processing workers (e.g., `--num-proc 16`)
- Process data in smaller chunks if memory is limited
- The current preprocessing script can handle datasets up to 10B tokens efficiently

## Scaling to Multiple Instances

For training larger models or using larger datasets:

1. Use AWS S3 to store preprocessed data and model checkpoints
2. Consider using SageMaker for distributed training across multiple instances
3. For production-scale training, look into frameworks like DeepSpeed or Megatron-LM

## Monitoring and Debugging

The training script supports Weights & Biases (wandb) for experiment tracking:

```bash
accelerate launch training.py --use-wandb --wandb-project gpt2-dutch
```

This will log:
- Training and validation loss
- Learning rate schedule
- GPU memory usage
- Training speed (samples/second)

## Conclusion

This project provides a streamlined pipeline for training GPT-2 models on Dutch text. By separating preprocessing, training, and inference into distinct scripts, it's easy to experiment with different configurations and scale up as needed.

The modular design allows you to:
1. Start small and iterate quickly
2. Gradually scale up to larger datasets and models
3. Track experiments and compare different configurations
4. Generate text with your trained models

Happy training!