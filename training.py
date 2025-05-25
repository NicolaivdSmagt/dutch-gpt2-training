#!/usr/bin/env python3
"""
Dutch GPT-2 Training: Training Script

This script trains a GPT-2 model on preprocessed Dutch text data.
It loads the processed dataset, initializes a GPT-2 model, and trains it
using Hugging Face's Trainer API with distributed training on multiple GPUs.

Features:
- Configurable model size (small, medium, large)
- Distributed training across multiple GPUs
- Learning rate scheduling
- Gradient accumulation for large batch training
- Checkpointing and model saving
- Optional wandb integration for monitoring

Usage:
  python training.py --dataset-path processed_data/chunked_1M --model-size small

Author: Claude
"""

import os
import argparse
import math
import time
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Optional: Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Model size configurations
MODEL_CONFIGS = {
    "small": {  # ~124M params (standard GPT-2)
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12
    },
    "medium": {  # ~355M params
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16
    },
    "large": {  # ~774M params
        "n_embd": 1280,
        "n_layer": 36,
        "n_head": 20
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GPT-2 on Dutch text")
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to preprocessed dataset"
    )
    
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Size of GPT-2 model to train"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_models",
        help="Directory to save trained model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per GPU/CPU for training"
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward pass"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Proportion of training for learning rate warmup"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay to apply"
    )
    
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gpt2-dutch",
        help="Weights & Biases project name"
    )
    
    parser.add_argument(
        "--save-steps",
        type=int,
        default=5000,
        help="Save checkpoint every X updates steps"
    )
    
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate model every X updates steps"
    )
    
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Limit the total amount of checkpoints"
    )
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    model_output_dir = os.path.join(
        args.output_dir, 
        f"gpt2-dutch-{args.model_size}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Proceeding without wandb logging.")
        else:
            wandb.init(
                project=args.wandb_project,
                name=f"gpt2-{args.model_size}-dutch",
                config=vars(args)
            )
    
    # Load preprocessed dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset loaded. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    print(f"Initializing GPT-2 {args.model_size} model...")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_embd=MODEL_CONFIGS[args.model_size]["n_embd"],
        n_layer=MODEL_CONFIGS[args.model_size]["n_layer"],
        n_head=MODEL_CONFIGS[args.model_size]["n_head"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    print(f"Model initialized with {model.num_parameters():,} parameters")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_dir=os.path.join(model_output_dir, "logs"),
        logging_steps=100,
        report_to="wandb" if args.use_wandb and WANDB_AVAILABLE else "none",
        # Enable fp16 for faster training on GPUs
        fp16=True,
        # Enable distributed training
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    start_time = time.time()
    trainer.train()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    
    # Save final model
    final_model_path = os.path.join(model_output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Run final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    eval_perplexity = math.exp(eval_results["eval_loss"])
    print(f"Final perplexity: {eval_perplexity:.2f}")
    
    # Log final metrics to wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "final_perplexity": eval_perplexity,
            "training_time_hours": training_time/3600
        })
        wandb.finish()

if __name__ == "__main__":
    main()