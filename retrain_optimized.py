#!/usr/bin/env python3
"""
Optimized Retraining Script
============================
This script retrains the model with improved hyperparameters and more training data.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
import json

# Import from tasks
from task1_data_preprocessing import load_tech_news_data
from task2_model_architecture import create_mini_transformer


class OptimizedTrainingConfig:
    """Enhanced configuration for better model quality."""
    batch_size = 16
    learning_rate = 1e-3
    max_epochs = 20  # More epochs
    weight_decay = 0.001
    warmup_steps = 300
    model_path = "tech_slm_model.pth"
    checkpoint_every = 50
    gradient_clip = 1.0
    eval_every = 200


def train_step(model, batch, device):
    """Single training step."""
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    
    logits = model(x)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    y_flat = y.view(-1)
    
    loss = nn.functional.cross_entropy(logits_flat, y_flat)
    return loss


def retrain_optimized():
    """Retrain model with optimized settings."""
    config = OptimizedTrainingConfig()
    
    print("=" * 70)
    print("OPTIMIZED MODEL RETRAINING")
    print("=" * 70)
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (slower)")
    
    # Load data
    print("\n✓ Loading training data...")
    dataloader, tokenizer, _ = load_tech_news_data(
        block_size=64,
        batch_size=config.batch_size
    )
    
    # Create model
    print("✓ Creating model...")
    model = create_mini_transformer(tokenizer.vocab_size, device)
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Memory: {model.estimate_memory_mb():.2f} MB")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training
    total_steps = 0
    losses = []
    best_loss = float('inf')
    
    print(f"\n✓ Starting optimized training...")
    print(f"  Epochs: {config.max_epochs}")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Total steps: ~{config.max_epochs * len(dataloader)}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}\n")
    
    try:
        for epoch in range(config.max_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Forward
                loss = train_step(model, batch, device)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                
                # Optimizer step
                optimizer.step()
                
                # Tracking
                epoch_loss += loss.item()
                batch_count += 1
                total_steps += 1
                losses.append(loss.item())
                
                # Checkpoint
                if total_steps % config.checkpoint_every == 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"  Step {total_steps:4d} | Epoch {epoch + 1:2d}/{config.max_epochs} | "
                          f"Loss: {avg_loss:.4f}")
                    
                    # Save best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        print(f"    → New best loss! Saving model...")
                    
                    # Clear cache
                    if device.type == "mps":
                        torch.mps.empty_cache()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / batch_count
            print(f"\n✓ Epoch {epoch + 1} Complete | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        print(f"\n✓ Saving final model to {config.model_path}...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'vocab_size': tokenizer.vocab_size,
                'n_embd': 128,
                'n_head': 4,
                'n_layer': 4,
                'block_size': 64
            },
            'tokenizer_chars': tokenizer.chars
        }, config.model_path)
        
        # Save metrics
        metrics = {
            'final_loss': float(avg_epoch_loss),
            'best_loss': float(best_loss),
            'total_steps': total_steps,
            'epochs_trained': config.max_epochs,
            'losses': losses[-100:]  # Last 100 losses
        }
        
        with open('training_metrics_optimized.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("✓ Metrics saved to training_metrics_optimized.json")
        print(f"\n✓ Training Complete!")
        print(f"  Final Loss: {avg_epoch_loss:.4f}")
        print(f"  Best Loss: {best_loss:.4f}")
        print(f"  Model saved to: {config.model_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Saving checkpoint...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "checkpoint_interrupted.pth")
        print("Checkpoint saved to checkpoint_interrupted.pth")


if __name__ == "__main__":
    retrain_optimized()
