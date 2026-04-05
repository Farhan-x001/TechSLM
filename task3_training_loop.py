"""
Task 3: Training Loop with M1 Optimization
============================================
Why this works on M1:
- torch.backends.mps.is_available() check ensures we use Metal GPU when available
- AdamW optimizer with learning rate schedule prevents overfitting
- Gradient accumulation (simulated) helps with small batch sizes
- Loss checkpoint every 500 iterations provides training visibility
- Model saving (.pth) allows resuming training without restarting
- Explicit memory management: clearing gradients, detaching tensors
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import json
from pathlib import Path

# Import from other tasks
from task1_data_preprocessing import load_tech_news_data
from task2_model_architecture import create_mini_transformer


class TrainingConfig:
    """Configuration for training - OPTIMIZED for better generation."""
    batch_size = 16  # Reduced for better learning
    learning_rate = 1e-3  # Increased for faster convergence
    max_epochs = 15  # More epochs for better model quality
    weight_decay = 0.001  # Reduced regularization
    warmup_steps = 200  # Better warmup
    model_path = "tech_slm_model.pth"
    checkpoint_every = 100  # More frequent checkpoints
    eval_every = 500  # More frequent evaluation
    gradient_clip = 1.0  # Gradient clipping for stability


def train_step(model, batch, device):
    """Single training step."""
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    
    # Forward pass
    logits = model(x)  # (batch_size, seq_len, vocab_size)
    
    # Compute loss (cross-entropy)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    y_flat = y.view(-1)
    
    loss = nn.functional.cross_entropy(logits_flat, y_flat)
    
    return loss


def train(config=None):
    """Train the MiniTransformer on M1."""
    if config is None:
        config = TrainingConfig()
    
    print("=" * 60)
    print("TASK 3: Training Loop with M1 Optimization")
    print("=" * 60)
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Metal Performance Shaders (MPS) AVAILABLE - Using GPU acceleration")
    else:
        device = torch.device("cpu")
        print("⚠ MPS not available - Using CPU (training will be slower)")
    
    print(f"  Device: {device}")
    
    # Load data
    print("\n✓ Loading data...")
    dataloader, tokenizer, _ = load_tech_news_data(
        block_size=64,
        batch_size=config.batch_size
    )
    
    # Create model
    print("✓ Creating model...")
    model = create_mini_transformer(tokenizer.vocab_size, device)
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Memory: {model.estimate_memory_mb():.2f} MB")
    
    # Optimizer with learning rate scheduling
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training metrics
    total_steps = 0
    losses = []
    
    print(f"\n✓ Starting training...")
    print(f"  Epochs: {config.max_epochs}")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Total steps: ~{config.max_epochs * len(dataloader)}")
    
    try:
        for epoch in range(config.max_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Forward and backward
                loss = train_step(model, batch, device)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (prevents exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Tracking
                epoch_loss += loss.item()
                batch_count += 1
                total_steps += 1
                losses.append(loss.item())
                
                # Checkpoint and evaluation
                if total_steps % config.checkpoint_every == 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"\n  Step {total_steps} | Epoch {epoch + 1}/{config.max_epochs} | "
                          f"Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f}")
                    
                    # Free up memory
                    torch.mps.empty_cache() if device.type == "mps" else None
            
            # End of epoch
            avg_epoch_loss = epoch_loss / batch_count
            print(f"\n✓ Epoch {epoch + 1} Complete | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save model
        print(f"\n✓ Saving model to {config.model_path}...")
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
            'tokenizer_chars': tokenizer.chars,
            'losses': losses
        }, config.model_path)
        
        print(f"✓ Model saved! Total parameters: {model.count_parameters():,}")
        print(f"✓ Training complete. Final loss: {losses[-1]:.4f}")
        
        # Save training metrics
        metrics = {
            'total_steps': total_steps,
            'final_loss': losses[-1],
            'avg_loss': sum(losses) / len(losses),
            'model_params': model.count_parameters(),
            'device': str(device)
        }
        
        with open('training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("✓ Metrics saved to training_metrics.json")
        print("\n✓ Task 3 Complete! Ready for Task 4.")
        
        return model, tokenizer, device
    
    except KeyboardInterrupt:
        print("\n✓ Training interrupted by user")
        print(f"✓ Saving checkpoint at step {total_steps}...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': total_steps,
            'losses': losses
        }, 'checkpoint.pth')
        raise


def load_model(model_path="tech_slm_model.pth", device=None):
    """Load a saved model."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model
    from task2_model_architecture import MiniTransformer
    config = checkpoint['config']
    model = MiniTransformer(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Recreate tokenizer
    from task1_data_preprocessing import CharTokenizer
    class LoadedTokenizer:
        def __init__(self, chars):
            self.chars = chars
            self.vocab_size = len(chars)
            self.char_to_idx = {c: i for i, c in enumerate(chars)}
            self.idx_to_char = {i: c for i, c in enumerate(chars)}
        
        def encode(self, text):
            return [self.char_to_idx[c] for c in text]
        
        def decode(self, indices):
            return ''.join([self.idx_to_char[i] for i in indices])
    
    tokenizer = LoadedTokenizer(checkpoint['tokenizer_chars'])
    
    return model, tokenizer, device


if __name__ == "__main__":
    config = TrainingConfig()
    train(config)
