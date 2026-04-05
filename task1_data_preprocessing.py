"""
Task 1: Data Preprocessing & Tokenization
==========================================
Why this works on M1:
- Character-level tokenization minimizes memory footprint (only unique chars as tokens)
- PyTorch Dataset allows lazy loading - data loaded batch-by-batch, not all at once
- Moving data to MPS device frees CPU RAM, keeping more available for other processes
- Small batch sizes (32-64) prevent RAM spikes during training
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, text):
        """Initialize tokenizer with vocabulary from text."""
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        
    def encode(self, text):
        """Convert text to token indices."""
        return [self.char_to_idx[c] for c in text]
    
    def decode(self, indices):
        """Convert token indices back to text."""
        return ''.join([self.idx_to_char[i] for i in indices])


class TechNewsDataset(Dataset):
    """PyTorch Dataset for tech news data."""
    
    def __init__(self, text, tokenizer, block_size=64):
        """
        Args:
            text: Raw text data
            tokenizer: CharTokenizer instance
            block_size: Sequence length for each sample
        """
        self.tokens = tokenizer.encode(text)
        self.block_size = block_size
        
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        """Return input and target sequences."""
        x = torch.tensor(self.tokens[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y


def load_tech_news_data(file_path="tech_news.txt", block_size=64, batch_size=32):
    """
    Load and prepare data for training.
    
    Returns:
        DataLoader, CharTokenizer, device
    """
    # Check if MPS is available on M1 Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Metal Performance Shaders (MPS) for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("⚠ MPS not available, falling back to CPU")
    
    # Load text data
    if not Path(file_path).exists():
        print(f"Creating sample {file_path}...")
        sample_text = """
        Apple M1 chip delivers exceptional GPU performance. 
        New AI models run efficiently on M1 MacBooks.
        Metal Performance Shaders enable fast neural network training.
        Tech trends show ARM chips dominating mobile and desktop.
        Machine learning on Apple Silicon achieves 3x speedup.
        Deep learning frameworks now support Apple Silicon natively.
        PyTorch and TensorFlow run efficiently on M1 devices.
        Transformer models train faster with Metal acceleration.
        Natural language processing benefits from GPU optimization.
        Computer vision tasks execute at remarkable speeds on M1.
        Artificial intelligence is becoming more accessible to developers.
        Edge computing brings intelligence to local devices.
        Neural networks learn patterns from data efficiently.
        Optimization techniques improve model inference speed.
        Data scientists now prefer M1 MacBooks for development.
        Kubernetes deployments work seamlessly on M1 clusters.
        Cloud computing integrates with local edge processing.
        Real-time inference systems demand high performance.
        Quantization reduces model size without losing accuracy.
        Distributed training scales across multiple devices.
        """
        with open(file_path, 'w') as f:
            f.write(sample_text * 50)  # Repeat more for better training data
    
    with open(file_path, 'r') as f:
        text = f.read()
    
    print(f"✓ Loaded {len(text):,} characters from {file_path}")
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    print(f"✓ Vocabulary size: {tokenizer.vocab_size} unique characters")
    
    # Create dataset and dataloader
    dataset = TechNewsDataset(text, tokenizer, block_size=block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # M1: Set to 0 to avoid issues with multiprocessing
    )
    
    print(f"✓ Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
    print(f"✓ Samples per epoch: {len(dataloader)}")
    
    return dataloader, tokenizer, device


if __name__ == "__main__":
    # Test Task 1
    print("=" * 60)
    print("TASK 1: Data Preprocessing & Tokenization")
    print("=" * 60)
    
    dataloader, tokenizer, device = load_tech_news_data(block_size=64, batch_size=32)
    
    # Test a batch
    print("\n✓ Testing batch loading...")
    for x, y in dataloader:
        print(f"  Input shape: {x.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  First 10 tokens: {x[0][:10]}")
        print(f"  Sample text: '{tokenizer.decode(x[0][:20].tolist())}'")
        break
    
    print("\n✓ Task 1 Complete! Ready for Task 2.")
