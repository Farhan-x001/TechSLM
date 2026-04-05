"""
Task 2: The 'Mini-Transformer' Architecture
==============================================
Why this works on M1:
- Small parameter count (1M-5M) = lower memory usage and faster training
- n_layer=4, n_head=4, n_embd=128: Optimized for 8GB RAM constraints
- Positional embeddings are lightweight (just addition, no extra parameters)
- Using flash attention patterns (scaled dot-product) for efficiency
- Block_size=64: Short sequences = less memory for attention matrices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        
        # Linear projections
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        
        # Causal mask (lower triangular matrix for autoregressive)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, n_embd)
        Returns:
            output: (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.query(x)  # (batch_size, seq_len, n_embd)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head: (batch_size, seq_len, n_head, head_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        # Now: (batch_size, n_head, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (batch_size, n_head, seq_len, head_dim)
        
        # Concatenate heads: (batch_size, seq_len, n_embd)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        
        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block (attention + feed-forward)."""
    
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, block_size)
        self.feed_forward = FeedForward(n_embd)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # Pre-norm residual connections
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    """
    Tiny Transformer (NanoGPT-style) for M1 Mac.
    
    Parameters for 8GB M1 Mac:
    - n_layer=4: 4 transformer blocks
    - n_head=4: 4 attention heads
    - n_embd=128: 128-dimensional embeddings
    - block_size=64: Context window of 64 tokens
    """
    
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=64):
        super().__init__()
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small values."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - token indices
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, n_embd)
        
        # Position embeddings
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.position_embedding(pos)  # (1, seq_len, n_embd)
        
        # Combine embeddings
        x = token_emb + pos_emb  # Broadcasting: (batch_size, seq_len, n_embd)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and output projection
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_memory_mb(self):
        """Rough estimate of model memory usage in MB."""
        param_count = self.count_parameters()
        # 4 bytes per float32 parameter
        param_memory = (param_count * 4) / (1024 ** 2)
        return param_memory


def create_mini_transformer(vocab_size, device):
    """Create and initialize MiniTransformer for M1."""
    model = MiniTransformer(
        vocab_size=vocab_size,
        n_embd=128,
        n_head=4,
        n_layer=4,
        block_size=64
    ).to(device)
    
    return model


if __name__ == "__main__":
    # Test Task 2
    print("=" * 60)
    print("TASK 2: The 'Mini-Transformer' Architecture")
    print("=" * 60)
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠ MPS not available, using CPU")
    
    # Create model
    vocab_size = 256  # Typical for character-level tokenization
    model = create_mini_transformer(vocab_size, device)
    
    print(f"\n✓ Model created and moved to {device}")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Estimated memory: {model.estimate_memory_mb():.2f} MB")
    
    # Test forward pass
    print("\n✓ Testing forward pass...")
    batch_size, seq_len = 8, 64
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    logits = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {vocab_size})")
    
    print("\n✓ Task 2 Complete! Ready for Task 3.")
