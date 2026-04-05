#!/usr/bin/env python3
"""
Improved FastAPI deployment for TechSLM
- Works with word-level tokenization
- Generates actual words instead of characters
- Better sampling strategies (top-k, temperature)
"""

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import re
import json
from pathlib import Path
import uvicorn


# ============================================================================
# WORD-LEVEL TOKENIZER (matches training)
# ============================================================================

class WordTokenizer:
    """Simple word-level tokenizer with special tokens."""
    
    def __init__(self, vocab):
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
    
    def encode(self, text):
        """Convert text to token indices."""
        words = re.findall(r'\b\w+\b|[.,!?;:]', text.lower())
        return [self.word_to_idx.get(w, 1) for w in words]  # 1 is <unk>
    
    def decode(self, indices):
        """Convert token indices back to text."""
        words = [self.idx_to_word.get(i, '<unk>') for i in indices]
        # Clean up spacing around punctuation
        text = ' '.join(words)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text


# ============================================================================
# MODEL ARCHITECTURE (matches training)
# ============================================================================

class ImprovedMiniTransformer(nn.Module):
    """Enhanced transformer with larger capacity."""
    
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(n_layer)
        ])
        
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        for block in self.transformer_blocks:
            x = block(x, src_key_padding_mask=None, is_causal=True, src_mask=causal_mask)
        
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits


# ============================================================================
# API MODELS
# ============================================================================

class GenerateRequest(BaseModel):
    """Request to generate text."""
    prompt: str = Field(..., description="Initial text prompt")
    max_tokens: int = Field(100, ge=10, le=2000, description="Maximum tokens to generate (words)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=0, le=100, description="Top-k sampling parameter")


class GenerateResponse(BaseModel):
    """Response with generated text."""
    prompt: str
    generated_text: str
    num_tokens: int
    temperature: float
    top_k: int


# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_text(
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    temperature=0.7,
    top_k=50,
    device='cpu'
):
    """Generate text using the improved model."""
    
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).to(device)
    
    generated = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Prepare input - ensure it's a batch (B, T)
            if tokens.dim() == 1:
                input_tokens = tokens.unsqueeze(0)  # (1, T)
            else:
                input_tokens = tokens
            
            # Keep only last 256 tokens (block size)
            if input_tokens.shape[1] > 256:
                input_tokens = input_tokens[:, -256:]
            
            # Forward pass
            logits = model(input_tokens)  # (B, T, V)
            next_logits = logits[:, -1, :] / temperature  # (B, V)
            
            # Top-k sampling
            if top_k > 0:
                top_values, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                threshold = top_values[..., -1:]
                next_logits[next_logits < threshold] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()  # scalar
            
            if next_token.item() == 0:  # <pad> token - stop
                break
            
            generated.append(next_token.item())
            tokens = torch.cat([tokens, next_token.unsqueeze(0)])
    
    # Decode
    full_tokens = tokenizer.encode(prompt) + generated
    generated_text = tokenizer.decode(full_tokens)
    
    return generated_text, len(generated)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Improved TechSLM API",
    description="Small Language Model with word-level generation",
    version="2.0"
)

# Global variables for model and tokenizer
_model = None
_tokenizer = None
_device = None


def load_model():
    """Load the trained model and tokenizer."""
    global _model, _tokenizer, _device
    
    _device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load('tech_slm_model.pth', map_location=_device)
    
    vocab = checkpoint['tokenizer_vocab']
    _tokenizer = WordTokenizer(vocab)
    
    config = checkpoint['config']
    _model = ImprovedMiniTransformer(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size']
    ).to(_device)
    
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()
    
    print(f"✓ Model loaded on {_device}")
    print(f"✓ Vocabulary size: {_tokenizer.vocab_size}")


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    load_model()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": str(_device)
    }


@app.get("/info")
async def info():
    """Get model information."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "Improved TechSLM",
        "version": "2.0",
        "tokenization": "word-level",
        "vocab_size": _tokenizer.vocab_size,
        "architecture": {
            "n_embd": 256,
            "n_head": 8,
            "n_layer": 6,
            "block_size": 256
        },
        "device": str(_device)
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text based on a prompt."""
    
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated_text, num_tokens = generate_text(
            model=_model,
            tokenizer=_tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            device=_device
        )
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            num_tokens=num_tokens,
            temperature=request.temperature,
            top_k=request.top_k
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-generate")
async def batch_generate(requests: list[GenerateRequest]):
    """Generate text for multiple prompts."""
    
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for req in requests:
        try:
            generated_text, num_tokens = generate_text(
                model=_model,
                tokenizer=_tokenizer,
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                device=_device
            )
            
            results.append({
                "prompt": req.prompt,
                "generated_text": generated_text,
                "num_tokens": num_tokens,
                "status": "success"
            })
        
        except Exception as e:
            results.append({
                "prompt": req.prompt,
                "error": str(e),
                "status": "failed"
            })
    
    return {"results": results}


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting Improved TechSLM API on http://localhost:8000")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
