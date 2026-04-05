"""
Task 4: Zero-Cost Local Deployment (FastAPI)
==============================================
Why this works on M1:
- FastAPI is lightweight and runs efficiently on M1
- No cloud costs - runs locally on your machine
- MPS device handling ensures GPU acceleration during inference
- Batch processing of tokens allows for efficient text generation
- Simple POST endpoint makes it easy to integrate with other applications
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
from pathlib import Path

# Import from other tasks
from task3_training_loop import load_model


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_length: int = 5000  # Generate up to 20,000 words
    temperature: float = 1.0
    top_k: int = 50


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    prompt: str
    generated_text: str
    tokens_generated: int


# ============================================================================
# Text Generation
# ============================================================================

def generate_text(model, tokenizer, prompt, max_length=5000, temperature=1.0, top_k=50, device=None):
    """
    Generate text from a prompt using the trained model.
    
    Args:
        model: MiniTransformer model
        tokenizer: CharTokenizer instance
        prompt: Starting text
        max_length: Maximum length of generated text (up to 20,000 tokens)
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top-k most likely tokens
        device: torch device
    
    Returns:
        Generated text string
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Generate tokens one by one
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions for the last token
            logits = model(tokens)  # (1, seq_len, vocab_size)
            logits = logits[0, -1, :] / temperature  # (vocab_size,)
            
            # Top-k sampling
            if top_k > 0:
                # Get top-k logits
                top_logits, top_indices = torch.topk(logits, min(top_k, logits.shape[0]))
                
                # Zero out non-top-k logits
                logits_new = torch.full_like(logits, float('-inf'))
                logits_new[top_indices] = top_logits
                logits = logits_new
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence (keep only last block_size tokens to avoid context explosion)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Keep only the last 64 tokens (model's context window)
            if tokens.shape[1] > 64:
                tokens = tokens[:, -64:]
    
    # Decode tokens
    generated_tokens = tokens[0].cpu().numpy().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="TechSLM API",
    description="Tech Trends Small Language Model API",
    version="1.0"
)

# Global variables to store model and tokenizer
_model = None
_tokenizer = None
_device = None


def load_resources():
    """Load model and tokenizer on startup."""
    global _model, _tokenizer, _device
    
    model_path = "tech_slm_model.pth"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Please run task3_training_loop.py first to train the model."
        )
    
    print(f"✓ Loading model from {model_path}...")
    _model, _tokenizer, _device = load_model(model_path)
    print(f"✓ Model loaded on device: {_device}")


@app.on_event("startup")
async def startup_event():
    """Run on API startup."""
    try:
        load_resources()
        print("✓ TechSLM API ready!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(_device),
        "model_loaded": _model is not None
    }


@app.get("/info")
async def model_info():
    """Get model information."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model": "MiniTransformer (NanoGPT-style)",
        "parameters": _model.count_parameters(),
        "memory_mb": _model.estimate_memory_mb(),
        "vocab_size": _tokenizer.vocab_size,
        "device": str(_device),
        "architecture": {
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "block_size": 64
        }
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt (up to 20,000 words).
    
    Example:
    ```
    POST /generate
    {
        "prompt": "This week in AI",
        "max_length": 5000,
        "temperature": 0.8,
        "top_k": 40
    }
    ```
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if request.max_length < 200 or request.max_length > 20000:
        raise HTTPException(status_code=400, detail="max_length must be between 200 and 20,000 words")
    
    if request.temperature <= 0:
        raise HTTPException(status_code=400, detail="temperature must be positive")
    
    try:
        # Generate text
        generated = generate_text(
            _model,
            _tokenizer,
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            device=_device
        )
        
        # Count tokens generated (rough estimate)
        tokens_generated = len(_tokenizer.encode(generated)) - len(_tokenizer.encode(request.prompt))
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated,
            tokens_generated=max(tokens_generated, 0)
        )
    
    except Exception as e:
        print(f"✗ Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/batch-generate")
async def batch_generate(requests: list[GenerateRequest]):
    """
    Generate text for multiple prompts (batch).
    
    Example:
    ```
    POST /batch-generate
    [
        {"prompt": "Apple announces", "max_length": 30},
        {"prompt": "AI breakthrough", "max_length": 30}
    ]
    ```
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for request in requests:
        try:
            generated = generate_text(
                _model,
                _tokenizer,
                request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                device=_device
            )
            
            results.append({
                "prompt": request.prompt,
                "generated_text": generated,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "prompt": request.prompt,
                "error": str(e),
                "status": "failed"
            })
    
    return {"results": results}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("TASK 4: Zero-Cost Local Deployment (FastAPI)")
    print("=" * 60)
    print("\n✓ Starting TechSLM API server...")
    print("  API will be available at: http://localhost:8000")
    print("  Docs at: http://localhost:8000/docs")
    print("  ReDoc at: http://localhost:8000/redoc")
    print("\nExample request (generates up to 20,000 words):")
    print("""
    curl -X POST "http://localhost:8000/generate" \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "This week in AI", "max_length": 5000}'
    """)
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
