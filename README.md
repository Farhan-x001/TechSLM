# TechSLM: Small Language Model for M1 Mac

A complete implementation of a custom Small Language Model (SLM) optimized for 8GB M1 MacBooks. This project is designed to run efficiently on Apple Silicon with Metal Performance Shaders (MPS) GPU acceleration.

## 📋 Project Overview

This is a 4-task implementation of a tiny transformer-based language model trained on tech news data:

1. **Task 1**: Data Preprocessing & Tokenization
2. **Task 2**: MiniTransformer Architecture  
3. **Task 3**: Training Loop with M1 Optimization
4. **Task 4**: FastAPI Deployment

## 🔧 Requirements

### Hardware
- **Apple M1 Mac** (or Apple Silicon)
- **8GB RAM** minimum (tested and working)
- **macOS 11.0+** with PyTorch MPS support

### Software Dependencies

```bash
pip install torch
pip install numpy
pip install fastapi
pip install uvicorn
pip install pydantic
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Task 1 (Data Preparation)
```bash
python task1_data_preprocessing.py
```

This will:
- Create a `tech_news.txt` file with sample data
- Build a character-level tokenizer (256 unique chars)
- Create PyTorch DataLoader for efficient batch loading

### Step 3: Run Task 2 (Model Architecture)
```bash
python task2_model_architecture.py
```

This will:
- Create the MiniTransformer model (~1M parameters)
- Display architecture summary
- Test a forward pass to ensure everything works

### Step 4: Run Task 3 (Training)
```bash
python task3_training_loop.py
```

This will:
- Load the data and model
- Train for 5 epochs on the M1 GPU
- Save checkpoints every 500 steps
- Save final model to `tech_slm_model.pth`
- Export training metrics to `training_metrics.json`

⏱️ **Expected training time**: ~2-5 minutes on M1 (depending on dataset size)

### Step 5: Run Task 4 (API Deployment)
```bash
python task4_fastapi_deployment.py
```

This will:
- Start a FastAPI server on `http://localhost:8000`
- Load the trained model
- Expose `/generate` endpoint for text generation

## 📡 API Usage

### Check API Health
```bash
curl http://localhost:8000/health
```

### Get Model Info
```bash
curl http://localhost:8000/info
```

### Generate Text
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "This week in AI",
    "max_length": 50,
    "temperature": 0.8,
    "top_k": 40
  }'
```

### Interactive API Documentation
Open in your browser: `http://localhost:8000/docs`

## 🎯 Architecture Details

### MiniTransformer Specs
- **Parameters**: ~1.3M (fits in 8GB RAM)
- **Layers**: 4 transformer blocks
- **Attention Heads**: 4
- **Embedding Dimension**: 128
- **Block Size**: 64 tokens
- **Vocabulary**: 256 characters

### Why This Works on M1

1. **Small Model Size**
   - 1.3M parameters = ~5.2 MB of weights
   - Leaves plenty of RAM for activations and gradients

2. **Metal Performance Shaders (MPS)**
   - PyTorch MPS provides GPU acceleration on Apple Silicon
   - Batch operations run on GPU, freeing CPU RAM
   - Typical 3-5x speedup vs CPU training

3. **Memory-Efficient Design**
   - Character-level tokenization (256 vocab)
   - Small batch sizes (32)
   - Short context window (64 tokens)
   - Gradient clipping prevents memory spikes

4. **Optimized Data Pipeline**
   - PyTorch DataLoader loads batches on-demand
   - Data moved to GPU immediately after loading
   - No pre-loading entire dataset into memory

## 📊 Performance Expectations

On an M1 Mac with 8GB RAM:

| Metric | Expected Value |
|--------|----------------|
| Model Size | ~5 MB |
| Memory Usage | ~2-3 GB during training |
| Training Time (5 epochs) | 2-5 minutes |
| Inference Time | <100ms per generation |
| Max Batch Size | 32 |

## 🔍 Troubleshooting

### "MPS not available"
```python
if torch.backends.mps.is_available():
    # Using GPU
else:
    # Falling back to CPU (slower)
```

**Solution**: Ensure you have PyTorch built with MPS support:
```bash
pip install --upgrade torch
```

### Out of Memory Errors
- Reduce `batch_size` in `TrainingConfig` (try 16 or 8)
- Reduce `block_size` (try 32 instead of 64)
- Reduce `n_layer` or `n_embd` in model config

### Model Generation is Slow
- The first generation is slower (model warm-up)
- Subsequent generations should be <100ms
- Check if model is on MPS device: `print(model.lm_head.weight.device)`

## 📁 File Structure

```
TechSLM/
├── task1_data_preprocessing.py    # Data loading & tokenization
├── task2_model_architecture.py    # MiniTransformer definition
├── task3_training_loop.py         # Training script
├── task4_fastapi_deployment.py    # API server
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── tech_news.txt                  # Generated sample data
├── tech_slm_model.pth             # Trained model weights
├── training_metrics.json          # Training history
└── checkpoint.pth                 # Training checkpoint (if interrupted)
```

## 🎓 Learning Resources

### Key Concepts
1. **Transformers**: https://arxiv.org/abs/1706.03762
2. **NanoGPT**: https://github.com/karpathy/nanoGPT
3. **PyTorch MPS**: https://pytorch.org/docs/stable/notes/mps.html
4. **Character-level LMs**: https://karpathy.github.io/2015/05/21/rnn-effectiveness/

### Customization Ideas
- Replace `tech_news.txt` with your own training data
- Adjust hyperparameters in `TrainingConfig`
- Add custom prompt templates in the API
- Implement fine-tuning on domain-specific data
- Add more advanced generation strategies (beam search, etc.)

## ⚡ Next Steps

Once you have a working model:

1. **Expand Training Data**: Add more tech news articles to `tech_news.txt`
2. **Fine-tune on Domain Data**: Train on specialized tech topics
3. **Add More API Features**: Batch processing, streaming responses
4. **Deploy to Production**: Use Docker + cloud platform
5. **Improve Generation**: Add beam search, temperature sampling
6. **Quantization**: Reduce model size further with 8-bit quantization

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues and bugs
- Suggest improvements
- Submit pull requests
- Share custom training data

## ⚠️ Disclaimer

This is an educational project. For production use:
- Add error handling and logging
- Implement rate limiting
- Add authentication to the API
- Monitor memory and GPU usage
- Test with larger datasets

Enjoy building your own AI on Apple Silicon! 🚀
