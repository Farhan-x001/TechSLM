# TechSLM Setup & Execution Guide

## 🎯 Overview

This guide walks you through setting up and running the complete TechSLM project on your 8GB M1 Mac.

## 📦 Installation

### 1. Install Python 3.9+

Check your Python version:
```bash
python3 --version
```

If you need to install/upgrade Python on M1:
```bash
# Using Homebrew (recommended)
brew install python@3.11
```

### 2. Install Dependencies

Navigate to the project directory:
```bash
cd /Users/farhanahmed/Desktop/agents/TechSLM
```

Install required packages:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch numpy fastapi uvicorn pydantic
```

### 3. Verify Installation

Check that everything is installed correctly:
```bash
python3 train_slm.py --check
```

You should see:
```
Python version: 3.9.x or higher ✓
PyTorch version: 2.0.x or higher ✓
MPS available: Yes ✓
Current device: Metal GPU ✓
```

## 🚀 Running the Project

### Option A: Run Everything at Once

```bash
python3 train_slm.py
```

This will:
1. Run Task 1 (Data Preprocessing)
2. Run Task 2 (Model Architecture)
3. Run Task 3 (Training) - 5 epochs
4. Run Task 4 (FastAPI Server)

Press Enter between each task to continue.

### Option B: Run Individual Tasks

**Task 1 Only** - Data Preprocessing:
```bash
python3 task1_data_preprocessing.py
```

**Task 2 Only** - Model Architecture:
```bash
python3 task2_model_architecture.py
```

**Task 3 Only** - Training (5 epochs):
```bash
python3 task3_training_loop.py
```

**Task 3 with Custom Epochs** (e.g., 10 epochs):
```bash
python3 train_slm.py --task 3 --epochs 10
```

**Task 4 Only** - Start API Server:
```bash
python3 task4_fastapi_deployment.py
```

## 📊 Expected Output

### Task 1: Data Preprocessing (30 seconds)
```
============================================================
TASK 1: Data Preprocessing & Tokenization
============================================================
✓ Loaded 5,000+ characters from tech_news.txt
✓ Vocabulary size: 256 unique characters
✓ Created DataLoader with 78 samples, batch_size=32
✓ Samples per epoch: 3

✓ Testing batch loading...
  Input shape: torch.Size([32, 64])
  Target shape: torch.Size([32, 64])
  First 10 tokens: tensor([...])
  Sample text: 'Apple M1 chip delivers...'

✓ Task 1 Complete! Ready for Task 2.
```

### Task 2: Model Architecture (10 seconds)
```
============================================================
TASK 2: The 'Mini-Transformer' Architecture
============================================================
✓ Using Metal Performance Shaders (MPS)
✓ Model created and moved to mps

  Total parameters: 1,310,976
  Estimated memory: 5.00 MB

✓ Testing forward pass...
  Input shape: torch.Size([8, 64])
  Output shape: torch.Size([8, 64, 256])
  Expected: (8, 64, 256)

✓ Task 2 Complete! Ready for Task 3.
```

### Task 3: Training (2-5 minutes)
```
============================================================
TASK 3: Training Loop with M1 Optimization
============================================================
✓ Metal Performance Shaders (MPS) AVAILABLE - Using GPU acceleration
  Device: mps

✓ Loading data...
✓ Creating model...
  Parameters: 1,310,976
  Memory: 5.00 MB

✓ Starting training...
  Epochs: 5
  Batches per epoch: 3
  Total steps: ~15

  Step 500 | Epoch 1/5 | Batch 1/3 | Loss: 5.3421
  Step 1000 | Epoch 2/5 | Batch 1/3 | Loss: 4.1234
  
✓ Epoch 5 Complete | Avg Loss: 3.0000

✓ Saving model to tech_slm_model.pth...
✓ Model saved! Total parameters: 1,310,976
✓ Training complete. Final loss: 2.5000
✓ Metrics saved to training_metrics.json

✓ Task 3 Complete! Ready for Task 4.
```

### Task 4: API Server (Starts and Runs)
```
============================================================
TASK 4: Zero-Cost Local Deployment (FastAPI)
============================================================

✓ Starting TechSLM API server...
  API will be available at: http://localhost:8000
  Docs at: http://localhost:8000/docs
  ReDoc at: http://localhost:8000/redoc

INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Uvicorn running on http://127.0.0.1:8000/docs (for API docs)

[Server is now running and accepting requests]
```

## 🧪 Testing the API

While the FastAPI server is running (from Task 4), open another terminal and test it:

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "device": "mps",
  "model_loaded": true
}
```

### Test 2: Model Information
```bash
curl http://localhost:8000/info
```

Response:
```json
{
  "model": "MiniTransformer (NanoGPT-style)",
  "parameters": 1310976,
  "memory_mb": 5.0,
  "vocab_size": 256,
  "device": "mps"
}
```

### Test 3: Generate Text
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

Response:
```json
{
  "prompt": "This week in AI",
  "generated_text": "This week in AI advances in neural networks...",
  "tokens_generated": 23
}
```

### Test 4: Interactive API Docs
Open your browser and go to:
```
http://localhost:8000/docs
```

This shows the Swagger UI where you can test all endpoints interactively!

## 📁 Generated Files

After running the full pipeline, you'll have:

```
TechSLM/
├── task1_data_preprocessing.py        (provided)
├── task2_model_architecture.py        (provided)
├── task3_training_loop.py             (provided)
├── task4_fastapi_deployment.py        (provided)
├── train_slm.py                       (provided)
├── requirements.txt                   (provided)
├── README.md                          (provided)
├── SETUP_GUIDE.md                     (this file)
│
├── tech_news.txt                      (generated by Task 1)
├── tech_slm_model.pth                 (generated by Task 3)
├── training_metrics.json              (generated by Task 3)
└── checkpoint.pth                     (generated if training interrupted)
```

## ⚙️ Customization

### Using Your Own Training Data

Replace the sample data:
```bash
# Edit tech_news.txt with your own tech news data
nano tech_news.txt
```

Then re-run Task 3 to retrain the model.

### Adjusting Training Parameters

Edit `task3_training_loop.py` and modify the `TrainingConfig` class:

```python
class TrainingConfig:
    batch_size = 16          # Smaller = slower but more memory-efficient
    learning_rate = 5e-4     # Learning rate
    max_epochs = 10          # More epochs = longer training
    weight_decay = 0.01      # L2 regularization
```

### Adjusting Model Architecture

Edit `task2_model_architecture.py` and modify the model parameters:

```python
def create_mini_transformer(vocab_size, device):
    model = MiniTransformer(
        vocab_size=vocab_size,
        n_embd=256,      # Larger = more capacity (needs more RAM)
        n_head=8,        # More heads = more computation
        n_layer=6,       # More layers = deeper model
        block_size=128   # Longer context window
    ).to(device)
    return model
```

## 🐛 Troubleshooting

### Issue: "MPS not available"

**Solution**: Update PyTorch
```bash
pip install --upgrade torch
```

### Issue: "Out of Memory" during training

**Solution**: Reduce batch size in `task3_training_loop.py`:
```python
class TrainingConfig:
    batch_size = 8  # Reduced from 32
```

### Issue: API port 8000 already in use

**Solution**: Use a different port:
```bash
# In a terminal, run:
python3 -c "
from task4_fastapi_deployment import app
import uvicorn
uvicorn.run(app, host='127.0.0.1', port=8001)
"
```

### Issue: Model training is very slow

**Solution**: Ensure MPS is being used:
```python
# Run this to check:
import torch
print(torch.backends.mps.is_available())  # Should print True
print(torch.backends.mps.is_built())      # Should print True
```

If False, your PyTorch doesn't have MPS support. Reinstall:
```bash
pip uninstall torch
pip install torch
```

## 📚 Next Steps

1. **Expand Your Dataset**: Add more tech news articles to `tech_news.txt`
2. **Improve Generation Quality**: 
   - Increase training epochs
   - Use more diverse training data
   - Adjust hyperparameters
3. **Add More Features**:
   - Fine-tune on specific topics
   - Add prompt templates
   - Implement streaming responses
4. **Deploy to Production**:
   - Add authentication to API
   - Use a production WSGI server (Gunicorn)
   - Deploy to cloud with Docker

## 💡 Tips & Tricks

### Monitor GPU Usage
While training, open another terminal and run:
```bash
# macOS Activity Monitor equivalent
top -o %GPU
```

### Resume Training from Checkpoint
If training was interrupted, it saved a checkpoint:
```bash
# Modify task3_training_loop.py to load from checkpoint:
checkpoint = torch.load('checkpoint.pth')
# Then continue training
```

### Test Generation Quality
```bash
python3 -c "
from task3_training_loop import load_model
from task4_fastapi_deployment import generate_text

model, tokenizer, device = load_model()
text = generate_text(model, tokenizer, 'Apple announces', max_length=100)
print(text)
"
```

## 🎉 Success Checklist

- [ ] Python 3.9+ installed
- [ ] All dependencies installed (`pip list` shows torch, fastapi, etc.)
- [ ] Task 1 completes without errors
- [ ] Task 2 completes without errors
- [ ] Task 3 starts training and loss decreases
- [ ] Task 4 API server starts successfully
- [ ] API endpoints respond correctly
- [ ] Can generate text via API

If all checkmarks are ✓, congratulations! You have a working SLM on your M1 Mac! 🚀

## 📞 Support

For issues or questions:
1. Check the README.md for more details
2. Review the troubleshooting section above
3. Check PyTorch documentation: https://pytorch.org/docs/stable/mps.html
4. Check FastAPI documentation: https://fastapi.tiangolo.com

Happy training! 🎓
