#!/usr/bin/env python3
"""
TechSLM: Complete Training Pipeline
===================================

This is the main entry point that orchestrates all 4 tasks.
Run this script to train a complete Small Language Model on your M1 Mac.

Usage:
    python train_slm.py                    # Run all tasks
    python train_slm.py --task 1           # Run only task 1
    python train_slm.py --task 3 --epochs 10  # Train for 10 epochs
"""

import argparse
import torch
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_environment():
    """Check if the environment is ready."""
    print_header("Environment Check")
    
    print("✓ Python version:", end=" ")
    import sys
    print(f"{sys.version.split()[0]}")
    
    print("✓ PyTorch version:", end=" ")
    print(torch.__version__)
    
    print("✓ MPS available:", end=" ")
    print("Yes" if torch.backends.mps.is_available() else "No (CPU mode)")
    
    print("✓ Current device:", end=" ")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Metal GPU")
    else:
        device = torch.device("cpu")
        print("CPU")
    
    return device


def run_task_1():
    """Run Task 1: Data Preprocessing & Tokenization"""
    print_header("Task 1: Data Preprocessing & Tokenization")
    print("Loading and preparing data...")
    
    from task1_data_preprocessing import load_tech_news_data
    
    dataloader, tokenizer, device = load_tech_news_data(
        block_size=64,
        batch_size=32
    )
    
    print("\n✓ Task 1 Complete!")
    print(f"  - Vocabulary size: {tokenizer.vocab_size}")
    print(f"  - Batches: {len(dataloader)}")
    print(f"  - Device: {device}")
    
    return dataloader, tokenizer, device


def run_task_2():
    """Run Task 2: Model Architecture"""
    print_header("Task 2: Model Architecture")
    print("Building MiniTransformer...")
    
    from task2_model_architecture import create_mini_transformer
    
    # Create with vocab size from task 1
    from task1_data_preprocessing import load_tech_news_data
    _, tokenizer, device = load_tech_news_data()
    
    model = create_mini_transformer(tokenizer.vocab_size, device)
    
    print("\n✓ Task 2 Complete!")
    print(f"  - Parameters: {model.count_parameters():,}")
    print(f"  - Memory: {model.estimate_memory_mb():.2f} MB")
    print(f"  - Device: {device}")
    
    return model, tokenizer, device


def run_task_3(max_epochs=5):
    """Run Task 3: Training Loop"""
    print_header("Task 3: Training Loop with M1 Optimization")
    print(f"Training for {max_epochs} epochs...")
    
    from task3_training_loop import train, TrainingConfig
    
    config = TrainingConfig()
    config.max_epochs = max_epochs
    
    model, tokenizer, device = train(config)
    
    print("\n✓ Task 3 Complete!")
    print(f"  - Model saved to: tech_slm_model.pth")
    print(f"  - Metrics saved to: training_metrics.json")
    
    return model, tokenizer, device


def run_task_4():
    """Run Task 4: FastAPI Deployment"""
    print_header("Task 4: FastAPI Deployment")
    print("Starting API server...")
    
    model_path = Path("tech_slm_model.pth")
    if not model_path.exists():
        print("\n✗ Error: tech_slm_model.pth not found!")
        print("  Please run Task 3 first to train the model.")
        return False
    
    print("\n✓ Model file found!")
    print("  Starting FastAPI server on http://localhost:8000")
    print("  Press Ctrl+C to stop the server")
    print("\nDocs available at:")
    print("  - http://localhost:8000/docs")
    print("  - http://localhost:8000/redoc")
    
    try:
        from task4_fastapi_deployment import app
        import uvicorn
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")
        return True


def run_all_tasks(max_epochs=5):
    """Run all tasks in sequence."""
    print_header("TechSLM: Complete Training Pipeline")
    print("This will train a Small Language Model on your M1 Mac")
    print(f"Training epochs: {max_epochs}")
    
    # Check environment
    device = check_environment()
    
    # Run tasks
    try:
        print_header("Running all tasks...")
        
        # Task 1
        dataloader, tokenizer, device = run_task_1()
        input("Press Enter to continue to Task 2...")
        
        # Task 2
        model, tokenizer, device = run_task_2()
        input("Press Enter to continue to Task 3...")
        
        # Task 3
        model, tokenizer, device = run_task_3(max_epochs)
        input("Press Enter to continue to Task 4 (API)...")
        
        # Task 4
        run_task_4()
        
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="TechSLM: Small Language Model for M1 Mac",
        epilog="Example: python train_slm.py --task 3 --epochs 10"
    )
    
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific task (1-4). If not specified, run all tasks."
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check environment and exit"
    )
    
    args = parser.parse_args()
    
    # Environment check
    if args.check:
        check_environment()
        return
    
    # Run specific task or all
    try:
        if args.task == 1:
            run_task_1()
        elif args.task == 2:
            run_task_2()
        elif args.task == 3:
            run_task_3(args.epochs)
        elif args.task == 4:
            run_task_4()
        else:
            # Run all tasks
            run_all_tasks(args.epochs)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
