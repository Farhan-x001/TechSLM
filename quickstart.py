#!/usr/bin/env python3
"""
Quick Start: Train and Deploy TechSLM in 5 Minutes
===================================================

This is the fastest way to get started. Just run:
    python3 quickstart.py

It will handle all 4 tasks with minimal user interaction.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║         TechSLM: Quick Start on M1 Mac (5 minutes)        ║
╚════════════════════════════════════════════════════════════╝

This script will:
  1. Install dependencies
  2. Run Task 1: Data Preprocessing
  3. Run Task 2: Model Architecture
  4. Run Task 3: Training (1 epoch for speed)
  5. Start Task 4: API Server

Let's begin!
""")
    
    input("Press Enter to continue...")
    
    # Step 1: Check Python
    print("\n✓ Step 1: Checking Python...")
    result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
    print(f"  {result.stdout.strip()}")
    
    # Step 2: Install dependencies
    print("\n✓ Step 2: Installing dependencies...")
    if not Path("requirements.txt").exists():
        print("  ✗ requirements.txt not found!")
        return False
    
    success = run_command(
        f"{sys.executable} -m pip install -q -r requirements.txt",
        "Installing PyTorch, FastAPI, and other dependencies"
    )
    if not success:
        print("  ✗ Failed to install dependencies")
        return False
    print("  ✓ Dependencies installed!")
    
    # Step 3: Run Task 1
    success = run_command(
        f"{sys.executable} task1_data_preprocessing.py",
        "Task 1: Data Preprocessing & Tokenization"
    )
    if not success:
        print("  ✗ Task 1 failed")
        return False
    
    # Step 4: Run Task 2
    success = run_command(
        f"{sys.executable} task2_model_architecture.py",
        "Task 2: Mini-Transformer Architecture"
    )
    if not success:
        print("  ✗ Task 2 failed")
        return False
    
    # Step 5: Run Task 3 (1 epoch for quick demo)
    print(f"\n{'='*60}")
    print(f"  Task 3: Training (1 epoch - ~30 seconds)")
    print(f"{'='*60}")
    
    from task3_training_loop import train, TrainingConfig
    config = TrainingConfig()
    config.max_epochs = 1
    try:
        train(config)
    except Exception as e:
        print(f"  ✗ Task 3 failed: {e}")
        return False
    
    # Step 6: Confirm Task 4
    print(f"\n{'='*60}")
    print(f"  Ready for Task 4: API Deployment")
    print(f"{'='*60}")
    
    print("""
✓ Training complete! Your model is saved as: tech_slm_model.pth

Next steps:
  1. To start the API server, run:
     python3 task4_fastapi_deployment.py
  
  2. Then open in your browser:
     http://localhost:8000/docs
  
  3. Try generating text with the /generate endpoint!

For more details, see README.md and SETUP_GUIDE.md
""")
    
    response = input("Start API server now? (y/n): ").lower()
    if response == 'y':
        run_command(
            f"{sys.executable} task4_fastapi_deployment.py",
            "Task 4: Starting FastAPI Server"
        )
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n✗ Quickstart cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
