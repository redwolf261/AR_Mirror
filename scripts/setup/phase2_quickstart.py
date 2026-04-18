#!/usr/bin/env python3
"""
Phase 2 Quick Start
Automated setup and validation for Phase 2
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command with progress indication"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
        return True
    else:
        print(f"❌ {description} - FAILED")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                   PHASE 2 QUICK START                             ║
║          Neural Models + GPU Acceleration Setup                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    steps = []
    
    # Step 1: Check PyTorch
    print("\n[1/5] Checking PyTorch installation...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} already installed")
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} available")  # type: ignore
        else:
            print("⚠️  CUDA not available - will use CPU")
        steps.append(("PyTorch", True))
    except ImportError:
        print("❌ PyTorch not installed")
        print("\nInstalling PyTorch with CUDA 12.1...")
        success = run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "PyTorch Installation"
        )
        steps.append(("PyTorch", success))
        if not success:
            print("\n⚠️  You can continue with CPU-only mode")
    
    # Step 2: Install gdown
    print("\n[2/5] Installing model download utility...")
    success = run_command("pip install gdown", "gdown Installation")
    steps.append(("gdown", success))
    
    # Step 3: Download TOM checkpoint
    print("\n[3/5] Downloading TOM checkpoint...")
    tom_path = Path("cp-vton/checkpoints/tom_train_new/tom_final.pth")
    if tom_path.exists():
        print(f"✅ TOM checkpoint already exists ({tom_path.stat().st_size / 1024 / 1024:.1f} MB)")
        steps.append(("TOM Download", True))
    else:
        success = run_command("python download_tom_checkpoint.py", "TOM Checkpoint Download")
        steps.append(("TOM Download", success))
    
    # Step 4: Validate setup
    print("\n[4/5] Validating Phase 2 setup...")
    success = run_command("python phase2_setup.py", "Setup Validation")
    steps.append(("Validation", success))
    
    # Step 5: Run comprehensive tests
    print("\n[5/5] Running Phase 2 validation tests...")
    success = run_command("python phase2_validation.py", "Comprehensive Testing")
    steps.append(("Testing", success))
    
    # Final report
    print("\n" + "="*70)
    print("QUICK START SUMMARY")
    print("="*70)
    
    for step_name, success in steps:
        status = "✅" if success else "❌"
        print(f"  {status} {step_name}")
    
    total = len(steps)
    passed = sum(1 for _, s in steps if s)
    
    print(f"\nCompleted: {passed}/{total} steps")
    
    if passed == total:
        print("\n🎉 PHASE 2 SETUP COMPLETE!")
        print("\nNext steps:")
        print("  1. Run demo: python app.py --phase2")
        print("  2. Check guide: docs/guides/PHASE_2_COMPLETE_GUIDE.md")
        print("  3. Run benchmarks: python tests/stress_test_pipeline.py")
    elif passed >= total * 0.7:
        print("\n⚠️  SETUP PARTIALLY COMPLETE")
        print("Review failed steps above and retry")
    else:
        print("\n❌ SETUP FAILED")
        print("Please review errors and run: python phase2_setup.py")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
