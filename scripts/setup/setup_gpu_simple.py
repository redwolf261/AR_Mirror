"""
Simple GPU Setup for AR Mirror
Installs CUDA-enabled PyTorch and verifies GPU
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("AR MIRROR - GPU ACCELERATION SETUP")
    print("="*70)
    
    # Step 1: Uninstall old PyTorch
    print("\n[1/3] Uninstalling old PyTorch (if present)...")
    run_command(
        f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio",
        "Uninstall old PyTorch"
    )
    
    # Step 2: Install CUDA PyTorch
    print("\n[2/3] Installing PyTorch with CUDA 12.1...")
    print("This will download ~2GB, please wait...")
    
    success = run_command(
        f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "Install PyTorch with CUDA"
    )
    
    if not success:
        print("\n❌ PyTorch installation failed!")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Step 3: Verify
    print("\n[3/3] Verifying GPU setup...")
    try:
        import torch
        
        print("\n" + "="*70)
        print("GPU VERIFICATION")
        print("="*70)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")  # type: ignore
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            
            props = torch.cuda.get_device_properties(0)
            print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
            print("="*70)
            print("✅ GPU READY FOR AR MIRROR!")
            print("="*70)
            
            print("\nNext steps:")
            print("  1. Run: python scripts\\verify_gpu.py")
            print("  2. Run: python app.py --phase 2")
            print("  3. Expected FPS: 200-300 (vs current 21)")
            
        else:
            print("="*70)
            print("⚠ WARNING: CUDA NOT AVAILABLE")
            print("="*70)
            print("\nPossible issues:")
            print("  1. NVIDIA drivers not installed")
            print("  2. GPU disabled in BIOS")
            print("  3. Restart may be needed")
            print("\nThe system will fallback to CPU (21 FPS)")
    
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)
    
    print("\n")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
