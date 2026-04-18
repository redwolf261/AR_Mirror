#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Detection & Model Status Verification Script
Checks GPU availability and validates downloaded model checkpoints
"""

import os
import sys
from pathlib import Path

def check_gpu_status():
    """Check GPU availability and print status"""
    print("\n" + "="*70)
    print("GPU DETECTION & STATUS")
    print("="*70)
    
    try:
        from src.hybrid.gpu_acceleration import GPUAccelerator
        
        accelerator = GPUAccelerator()
        
        # Get available status
        is_available = accelerator.is_available()
        device_name = accelerator.device_name
        device_type = accelerator.device_type
        
        print(f"\n[OK] GPU Detection Module: LOADED")
        print(f"  Device Type: {device_type}")
        print(f"  Available: {is_available}")
        print(f"  Device Name: {device_name}")
        
        return is_available
    
    except Exception as e:
        print(f"\n[ERROR] GPU Detection failed: {e}")
        return False

def check_model_files():
    """Check if model checkpoint files exist"""
    print("\n" + "="*70)
    print("MODEL CHECKPOINT FILES")
    print("="*70)
    
    models_dir = Path("models")
    models_to_check = {
        "gmm": ("hr_viton_gmm.pth", 120),
        "tom": ("hr_viton_tom.pth", 380),
    }
    
    results = {}
    for name, (filename, expected_size_mb) in models_to_check.items():
        filepath = models_dir / filename
        
        if filepath.exists():
            actual_size_mb = filepath.stat().st_size / (1024**2)
            size_match = abs(actual_size_mb - expected_size_mb) / expected_size_mb < 0.1
            
            status = "[OK]" if size_match else "[WARN]"
            print(f"\n{status} {name.upper()}: {filename}")
            print(f"    Path: {filepath}")
            print(f"    Size: {actual_size_mb:.2f} MB (Expected: ~{expected_size_mb} MB)")
            print(f"    Match: {'YES' if size_match else 'NO - POSSIBLE CORRUPTION'}")
            
            results[name] = {"exists": True, "valid_size": size_match}
        else:
            print(f"\n[MISSING] {name.upper()}: MISSING - {filename}")
            print(f"    Expected at: {filepath}")
            print(f"    Download from: PHASE_2_MODEL_DOWNLOAD_GUIDE.md")
            
            results[name] = {"exists": False, "valid_size": False}
    
    return results

def check_model_loading():
    """Attempt to load models and verify integrity"""
    print("\n" + "="*70)
    print("MODEL LOADING & VALIDATION")
    print("="*70)
    
    try:
        from src.hybrid.neural_models import NeuralModelManager
        
        manager = NeuralModelManager()
        status = manager.get_model_status()
        
        print("\nModel Status Report:")
        for model_name, model_info in status.items():
            print(f"\n  {model_name.upper()}:")
            
            # Print all status details
            for key, value in model_info.items():
                if key == "status":
                    symbol = "[OK]" if value == "[READY]" else "✗" if value == "[MISSING]" else "⚠"
                    print(f"    {symbol} Status: {value}")
                elif key == "error":
                    print(f"    ✗ Error: {value}")
                else:
                    print(f"      {key}: {value}")
        
        return status
    
    except Exception as e:
        print(f"\n✗ Model Loading Error: {e}")
        print(f"   This may be expected if models aren't downloaded yet")
        return None

def print_next_steps():
    """Print next steps based on status"""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    print("""
1. IMMEDIATE (0-30 min):
   □ Download GMM checkpoint (120 MB)
   □ Download TOM checkpoint (380 MB)
   □ Save to models/ directory
   □ Re-run this script to verify

2. THIS HOUR (30-60 min):
   □ Load models into GPU
   □ Run benchmarking tests
   □ Verify performance

3. THIS SESSION (1-3 hours):
   □ Integrate into hybrid pipeline
   □ Run stress tests with neural
   □ Validate quality improvement

See PHASE_2_MODEL_DOWNLOAD_GUIDE.md for detailed instructions.
""")

def main():
    """Run all checks"""
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("\n" + "="*70)
    print(" PHASE 2 - GPU & MODEL VERIFICATION SCRIPT".center(70))
    print("="*70)
    
    # Check GPU
    gpu_available = check_gpu_status()
    
    # Check files
    files_status = check_model_files()
    
    # Check loading
    models_status = check_model_loading()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n[OK] GPU Available: {gpu_available}")
    
    gmm_ready = files_status.get("gmm", {}).get("valid_size", False)
    tom_ready = files_status.get("tom", {}).get("valid_size", False)
    
    print(f"[{'OK' if gmm_ready else 'MISSING'}] GMM Ready: {gmm_ready}")
    print(f"[{'OK' if tom_ready else 'MISSING'}] TOM Ready: {tom_ready}")
    
    phase2_ready = gpu_available and gmm_ready and tom_ready
    status_icon = "[READY]" if phase2_ready else "[PENDING]"
    print(f"\n{status_icon} PHASE 2A STATUS: {phase2_ready}")
    
    print_next_steps()
    
    print("="*70 + "\n")
    
    return 0 if phase2_ready else 1

if __name__ == "__main__":
    sys.exit(main())
