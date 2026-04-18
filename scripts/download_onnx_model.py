#!/usr/bin/env python3
"""
Download pre-converted ONNX human parsing model from Hugging Face
Much simpler than converting from PyTorch!
"""

import os
import sys
import urllib.request
from pathlib import Path

print("=" * 70)
print("DOWNLOADING ONNX HUMAN PARSING MODEL")
print("=" * 70)

# Configuration - using pre-converted ONNX model from Hugging Face
# Correct path: humanparsing folder, not onnx folder
MODEL_URL = "https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx"
MODEL_PATH = "models/schp_lip.onnx"

def download_with_progress(url: str, output_path: str):
    """Download file with progress bar"""
    print(f"\n[1/2] Downloading ONNX model...")
    print(f"  Source: Hugging Face (yisol/IDM-VTON)")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")
    print(f"  Expected size: ~267 MB")
    
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n  [SKIP] Model already exists ({file_size_mb:.2f} MB)")
        print(f"  Delete {output_path} to re-download")
        return True
    
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                mb_downloaded = (count * block_size) / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
            else:
                mb_downloaded = (count * block_size) / (1024 * 1024)
                sys.stdout.write(f"\r  Downloaded: {mb_downloaded:.1f} MB")
                sys.stdout.flush()
        
        print("\n  Starting download (this may take a few minutes)...")
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print(f"\n  [OK] Download complete!")
        
        # Verify file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        
        if file_size_mb < 200:
            print(f"  [WARN] File seems too small, might be incomplete")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n  [FAIL] Download failed: {e}")
        print(f"\n  Manual download instructions:")
        print(f"  1. Visit: https://huggingface.co/yisol/IDM-VTON/tree/main/onnx")
        print(f"  2. Download: parsing_lip.onnx")
        print(f"  3. Place in: {output_path}")
        return False


def test_onnx_model(model_path: str):
    """Test the downloaded ONNX model"""
    print(f"\n[2/2] Testing ONNX model...")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Create session
        print(f"  Loading model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        active_provider = session.get_providers()[0]
        print(f"  [OK] Model loaded successfully")
        print(f"  Active provider: {active_provider}")
        
        # Get model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"\n  Model details:")
        print(f"    Input name: {input_info.name}")
        print(f"    Input shape: {input_info.shape}")
        print(f"    Output name: {output_info.name}")
        print(f"    Output shape: {output_info.shape}")
        
        # Test inference
        print(f"\n  Running test inference...")
        # Typical input for parsing models: (1, 3, H, W)
        test_input = np.random.randn(1, 3, 512, 384).astype(np.float32)
        outputs = session.run(None, {input_info.name: test_input})
        
        print(f"  [OK] Inference successful!")
        print(f"  Output shape: {outputs[0].shape}")  # pyright: ignore
        
        # Check if output has multiple classes (parsing map)
        if len(outputs[0].shape) == 4:  # pyright: ignore  # (batch, classes, H, W)
            num_classes = outputs[0].shape[1]  # pyright: ignore
            print(f"  Number of classes: {num_classes}")
            print(f"  This is a multi-class parsing model ✓")
        
        return True
        
    except ImportError:
        print(f"  [WARN] onnxruntime not installed")
        print(f"  Install with: pip install onnxruntime-gpu")
        return False
    except Exception as e:
        print(f"  [FAIL] Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main download and test pipeline"""
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Download model
    success = download_with_progress(MODEL_URL, MODEL_PATH)
    
    if not success:
        print("\n" + "=" * 70)
        print("[FAILED] Could not download model")
        print("=" * 70)
        return False
    
    # Test model
    test_success = test_onnx_model(MODEL_PATH)
    
    if test_success:
        print("\n" + "=" * 70)
        print("[SUCCESS] ONNX model ready for use!")
        print("=" * 70)
        print(f"\nModel location: {MODEL_PATH}")
        print(f"\nNext steps:")
        print(f"  1. Run: python test_backend_abstraction.py")
        print(f"  2. System should auto-select ONNXParsingBackend")
        print(f"  3. Run: python app.py --phase 2")
        print(f"  4. Verify improved occlusion handling")
    else:
        print("\n" + "=" * 70)
        print("[PARTIAL SUCCESS] Model downloaded but not tested")
        print("=" * 70)
        print(f"Model is at: {MODEL_PATH}")
        print(f"Install onnxruntime-gpu to test: pip install onnxruntime-gpu")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
