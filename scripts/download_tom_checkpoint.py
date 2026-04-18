#!/usr/bin/env python3
"""
Download TOM checkpoint from Google Drive
Uses gdown library for automated download
"""

import sys
import os
from pathlib import Path

def download_tom_checkpoint():
    """Download TOM checkpoint using gdown"""
    print("\n" + "="*70)
    print("TOM CHECKPOINT DOWNLOAD")
    print("="*70)
    
    # Install gdown if not present
    try:
        import gdown
    except ImportError:
        print("Installing gdown library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    # TOM checkpoint details
    file_id = "1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy"
    output_dir = Path("cp-vton/checkpoints/tom_train_new")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tom_final.pth"
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ TOM checkpoint already exists: {size_mb:.1f} MB")
        print(f"   Path: {output_path}")
        return True
    
    print(f"📥 Downloading TOM checkpoint...")
    print(f"   File ID: {file_id}")
    print(f"   Output: {output_path}")
    print(f"   Expected size: ~380 MB")
    print("\nThis may take several minutes depending on your connection...")
    
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n✅ Download complete: {size_mb:.1f} MB")
            print(f"   Saved to: {output_path}")
            return True
        else:
            print("\n❌ Download failed - file not found")
            return False
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://drive.google.com/file/d/1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy")
        print(f"2. Download tom_final.pth")
        print(f"3. Save to: {output_path}")
        return False

if __name__ == "__main__":
    success = download_tom_checkpoint()
    sys.exit(0 if success else 1)
