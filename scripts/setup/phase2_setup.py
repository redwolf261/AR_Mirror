#!/usr/bin/env python3
"""
Phase 2 Setup Script - Neural Models + GPU Acceleration
Checks dependencies, downloads models, and validates setup
"""

import sys
import subprocess
from pathlib import Path
import urllib.request
import json

class Phase2Setup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.checkpoints_dir = self.project_root / "cp-vton" / "checkpoints"
        self.models_dir = self.project_root / "models"
        self.results = {}
        
    def check_pytorch(self):
        """Check PyTorch installation and CUDA support"""
        print("\n" + "="*70)
        print("STEP 1: PyTorch & CUDA Check")
        print("="*70)
        
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__} installed")
            
            if torch.cuda.is_available():
                print(f"✅ CUDA {torch.version.cuda} available")  # type: ignore
                print(f"   Device: {torch.cuda.get_device_name(0)}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                self.results['pytorch'] = 'cuda'
            else:
                print("⚠️  CUDA not available - will use CPU mode")
                print("   (Install CUDA toolkit for GPU acceleration)")
                self.results['pytorch'] = 'cpu'
            return True
            
        except ImportError:
            print("❌ PyTorch not installed")
            print("\nInstall with:")
            print("  pip install torch torchvision")
            print("  # For CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            self.results['pytorch'] = 'missing'
            return False
    
    def check_gmm_checkpoint(self):
        """Check if GMM checkpoint exists"""
        print("\n" + "="*70)
        print("STEP 2: GMM Checkpoint Check")
        print("="*70)
        
        gmm_path = self.checkpoints_dir / "gmm_train_new" / "gmm_final.pth"
        
        if gmm_path.exists():
            size_mb = gmm_path.stat().st_size / (1024 * 1024)
            print(f"✅ GMM checkpoint found: {size_mb:.1f} MB")
            print(f"   Path: {gmm_path}")
            self.results['gmm'] = 'present'
            return True
        else:
            print("❌ GMM checkpoint not found")
            print(f"   Expected at: {gmm_path}")
            print("\nDownload from:")
            print("   https://drive.google.com/file/d/1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ")
            self.results['gmm'] = 'missing'
            return False
    
    def check_tom_checkpoint(self):
        """Check if TOM checkpoint exists"""
        print("\n" + "="*70)
        print("STEP 3: TOM Checkpoint Check")
        print("="*70)
        
        # Check multiple possible locations
        tom_paths = [
            self.checkpoints_dir / "tom_train_new" / "tom_final.pth",
            self.checkpoints_dir / "tom" / "tom_final.pth",
            self.models_dir / "tom_final.pth"
        ]
        
        for tom_path in tom_paths:
            if tom_path.exists():
                size_mb = tom_path.stat().st_size / (1024 * 1024)
                print(f"✅ TOM checkpoint found: {size_mb:.1f} MB")
                print(f"   Path: {tom_path}")
                self.results['tom'] = 'present'
                return True
        
        print("❌ TOM checkpoint not found")
        print("   Searched in:")
        for p in tom_paths:
            print(f"     - {p}")
        print("\n📥 DOWNLOAD REQUIRED:")
        print("   1. Visit: https://drive.google.com/file/d/1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy")
        print("   2. Download tom_final.pth")
        print(f"   3. Save to: {tom_paths[0].parent}/")
        print(f"   4. Create directory if needed: mkdir {tom_paths[0].parent}")
        
        self.results['tom'] = 'missing'
        return False
    
    def check_mediapipe(self):
        """Check MediaPipe installation"""
        print("\n" + "="*70)
        print("STEP 4: MediaPipe Check")
        print("="*70)
        
        try:
            import mediapipe as mp
            print(f"✅ MediaPipe {mp.__version__} installed")
            
            # Check for pose model
            pose_model = self.project_root / "models" / "pose_landmarker_lite.task"
            if pose_model.exists():
                print(f"✅ Pose model found: {pose_model.stat().st_size / 1024:.1f} KB")
                self.results['mediapipe'] = 'ready'
            else:
                print(f"⚠️  Pose model not found at {pose_model}")
                self.results['mediapipe'] = 'partial'
            return True
            
        except ImportError:
            print("❌ MediaPipe not installed")
            print("   Install with: pip install mediapipe")
            self.results['mediapipe'] = 'missing'
            return False
    
    def test_gpu_acceleration(self):
        """Test GPU acceleration capabilities"""
        print("\n" + "="*70)
        print("STEP 5: GPU Acceleration Test")
        print("="*70)
        
        if self.results.get('pytorch') == 'cuda':
            try:
                import torch
                # Simple GPU test
                x = torch.randn(1000, 1000).cuda()
                y = torch.matmul(x, x)
                print("✅ GPU computation test passed")
                print(f"   Device: {y.device}")
                self.results['gpu_test'] = 'passed'
                return True
            except Exception as e:
                print(f"⚠️  GPU test failed: {e}")
                self.results['gpu_test'] = 'failed'
                return False
        else:
            print("⏭️  Skipping GPU test (CUDA not available)")
            self.results['gpu_test'] = 'skipped'
            return True
    
    def generate_report(self):
        """Generate setup status report"""
        print("\n" + "="*70)
        print("PHASE 2 SETUP REPORT")
        print("="*70)
        
        status_icons = {
            'cuda': '✅', 'cpu': '⚠️', 'missing': '❌',
            'present': '✅', 'ready': '✅', 'partial': '⚠️',
            'passed': '✅', 'failed': '❌', 'skipped': '⏭️'
        }
        
        print("\nComponent Status:")
        print(f"  PyTorch:      {status_icons.get(self.results.get('pytorch', 'missing'), '❌')} {self.results.get('pytorch', 'missing').upper()}")
        print(f"  GMM Model:    {status_icons.get(self.results.get('gmm', 'missing'), '❌')} {self.results.get('gmm', 'missing').upper()}")
        print(f"  TOM Model:    {status_icons.get(self.results.get('tom', 'missing'), '❌')} {self.results.get('tom', 'missing').upper()}")
        print(f"  MediaPipe:    {status_icons.get(self.results.get('mediapipe', 'missing'), '❌')} {self.results.get('mediapipe', 'missing').upper()}")
        print(f"  GPU Test:     {status_icons.get(self.results.get('gpu_test', 'skipped'), '⏭️')} {self.results.get('gpu_test', 'skipped').upper()}")
        
        # Overall readiness
        print("\n" + "-"*70)
        all_present = (
            self.results.get('pytorch') in ['cuda', 'cpu'] and
            self.results.get('gmm') == 'present' and
            self.results.get('tom') == 'present' and
            self.results.get('mediapipe') in ['ready', 'partial']
        )
        
        if all_present:
            print("🎉 PHASE 2 READY - All components available!")
            print("\nNext steps:")
            print("  1. Run: python phase2_validation.py")
            print("  2. Test: python app.py --use-gmm")
            print("  3. Benchmark: python tests/stress_test_pipeline.py")
        elif self.results.get('gmm') == 'present' and self.results.get('pytorch') in ['cuda', 'cpu']:
            print("⚠️  PARTIAL READY - GMM warping available (without TOM synthesis)")
            print("\nYou can:")
            print("  1. Use GMM warping: python app.py")
            print("  2. Download TOM for full pipeline")
        else:
            print("❌ NOT READY - Missing critical components")
            print("\nRequired actions:")
            if self.results.get('pytorch') == 'missing':
                print("  • Install PyTorch: pip install torch torchvision")
            if self.results.get('gmm') == 'missing':
                print("  • Download GMM checkpoint (see Step 2 above)")
            if self.results.get('tom') == 'missing':
                print("  • Download TOM checkpoint (see Step 3 above)")
        
        print("="*70 + "\n")
        
        # Save report
        report_path = self.project_root / "phase2_setup_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Report saved to: {report_path}")
    
    def run(self):
        """Run all setup checks"""
        print("\n🚀 PHASE 2 SETUP - Neural Models + GPU Acceleration")
        print("=" * 70)
        
        self.check_pytorch()
        self.check_gmm_checkpoint()
        self.check_tom_checkpoint()
        self.check_mediapipe()
        self.test_gpu_acceleration()
        self.generate_report()


if __name__ == "__main__":
    setup = Phase2Setup()
    setup.run()
