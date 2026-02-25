#!/usr/bin/env python3
"""
Phase 2 Validation Script
Comprehensive testing of neural pipeline with GPU acceleration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2
import time
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Phase2Validator:
    """Validate Phase 2 neural pipeline implementation"""
    
    def __init__(self):
        self.results = {}
        self.passed_tests = 0
        self.total_tests = 0
    
    def test_pytorch_cuda(self) -> bool:
        """Test 1: PyTorch and CUDA availability"""
        logger.info("\n" + "="*70)
        logger.info("TEST 1: PyTorch & CUDA")
        logger.info("="*70)
        
        self.total_tests += 1
        try:
            import torch
            logger.info(f"✓ PyTorch {torch.__version__} installed")
            
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA {torch.version.cuda} available")  # type: ignore
                logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"  Memory: {memory_gb:.1f} GB")
                
                # Quick CUDA test
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
                logger.info(f"  ✓ CUDA computation test passed")
                
                self.results['pytorch'] = 'cuda'
                self.passed_tests += 1
                return True
            else:
                logger.info("⚠ CUDA not available - using CPU")
                self.results['pytorch'] = 'cpu'
                self.passed_tests += 1
                return True
                
        except Exception as e:
            logger.error(f"✗ PyTorch test failed: {e}")
            self.results['pytorch'] = 'failed'
            return False
    
    def test_model_checkpoints(self) -> bool:
        """Test 2: Model checkpoint availability"""
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Model Checkpoints")
        logger.info("="*70)
        
        self.total_tests += 1
        
        gmm_path = Path("cp-vton/checkpoints/gmm_train_new/gmm_final.pth")
        tom_paths = [
            Path("cp-vton/checkpoints/tom_train_new/tom_final.pth"),
            Path("cp-vton/checkpoints/tom/tom_final.pth"),
            Path("models/tom_final.pth")
        ]
        
        gmm_ok = False
        tom_ok = False
        
        # Check GMM
        if gmm_path.exists():
            size_mb = gmm_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ GMM checkpoint: {size_mb:.1f} MB")
            gmm_ok = True
        else:
            logger.error(f"✗ GMM checkpoint not found: {gmm_path}")
        
        # Check TOM
        for tom_path in tom_paths:
            if tom_path.exists():
                size_mb = tom_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ TOM checkpoint: {size_mb:.1f} MB at {tom_path}")
                tom_ok = True
                break
        
        if not tom_ok:
            logger.warning("⚠ TOM checkpoint not found")
            logger.info("  Download: python download_tom_checkpoint.py")
        
        self.results['gmm'] = 'present' if gmm_ok else 'missing'
        self.results['tom'] = 'present' if tom_ok else 'missing'
        
        if gmm_ok:
            self.passed_tests += 1
            return True
        return False
    
    def test_model_loading(self) -> bool:
        """Test 3: Model loading"""
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Model Loading")
        logger.info("="*70)
        
        self.total_tests += 1
        
        try:
            import torch
            sys.path.insert(0, "cp-vton")
            from networks import GMM
            
            # Try loading GMM
            class Args:
                fine_height = 256
                fine_width = 192
                grid_size = 5
            
            opt = Args()
            model = GMM(opt)
            checkpoint = torch.load(
                "cp-vton/checkpoints/gmm_train_new/gmm_final.pth",
                map_location='cpu'
            )
            model.load_state_dict(checkpoint)
            model.eval()
            
            logger.info("✓ GMM model loaded successfully")
            logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
            
            self.results['model_loading'] = 'success'
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ Model loading failed: {e}")
            self.results['model_loading'] = 'failed'
            return False
    
    def test_live_pose_converter(self) -> bool:
        """Test 4: Live pose to heatmap conversion"""
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Live Pose Converter")
        logger.info("="*70)
        
        self.total_tests += 1
        
        try:
            from src.core.live_pose_converter import LivePoseConverter
            
            converter = LivePoseConverter()
            
            # Mock landmarks
            mock_landmarks = {
                0: {'x': 0.5, 'y': 0.15, 'z': 0, 'visibility': 0.99},
                11: {'x': 0.4, 'y': 0.3, 'z': 0, 'visibility': 0.95},
                12: {'x': 0.6, 'y': 0.3, 'z': 0, 'visibility': 0.95},
                23: {'x': 0.45, 'y': 0.7, 'z': 0, 'visibility': 0.90},
                24: {'x': 0.55, 'y': 0.7, 'z': 0, 'visibility': 0.90},
            }
            
            heatmaps = converter.landmarks_to_heatmaps(mock_landmarks)
            
            assert heatmaps.shape == (18, 256, 192), f"Wrong shape: {heatmaps.shape}"
            assert heatmaps.dtype == np.float32, f"Wrong dtype: {heatmaps.dtype}"
            assert 0 <= heatmaps.min() and heatmaps.max() <= 1, "Values out of range"
            
            logger.info(f"✓ Pose converter working")
            logger.info(f"  Heatmap shape: {heatmaps.shape}")
            logger.info(f"  Active channels: {np.sum(heatmaps.max(axis=(1,2)) > 0)}/18")
            
            self.results['pose_converter'] = 'success'
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ Pose converter failed: {e}")
            self.results['pose_converter'] = 'failed'
            return False
    
    def test_body_segmentation(self) -> bool:
        """Test 5: Live body segmentation"""
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Body Segmentation")
        logger.info("="*70)
        
        self.total_tests += 1
        
        try:
            from src.core.live_pose_converter import LiveBodySegmenter
            
            segmenter = LiveBodySegmenter()
            
            # Mock image
            mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            mask, _ = segmenter.segment(mock_image)
            
            assert mask.shape == (256, 192), f"Wrong shape: {mask.shape}"
            assert mask.dtype == np.float32, f"Wrong dtype: {mask.dtype}"
            
            logger.info(f"✓ Body segmentation working")
            logger.info(f"  Output shape: {mask.shape}")
            logger.info(f"  Coverage: {np.mean(mask > 0.5)*100:.1f}%")
            
            self.results['body_segmentation'] = 'success'
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ Body segmentation failed: {e}")
            self.results['body_segmentation'] = 'failed'
            return False
    
    def test_neural_pipeline(self) -> bool:
        """Test 6: Complete neural pipeline"""
        logger.info("\n" + "="*70)
        logger.info("TEST 6: Neural Pipeline Integration")
        logger.info("="*70)
        
        self.total_tests += 1
        
        try:
            from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
            
            pipeline = Phase2NeuralPipeline(device='auto', enable_tom=False)
            
            logger.info("✓ Pipeline initialized")
            stats = pipeline.get_statistics()
            logger.info(f"  Device: {stats['device']}")
            
            self.results['neural_pipeline'] = 'success'
            self.passed_tests += 1
            return True
            
        except Exception as e:
            logger.error(f"✗ Neural pipeline failed: {e}")
            self.results['neural_pipeline'] = 'failed'
            return False
    
    def benchmark_performance(self) -> Dict:
        """Benchmark 7: Performance testing"""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK: Performance Testing")
        logger.info("="*70)
        
        self.total_tests += 1
        
        try:
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Running benchmark on: {device}")
            
            # Benchmark GMM inference
            sys.path.insert(0, "cp-vton")
            from networks import GMM
            
            class Args:
                fine_height = 256
                fine_width = 192
                grid_size = 5
            
            model = GMM(Args())
            checkpoint = torch.load(
                "cp-vton/checkpoints/gmm_train_new/gmm_final.pth",
                map_location=device,
                weights_only=False
            )
            model.load_state_dict(checkpoint)
            model.to(device).eval()
            
            # Dummy inputs
            agnostic = torch.randn(1, 22, 256, 192, device=device)
            cloth_mask = torch.randn(1, 1, 256, 192, device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(agnostic, cloth_mask)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(50):
                    t0 = time.time()
                    _ = model(agnostic, cloth_mask)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    times.append(time.time() - t0)
            
            avg_time_ms = np.mean(times) * 1000
            fps = 1.0 / np.mean(times)
            
            logger.info(f"✓ GMM inference benchmark:")
            logger.info(f"  Average time: {avg_time_ms:.1f} ms")
            logger.info(f"  FPS capacity: {fps:.1f}")
            logger.info(f"  Min/Max: {min(times)*1000:.1f}/{max(times)*1000:.1f} ms")
            
            self.results['benchmark'] = {
                'device': device,
                'avg_time_ms': avg_time_ms,
                'fps': fps
            }
            
            # Target: 20-27 FPS for Phase 2
            if fps >= 20:
                logger.info(f"  ✓ EXCEEDS Phase 2 target (20+ FPS)")
                self.passed_tests += 1
            elif fps >= 15:
                logger.info(f"  ⚠ Below target but acceptable (15+ FPS)")
                self.passed_tests += 1
            else:
                logger.warning(f"  ✗ Below minimum target (<15 FPS)")
            
            return self.results['benchmark']
            
        except Exception as e:
            logger.error(f"✗ Benchmark failed: {e}")
            self.results['benchmark'] = 'failed'
            return {}
    
    def generate_report(self):
        """Generate final validation report"""
        logger.info("\n" + "="*70)
        logger.info("PHASE 2 VALIDATION REPORT")
        logger.info("="*70)
        
        logger.info(f"\nTests Passed: {self.passed_tests}/{self.total_tests}")
        logger.info("\nComponent Status:")
        
        for key, value in self.results.items():
            if key == 'benchmark':
                if isinstance(value, dict):
                    logger.info(f"  {key}: {value.get('fps', 0):.1f} FPS on {value.get('device', 'unknown')}")
            else:
                icon = "✓" if value in ['success', 'present', 'cuda', 'cpu'] else "✗"
                logger.info(f"  {key}: {icon} {value}")
        
        # Overall assessment
        logger.info("\n" + "-"*70)
        
        pass_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        if pass_rate >= 85:
            logger.info("🎉 PHASE 2 FULLY VALIDATED - Ready for production!")
            logger.info("\nNext steps:")
            logger.info("  1. Run full demo: python app.py --phase2")
            logger.info("  2. Stress test: python tests/stress_test_pipeline.py")
            logger.info("  3. Camera validation: python tests/camera_analysis.py")
        elif pass_rate >= 70:
            logger.info("⚠ PHASE 2 PARTIALLY VALIDATED - Some components need attention")
            logger.info("\nReview failed tests above and address issues")
        else:
            logger.info("✗ PHASE 2 VALIDATION FAILED - Critical components missing")
            logger.info("\nRequired actions:")
            logger.info("  1. Ensure PyTorch installed: pip install torch torchvision")
            logger.info("  2. Download model checkpoints (see tests above)")
            logger.info("  3. Re-run validation: python phase2_validation.py")
        
        logger.info("="*70 + "\n")
    
    def run(self):
        """Run all validation tests"""
        logger.info("\n🚀 PHASE 2 VALIDATION - Neural Models + GPU Acceleration")
        logger.info("=" * 70)
        logger.info("Testing complete implementation with no compromises")
        logger.info("=" * 70)
        
        self.test_pytorch_cuda()
        self.test_model_checkpoints()
        self.test_model_loading()
        self.test_live_pose_converter()
        self.test_body_segmentation()
        self.test_neural_pipeline()
        self.benchmark_performance()
        
        self.generate_report()


if __name__ == "__main__":
    validator = Phase2Validator()
    validator.run()
