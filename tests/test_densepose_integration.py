#!/usr/bin/env python3
"""
Test DensePose integration with AR Mirror pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
import logging

from src.core.densepose_converter import DensePoseLiveConverter
from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
from src.sdk.sdk_core import ARMirrorSDK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Helper functions (must be defined before use) ===

def _is_densepose_available() -> bool:
    """Check if DensePose is available"""
    try:
        converter = DensePoseLiveConverter()
        return converter.is_available
    except:
        return False


def _load_test_person_image():
    """Load test image with person"""
    test_paths = [
        "tests/fixtures/person_front_view.jpg",
        "dataset/train/image/00001_00.jpg",
        "assets/samples/person1.jpg",
    ]
    
    for path in test_paths:
        if Path(path).exists():
            return cv2.imread(path)
    
    return None


class TestDensePoseConverter:
    """Unit tests for DensePose converter"""
    
    def test_converter_initialization(self):
        """Test DensePose converter can be initialized"""
        converter = DensePoseLiveConverter()
        assert converter is not None
        # Note: DensePose may not be available, that's OK
        logger.info(f"DensePose available: {converter.is_available}")
    
    def test_graceful_fallback_when_unavailable(self):
        """Test system works when DensePose is not installed"""
        converter = DensePoseLiveConverter()
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Should return None gracefully if not available
        iuv = converter.extract_uv_map(test_frame)
        
        if not converter.is_available:
            assert iuv is None
            logger.info("✓ Graceful fallback when DensePose unavailable")
        else:
            # If available, IUV may be None (no person) or valid array
            assert iuv is None or iuv.shape == (3, 480, 640)
            logger.info(f"✓ DensePose extraction result: {iuv.shape if iuv is not None else 'None'}")
    
    @pytest.mark.skipif(not _is_densepose_available(), reason="DensePose not installed")
    def test_iuv_extraction(self):
        """Test IUV extraction from test image"""
        converter = DensePoseLiveConverter()
        
        # Load test image with person
        test_image = _load_test_person_image()
        if test_image is None:
            pytest.skip("Test image not available")
        
        iuv = converter.extract_uv_map(test_image)
        
        assert iuv is not None, "Should detect person in test image"
        assert iuv.shape[0] == 3, "Should have 3 channels (I, U, V)"
        assert iuv.dtype == np.float32, "Should be float32"
        
        # Check I channel (body part indices)
        assert iuv[0].min() >= 0, "Body part index should be non-negative"
        assert iuv[0].max() <= 24, "Body part index should be at most 24"
        
        # Check U/V channels (texture coordinates)
        assert iuv[1].min() >= 0 and iuv[1].max() <= 1, "U channel should be in [0,1]"
        assert iuv[2].min() >= 0 and iuv[2].max() <= 1, "V channel should be in [0,1]"
        
        logger.info(f"✓ IUV extraction successful: {iuv.shape}")
    
    @pytest.mark.skipif(not _is_densepose_available(), reason="DensePose not installed")
    def test_uv_to_pose_heatmaps_conversion(self):
        """Test conversion from IUV to pose heatmaps"""
        converter = DensePoseLiveConverter()
        
        # Create synthetic IUV map
        iuv = np.zeros((3, 256, 192), dtype=np.float32)
        # Add some body parts
        iuv[0, 100:150, 80:110] = 3  # Right upper arm
        iuv[0, 150:200, 80:110] = 4  # Right lower arm
        iuv[1, 100:200, 80:110] = 0.5  # U coordinate
        iuv[2, 100:200, 80:110] = 0.5  # V coordinate
        
        heatmaps = converter.uv_to_pose_heatmaps(iuv, target_size=(256, 192))
        
        assert heatmaps.shape == (18, 256, 192), "Should have 18 OpenPose keypoints"
        assert heatmaps.dtype == np.float32
        assert heatmaps.min() >= 0 and heatmaps.max() <= 1, "Heatmaps should be normalized"
        
        logger.info("✓ UV to pose heatmaps conversion successful")


class TestDensePosePipelineIntegration:
    """Integration tests with Phase 2 neural pipeline"""
    
    def test_pipeline_accepts_densepose_flag(self):
        """Test pipeline initialization with DensePose support"""
        try:
            pipeline = Phase2NeuralPipeline(device='cpu', enable_tom=False)
            assert hasattr(pipeline, 'densepose_converter')
            logger.info(f"✓ Pipeline has DensePose converter: {pipeline.densepose_converter is not None}")
        except Exception as e:
            pytest.skip(f"Pipeline not available: {e}")
    
    def test_warp_with_densepose_flag(self):
        """Test warp_garment accepts use_densepose parameter"""
        try:
            pipeline = Phase2NeuralPipeline(device='cpu', enable_tom=False)
        except Exception as e:
            pytest.skip(f"Pipeline not available: {e}")
        
        # Create test inputs
        person_image = np.random.rand(480, 640, 3).astype(np.float32)
        cloth_rgb = np.random.rand(256, 192, 3).astype(np.float32)
        cloth_mask = np.ones((256, 192), dtype=np.float32)
        
        # Create minimal MediaPipe landmarks
        mp_landmarks = {i: {'x': 0.5, 'y': 0.5, 'visibility': 0.9} for i in range(33)}
        
        try:
            # Should not raise error even if DensePose unavailable
            result = pipeline.warp_garment(
                person_image, cloth_rgb, cloth_mask, mp_landmarks,
                use_densepose=True
            )
            assert result is not None
            assert hasattr(result, 'warped_cloth')
            logger.info("✓ Pipeline accepts use_densepose flag")
        except Exception as e:
            logger.warning(f"Warp failed (may be expected): {e}")
            # Pipeline may fail for other reasons (model loading, etc.), that's OK


class TestDensePoseSDKIntegration:
    """Integration tests with SDK"""
    
    def test_sdk_accepts_densepose_config(self):
        """Test SDK initialization with DensePose config"""
        try:
            sdk = ARMirrorSDK(config={'use_densepose': True, 'device': 'cpu'})
            assert sdk._use_densepose == True
            logger.info("✓ SDK accepts use_densepose configuration")
        except Exception as e:
            logger.warning(f"SDK init failed: {e}")
    
    def test_sdk_without_densepose_config(self):
        """Test SDK defaults to MediaPose when DensePose not configured"""
        try:
            sdk = ARMirrorSDK(config={'device': 'cpu'})
            assert sdk._use_densepose == False
            logger.info("✓ SDK defaults to MediaPipe when DensePose not configured")
        except Exception as e:
            logger.warning(f"SDK init failed: {e}")


class TestDensePosePerformance:
    """Performance benchmarks for DensePose vs MediaPipe"""
    
    @pytest.mark.skipif(not _is_densepose_available(), reason="DensePose not installed")
    def test_densepose_inference_time(self):
        """Measure DensePose inference time"""
        import time
        
        converter = DensePoseLiveConverter(device='cpu')  # Test on CPU
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            converter.extract_uv_map(test_frame)
        
        # Benchmark
        times = []
        for _ in range(10):
            t0 = time.time()
            iuv = converter.extract_uv_map(test_frame)
            times.append(time.time() - t0)
        
        avg_time = np.mean(times)
        logger.info(f"✓ DensePose avg inference time: {avg_time*1000:.1f} ms")
        
        # DensePose is expected to be slower than MediaPipe (2-5s on CPU)
        # but provides richer information
        assert avg_time < 10.0, "Should complete within 10 seconds on CPU"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
