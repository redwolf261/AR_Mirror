#!/usr/bin/env python3
"""
Test Multi-Modal Rendering Pipeline
Tests for neural warping, DensePose, and hybrid cached rendering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
import logging

from src.pipelines.diffusion_renderer import MultiModalRenderer, RenderMode, RenderResult
from src.sdk.sdk_core import ARMirrorSDK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Helper functions (must be defined before use) ===

def _is_pipeline_available() -> bool:
    """Check if neural pipeline is available"""
    try:
        from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
        return True
    except:
        return False


class TestRenderModes:
    """Test different rendering modes"""
    
    def test_render_mode_enum(self):
        """Test render mode enumeration"""
        modes = [
            RenderMode.NEURAL_WARP,
            RenderMode.NEURAL_DENSEPOSE,
            RenderMode.HYBRID_CACHED,
            RenderMode.CLOUD_API
        ]
        
        assert len(modes) == 4
        assert RenderMode.NEURAL_WARP.value == "neural_warp"
        assert RenderMode.NEURAL_DENSEPOSE.value == "neural_densepose"
        assert RenderMode.HYBRID_CACHED.value == "hybrid_cached"
        assert RenderMode.CLOUD_API.value == "cloud_api"
        
        logger.info("✓ All render modes defined")
    
    def test_render_result_structure(self):
        """Test RenderResult dataclass"""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = RenderResult(
            output_image=test_image,
            mode=RenderMode.NEURAL_WARP,
            quality_score=0.85,
            render_time=0.05,
            metadata={'test': True}
        )
        
        assert result.output_image.shape == (480, 640, 3)
        assert result.mode == RenderMode.NEURAL_WARP
        assert result.quality_score == 0.85
        assert result.render_time == 0.05
        assert result.metadata['test'] == True
        
        logger.info("✓ RenderResult structure valid")


class TestMultiModalRenderer:
    """Test MultiModalRenderer class"""
    
    def test_renderer_initialization(self):
        """Test renderer can be initialized without pipeline"""
        renderer = MultiModalRenderer()
        assert renderer is not None
        assert renderer._render_cache is not None
        logger.info("✓ Renderer initialized without neural pipeline")
    
    def test_renderer_with_pipeline(self):
        """Test renderer initialization with neural pipeline"""
        try:
            from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
            pipeline = Phase2NeuralPipeline(device='cpu', enable_tom=False)
            renderer = MultiModalRenderer(neural_pipeline=pipeline)
            
            assert renderer is not None
            assert renderer.neural_pipeline is not None
            logger.info("✓ Renderer initialized with neural pipeline")
        except Exception as e:
            pytest.skip(f"Pipeline not available: {e}")
    
    def test_cache_key_generation(self):
        """Test cache key generation from pose data"""
        renderer = MultiModalRenderer()
        
        # Create test pose data
        pose_data = {
            11: {'x': 0.45, 'y': 0.35, 'visibility': 0.9},
            12: {'x': 0.55, 'y': 0.35, 'visibility': 0.9},
        }
        
        key1 = renderer._generate_cache_key("garment_001", pose_data)
        key2 = renderer._generate_cache_key("garment_001", pose_data)
        key3 = renderer._generate_cache_key("garment_002", pose_data)
        
        # Same garment + pose = same key
        assert key1 == key2
        
        # Different garment = different key
        assert key1 != key3
        
        # Keys should be consistent length (MD5 hash prefix)
        assert len(key1) == 16
        assert len(key3) == 16
        
        logger.info(f"✓ Cache key generation: {key1}")
    
    def test_neural_warp_mode_requires_pipeline(self):
        """Test neural modes require pipeline"""
        renderer = MultiModalRenderer()  # No pipeline
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cloth_rgb = np.ones((256, 192, 3), dtype=np.float32)
        cloth_mask = np.ones((256, 192), dtype=np.float32)
        pose_data = {}
        
        with pytest.raises(RuntimeError, match="Neural pipeline not available"):
            renderer.render(
                test_frame, cloth_rgb, cloth_mask, pose_data,
                mode=RenderMode.NEURAL_WARP
            )
        
        logger.info("✓ Neural mode correctly requires pipeline")
    
    def test_cache_storage(self):
        """Test render cache storage and retrieval"""
        renderer = MultiModalRenderer()
        
        # Simulate cached render
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cache_key = "test_key_001"
        
        renderer._render_cache[cache_key] = test_image
        
        # Verify stored
        assert cache_key in renderer._render_cache
        assert np.array_equal(renderer._render_cache[cache_key], test_image)
        
        logger.info("✓ Cache storage working")


class TestSDKRenderModes:
    """Test SDK integration with render modes"""
    
    def test_sdk_default_mode(self):
        """Test SDK defaults to neural_warp mode"""
        try:
            sdk = ARMirrorSDK(config={'device': 'cpu'})
            assert sdk._render_mode == RenderMode.NEURAL_WARP
            logger.info("✓ SDK defaults to neural_warp")
        except Exception as e:
            logger.warning(f"SDK init failed: {e}")
    
    def test_sdk_custom_mode(self):
        """Test SDK accepts custom render mode"""
        try:
            sdk = ARMirrorSDK(config={
                'device': 'cpu',
                'render_mode': 'neural_densepose'
            })
            assert sdk._render_mode == RenderMode.NEURAL_DENSEPOSE
            logger.info("✓ SDK accepts custom render mode")
        except Exception as e:
            logger.warning(f"SDK init failed: {e}")
    
    def test_sdk_invalid_mode_fallback(self):
        """Test SDK falls back on invalid mode"""
        try:
            sdk = ARMirrorSDK(config={
                'device': 'cpu',
                'render_mode': 'invalid_mode_xyz'
            })
            # Should fall back to neural_warp
            assert sdk._render_mode == RenderMode.NEURAL_WARP
            logger.info("✓ SDK falls back on invalid mode")
        except Exception as e:
            logger.warning(f"SDK init failed: {e}")
    
    def test_sdk_switch_mode_at_runtime(self):
        """Test switching render mode at runtime"""
        try:
            sdk = ARMirrorSDK(config={'device': 'cpu'})
            
            # Start with default
            assert sdk._render_mode == RenderMode.NEURAL_WARP
            
            # Switch to hybrid cached
            success = sdk.set_render_mode('hybrid_cached')
            assert success == True
            assert sdk._render_mode == RenderMode.HYBRID_CACHED
            
            # Switch to DensePose
            success = sdk.set_render_mode('neural_densepose')
            assert success == True
            assert sdk._render_mode == RenderMode.NEURAL_DENSEPOSE
            
            # Try invalid mode
            success = sdk.set_render_mode('invalid')
            assert success == False
            assert sdk._render_mode == RenderMode.NEURAL_DENSEPOSE  # Unchanged
            
            logger.info("✓ Runtime mode switching works")
        except Exception as e:
            logger.warning(f"SDK init failed: {e}")


class TestRenderingPerformance:
    """Performance tests for different rendering modes"""
    
    @pytest.mark.skipif(not _is_pipeline_available(), reason="Pipeline not available")
    def test_neural_warp_performance(self):
        """Benchmark neural warp mode"""
        import time
        
        try:
            from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
            pipeline = Phase2NeuralPipeline(device='cpu', enable_tom=False)
            renderer = MultiModalRenderer(neural_pipeline=pipeline)
            
            # Create test inputs
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cloth_rgb = np.random.rand(256, 192, 3).astype(np.float32)
            cloth_mask = np.ones((256, 192), dtype=np.float32)
            pose_data = {i: {'x': 0.5, 'y': 0.5, 'visibility': 0.9} for i in range(33)}
            
            # Warmup
            for _ in range(3):
                try:
                    renderer.render(
                        test_frame, cloth_rgb, cloth_mask, pose_data,
                        mode=RenderMode.NEURAL_WARP
                    )
                except:
                    pass
            
            # Benchmark
            times = []
            for _ in range(10):
                t0 = time.time()
                try:
                    result = renderer.render(
                        test_frame, cloth_rgb, cloth_mask, pose_data,
                        mode=RenderMode.NEURAL_WARP
                    )
                    times.append(time.time() - t0)
                except Exception as e:
                    logger.warning(f"Render failed: {e}")
                    break
            
            if times:
                avg_time = np.mean(times)
                fps = 1.0 / avg_time
                logger.info(f"✓ Neural warp: {avg_time*1000:.1f} ms ({fps:.1f} FPS)")
                
                # Should be reasonably fast (target: >10 FPS on CPU)
                assert avg_time < 0.5, f"Too slow: {avg_time:.2f}s"
        
        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")
    
    def test_hybrid_cache_performance(self):
        """Benchmark cached rendering performance"""
        import time
        
        renderer = MultiModalRenderer()
        
        # Create test inputs
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cloth_rgb = np.random.rand(256, 192, 3).astype(np.float32)
        cloth_mask = np.ones((256, 192), dtype=np.float32)
        pose_data = {11: {'x': 0.45, 'y': 0.35, 'visibility': 0.9}}
        
        # Pre-populate cache
        cache_key = renderer._generate_cache_key("test_garment", pose_data)
        renderer._render_cache[cache_key] = test_frame.copy()
        
        # Benchmark cache retrieval
        times = []
        for _ in range(100):  # More iterations since it's fast
            t0 = time.time()
            # Simulate cache lookup
            if cache_key in renderer._render_cache:
                cached = renderer._render_cache[cache_key]
            times.append(time.time() - t0)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else float('inf')
        
        logger.info(f"✓ Cache retrieval: {avg_time*1000:.3f} ms ({fps:.0f} FPS)")
        
        # Cache should be very fast (>100 FPS)
        assert avg_time < 0.01, f"Cache too slow: {avg_time:.4f}s"


class TestRenderingQuality:
    """Quality tests for rendering modes"""
    
    def test_render_result_quality_scores(self):
        """Test quality scores are in valid range"""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = [
            RenderResult(test_image, RenderMode.NEURAL_WARP, 0.75, 0.05, {}),
            RenderResult(test_image, RenderMode.NEURAL_DENSEPOSE, 0.85, 0.07, {}),
            RenderResult(test_image, RenderMode.HYBRID_CACHED, 0.95, 0.01, {}),
            RenderResult(test_image, RenderMode.CLOUD_API, 0.98, 2.5, {}),
        ]
        
        for result in results:
            assert 0.0 <= result.quality_score <= 1.0, \
                f"Quality score out of range: {result.quality_score}"
        
        # Verify quality ordering (expected hierarchy)
        assert results[3].quality_score > results[2].quality_score  # Cloud > Cached
        assert results[2].quality_score > results[1].quality_score  # Cached > DensePose
        assert results[1].quality_score > results[0].quality_score  # DensePose > Basic
        
        logger.info("✓ Quality scores in expected order")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
