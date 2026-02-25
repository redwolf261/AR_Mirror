#!/usr/bin/env python3
"""
Preprocessing Optimization Benchmark
Test the impact of preprocessing optimizations on overall FPS
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"{text.center(70)}")
    print(f"{'='*70}\n")


def benchmark_preprocessing():
    """Benchmark preprocessing components"""
    try:
        import torch
        sys.path.insert(0, "C:\\Users\\HP\\Projects\\AR Mirror")
        from src.core.live_pose_converter import LivePoseConverter, LiveBodySegmenter
        
        # Initialize components
        pose_converter = LivePoseConverter(heatmap_size=(256, 192), sigma=3.0)
        body_segmenter = LiveBodySegmenter()
        
        # Create test data
        person_image = (np.random.rand(512, 384, 3) * 255).astype(np.uint8)
        
        # Dummy landmarks
        mp_landmarks = {
            0: {'x': 0.5, 'y': 0.2, 'visibility': 0.99},
            11: {'x': 0.4, 'y': 0.4, 'visibility': 0.95},
            12: {'x': 0.6, 'y': 0.4, 'visibility': 0.95},
            23: {'x': 0.45, 'y': 0.7, 'visibility': 0.90},
            24: {'x': 0.55, 'y': 0.7, 'visibility': 0.90},
        }
        
        num_iterations = 100
        
        # Test 1: Pose heatmap generation
        print_header("TEST 1: Pose Heatmap Generation")
        logger.info(f"  Running {num_iterations} iterations...")
        
        times = []
        for _ in range(num_iterations):
            t0 = time.time()
            heatmaps = pose_converter.landmarks_to_heatmaps(mp_landmarks, frame_shape=(512, 384))
            times.append(time.time() - t0)
        
        avg_time_ms = np.mean(times) * 1000
        fps = 1000.0 / avg_time_ms
        
        print(f"[OK] Pose heatmap generation: {fps:.1f} FPS")
        print(f"  Average time: {avg_time_ms:.2f} ms")
        print(f"  Min/Max: {np.min(times)*1000:.2f} / {np.max(times)*1000:.2f} ms")
        
        # Test 2: Body segmentation
        print_header("TEST 2: Body Segmentation")
        logger.info(f"  Running {num_iterations} iterations...")
        
        times = []
        for _ in range(num_iterations):
            t0 = time.time()
            body_mask, _ = body_segmenter.segment(person_image)
            times.append(time.time() - t0)
        
        avg_time_ms = np.mean(times) * 1000
        fps = 1000.0 / avg_time_ms
        
        print(f"[OK] Body segmentation: {fps:.1f} FPS")
        print(f"  Average time: {avg_time_ms:.2f} ms")
        print(f"  Min/Max: {np.min(times)*1000:.2f} / {np.max(times)*1000:.2f} ms")
        
        # Test 3: Combined preprocessing
        print_header("TEST 3: Combined Preprocessing")
        logger.info(f"  Running {num_iterations} iterations...")
        
        times = []
        for _ in range(num_iterations):
            t0 = time.time()
            heatmaps = pose_converter.landmarks_to_heatmaps(mp_landmarks, frame_shape=(512, 384))
            body_mask, _ = body_segmenter.segment(person_image)
            times.append(time.time() - t0)
        
        avg_time_ms = np.mean(times) * 1000
        fps = 1000.0 / avg_time_ms
        
        print(f"[OK] Combined preprocessing: {fps:.1f} FPS")
        print(f"  Average time: {avg_time_ms:.2f} ms")
        print(f"  Min/Max: {np.min(times)*1000:.2f} / {np.max(times)*1000:.2f} ms")
        
        return {
            'pose_fps': 1000.0 / (np.mean([t for t in times if t > 0]) * 1000),
            'combined_fps': fps
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def benchmark_full_pipeline():
    """Benchmark full pipeline with optimized preprocessing"""
    try:
        import torch
        sys.path.insert(0, "C:\\Users\\HP\\Projects\\AR Mirror")
        from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
        
        # Initialize pipeline with optimizations
        pipeline = Phase2NeuralPipeline(device='cuda', enable_optimizations=True)
        
        # Create test data
        person_image = (np.random.rand(512, 384, 3) * 255).astype(np.uint8)
        cloth_rgb = np.random.rand(512, 384, 3).astype(np.float32)
        cloth_mask = np.random.rand(512, 384).astype(np.float32)
        
        mp_landmarks = {
            0: {'x': 0.5, 'y': 0.2, 'visibility': 0.99},
            11: {'x': 0.4, 'y': 0.4, 'visibility': 0.95},
            12: {'x': 0.6, 'y': 0.4, 'visibility': 0.95},
            23: {'x': 0.45, 'y': 0.7, 'visibility': 0.90},
            24: {'x': 0.55, 'y': 0.7, 'visibility': 0.90},
        }
        
        num_iterations = 50
        
        print_header("TEST 4: Full Pipeline (Optimized)")
        logger.info(f"  Warming up...")
        
        # Warmup
        for _ in range(5):
            _ = pipeline.warp_garment(person_image, cloth_rgb, cloth_mask, mp_landmarks)
            torch.cuda.synchronize()
        
        logger.info(f"  Running {num_iterations} iterations...")
        
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = pipeline.warp_garment(person_image, cloth_rgb, cloth_mask, mp_landmarks)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        
        avg_time_ms = np.mean(times) * 1000
        fps = 1000.0 / avg_time_ms
        
        print(f"[OK] Full pipeline: {fps:.1f} FPS")
        print(f"  Average time: {avg_time_ms:.2f} ms")
        print(f"  Min/Max: {np.min(times)*1000:.2f} / {np.max(times)*1000:.2f} ms")
        
        return fps
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Run preprocessing optimization benchmarks"""
    print_header("PREPROCESSING OPTIMIZATION BENCHMARK")
    print("Testing impact of optimized preprocessing on FPS\n")
    
    # Test preprocessing components
    preprocessing_results = benchmark_preprocessing()
    
    # Test full pipeline
    full_pipeline_fps = benchmark_full_pipeline()
    
    # Summary
    print_header("FINAL RESULTS")
    
    if full_pipeline_fps > 0:
        print(f"Full Pipeline FPS: {full_pipeline_fps:.1f}")
        
        if full_pipeline_fps >= 200:
            print(f"\n[OK] *** TARGET ACHIEVED! *** ({full_pipeline_fps:.1f} FPS >= 200 FPS)")
        else:
            gap = 200 / full_pipeline_fps
            print(f"\n[!] Need {gap:.2f}x more to reach 200 FPS")
            print(f"    Current: {full_pipeline_fps:.1f} FPS")
            print(f"    Target: 200 FPS")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
