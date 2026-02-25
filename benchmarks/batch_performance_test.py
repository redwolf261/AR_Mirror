#!/usr/bin/env python3
"""
Batch Processing Performance Benchmark
Test single-frame vs batch processing performance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import time
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"{text.center(70)}")
    print(f"{'='*70}\n")


def benchmark_batch_processing(batch_sizes: List[int] = [1, 2, 4, 8], num_iterations: int = 50):
    """Benchmark batch processing performance"""
    try:
        import torch
        import sys
        sys.path.insert(0, "C:\\Users\\HP\\Projects\\AR Mirror")
        from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
        
        # Initialize pipeline with optimizations
        pipeline = Phase2NeuralPipeline(device='cuda', enable_optimizations=True)
        
        # Create dummy data with correct types
        person_image = (np.random.rand(512, 384, 3) * 255).astype(np.uint8)  # uint8 for images
        cloth_rgb = np.random.rand(512, 384, 3).astype(np.float32)  # float32 normalized
        cloth_mask = np.random.rand(512, 384).astype(np.float32)  # float32 normalized
        
        # Dummy landmarks (minimal for testing)
        mp_landmarks = {
            0: {'x': 0.5, 'y': 0.2},  # Nose
            11: {'x': 0.4, 'y': 0.4},  # Left shoulder
            12: {'x': 0.6, 'y': 0.4},  # Right shoulder
        }
        
        results = {}
        
        for batch_size in batch_sizes:
            print_header(f"BATCH SIZE: {batch_size}")
            
            # Prepare batch data
            person_images = [person_image] * batch_size
            mp_landmarks_list = [mp_landmarks] * batch_size
            
            # Warmup
            logger.info(f"  Warming up (batch_size={batch_size})...")
            for _ in range(5):
                if batch_size == 1:
                    _ = pipeline.warp_garment(person_image, cloth_rgb, cloth_mask, mp_landmarks)
                else:
                    _ = pipeline.warp_garment_batch(person_images, cloth_rgb, cloth_mask, mp_landmarks_list)
                if pipeline.device == 'cuda':
                    torch.cuda.synchronize()
            
            # Benchmark
            logger.info(f"  Running {num_iterations} iterations...")
            times = []
            
            for _ in range(num_iterations):
                if pipeline.device == 'cuda':
                    torch.cuda.synchronize()
                
                t0 = time.time()
                
                if batch_size == 1:
                    _ = pipeline.warp_garment(person_image, cloth_rgb, cloth_mask, mp_landmarks)
                else:
                    _ = pipeline.warp_garment_batch(person_images, cloth_rgb, cloth_mask, mp_landmarks_list)
                
                if pipeline.device == 'cuda':
                    torch.cuda.synchronize()
                
                times.append(time.time() - t0)
            
            # Calculate statistics
            avg_time_ms = np.mean(times) * 1000
            std_time_ms = np.std(times) * 1000
            min_time_ms = np.min(times) * 1000
            max_time_ms = np.max(times) * 1000
            
            # Calculate per-frame metrics
            avg_time_per_frame_ms = avg_time_ms / batch_size
            fps_per_frame = 1000.0 / avg_time_per_frame_ms
            throughput_fps = batch_size / (avg_time_ms / 1000.0)
            
            results[batch_size] = {
                'batch_size': batch_size,
                'avg_time_ms': avg_time_ms,
                'std_time_ms': std_time_ms,
                'min_time_ms': min_time_ms,
                'max_time_ms': max_time_ms,
                'avg_time_per_frame_ms': avg_time_per_frame_ms,
                'fps_per_frame': fps_per_frame,
                'throughput_fps': throughput_fps
            }
            
            print(f"[OK] Batch {batch_size}: {throughput_fps:.1f} FPS throughput")
            print(f"  Total time: {avg_time_ms:.2f} +/- {std_time_ms:.2f} ms")
            print(f"  Per-frame: {avg_time_per_frame_ms:.2f} ms ({fps_per_frame:.1f} FPS)")
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Run batch processing benchmarks"""
    print_header("BATCH PROCESSING PERFORMANCE BENCHMARK")
    print("Testing single-frame vs batch processing\n")
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8]
    results = benchmark_batch_processing(batch_sizes=batch_sizes, num_iterations=50)
    
    if not results:
        print("[X] Benchmark failed!")
        return
    
    # Summary
    print_header("PERFORMANCE SUMMARY")
    
    print("Batch | Total Time | Per-Frame | Throughput | vs Batch-1 | GPU Util")
    print("-" * 80)
    
    baseline_fps = results[1]['throughput_fps'] if 1 in results else 0
    
    for batch_size in batch_sizes:
        if batch_size not in results:
            continue
        
        r = results[batch_size]
        speedup = r['throughput_fps'] / baseline_fps if baseline_fps > 0 else 0
        gpu_util = (r['throughput_fps'] / (baseline_fps * batch_size)) * 100 if baseline_fps > 0 else 0
        
        print(f"{batch_size:5d} | {r['avg_time_ms']:10.2f} | {r['avg_time_per_frame_ms']:9.2f} | "
              f"{r['throughput_fps']:10.1f} | {speedup:10.2f}x | {gpu_util:7.1f}%")
    
    print("\n" + "="*80)
    
    # Analysis
    print_header("ANALYSIS")
    
    if 1 in results and 4 in results:
        single_fps = results[1]['throughput_fps']
        batch4_fps = results[4]['throughput_fps']
        improvement = batch4_fps / single_fps
        
        print(f"[OK] Single-frame: {single_fps:.1f} FPS")
        print(f"[OK] Batch-4: {batch4_fps:.1f} FPS")
        print(f"[OK] Improvement: {improvement:.2f}x")
        
        if batch4_fps >= 200:
            print(f"\n[OK] TARGET ACHIEVED! ({batch4_fps:.1f} FPS >= 200 FPS)")
        else:
            remaining = 200 / batch4_fps
            print(f"\n[!] Need {remaining:.2f}x more to reach 200 FPS")
        
        # Optimal batch size
        best_batch = max(results.keys(), key=lambda k: results[k]['throughput_fps'])
        best_fps = results[best_batch]['throughput_fps']
        print(f"\n[OK] Optimal batch size: {best_batch} ({best_fps:.1f} FPS)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
