#!/usr/bin/env python3
"""
ONNX Runtime GMM with PyTorch CUDA Libraries
Uses PyTorch's bundled CUDA libraries for ONNX Runtime
"""

import numpy as np
import onnxruntime as ort
import logging
import os
import sys
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_cuda_for_onnx_runtime():
    """
    Add PyTorch's CUDA libraries to PATH for ONNX Runtime
    """
    # Get PyTorch's lib directory
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    
    if os.path.exists(torch_lib_dir):
        # Add to PATH
        os.environ['PATH'] = torch_lib_dir + os.pathsep + os.environ.get('PATH', '')
        logger.info(f"Added PyTorch CUDA libraries to PATH: {torch_lib_dir}")
        return True
    else:
        logger.warning(f"PyTorch lib directory not found: {torch_lib_dir}")
        return False


def benchmark_onnx_runtime(model_path: str = "models/gmm_model.onnx", num_iterations: int = 100):
    """
    Benchmark ONNX Runtime with CUDA support
    """
    print("="*70)
    print("ONNX RUNTIME CUDA BENCHMARK".center(70))
    print("="*70)
    print()
    
    # Setup CUDA libraries
    setup_cuda_for_onnx_runtime()
    
    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    print()
    
    # Try CUDA provider first, fall back to CPU
    providers_to_try = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider'
    ]
    
    # Create session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Create session
    logger.info(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers_to_try
    )
    
    active_provider = session.get_providers()[0]
    logger.info(f"Active provider: {active_provider}")
    
    # Create test inputs
    agnostic = np.random.randn(1, 22, 256, 192).astype(np.float32)
    cloth_mask = np.random.randn(1, 1, 256, 192).astype(np.float32)
    
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    # Warmup
    print(f"\nWarming up (10 iterations)...")
    for _ in range(10):
        ort_inputs = {
            input_names[0]: agnostic,
            input_names[1]: cloth_mask
        }
        _ = session.run(output_names, ort_inputs)
    
    # Benchmark
    print(f"\nBenchmarking ({num_iterations} iterations)...")
    
    import time
    times = []
    for _ in range(num_iterations):
        ort_inputs = {
            input_names[0]: agnostic,
            input_names[1]: cloth_mask
        }
        
        t0 = time.time()
        ort_outputs = session.run(output_names, ort_inputs)
        times.append(time.time() - t0)
    
    avg_time_ms = np.mean(times) * 1000
    fps = 1000.0 / avg_time_ms
    
    print(f"\n{'='*70}")
    print(f"RESULTS - {active_provider}".center(70))
    print(f"{'='*70}")
    print(f"\n  Average time: {avg_time_ms:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  Min/Max: {np.min(times)*1000:.2f} / {np.max(times)*1000:.2f} ms")
    
    if active_provider == 'CUDAExecutionProvider':
        print(f"\n  [OK] CUDA acceleration working!")
    else:
        print(f"\n  [!] Running on CPU (CUDA not available)")
    
    print(f"\n{'='*70}\n")
    
    return fps, active_provider


if __name__ == "__main__":
    try:
        fps, provider = benchmark_onnx_runtime()
        
        # Compare with PyTorch
        print(f"\nComparison:")
        print(f"  ONNX Runtime ({provider}): {fps:.1f} FPS")
        print(f"  PyTorch GPU Optimized: 163.6 FPS (baseline)")
        
        if fps > 163.6:
            improvement = fps / 163.6
            print(f"\n  [OK] ONNX Runtime is {improvement:.2f}x faster!")
        else:
            ratio = 163.6 / fps
            print(f"\n  [!] PyTorch is {ratio:.2f}x faster. Stick with PyTorch.")
        
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
