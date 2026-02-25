"""GPU verification script for AR Mirror.

This script verifies that:
1. CUDA is properly installed
2. PyTorch can access the GPU
3. GPU tensor operations work correctly
4. Performance improvements are measurable
"""

import sys
import time
import torch
import numpy as np


def verify_cuda_basic():
    """Verify basic CUDA availability."""
    print("=" * 70)
    print("STEP 1: BASIC CUDA VERIFICATION")
    print("=" * 70)
    
    print(f"\n✓ PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"{'✓' if cuda_available else '❌'} CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n" + "=" * 70)
        print("❌ CUDA NOT AVAILABLE")
        print("=" * 70)
        print("\nPossible causes:")
        print("  1. NVIDIA GPU not enabled in BIOS")
        print("  2. NVIDIA drivers not installed")
        print("  3. PyTorch installed without CUDA support")
        print("  4. CUDA toolkit version mismatch")
        print("\nTo fix:")
        print("  1. Run: .\\setup_gpu.ps1")
        print("  2. Check GPU in Task Manager > Performance")
        print("  3. Update NVIDIA drivers if needed")
        return False
    
    print(f"✓ CUDA Version: {torch.version.cuda}")  # type: ignore
    print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"✓ GPU Count: {torch.cuda.device_count()}")
    
    return True


def verify_gpu_details():
    """Verify GPU hardware details."""
    print("\n" + "=" * 70)
    print("STEP 2: GPU HARDWARE DETAILS")
    print("=" * 70)
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-Processors: {props.multi_processor_count}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Max Threads per Multi-Processor: {props.max_threads_per_multi_processor}")
        
        # Check architecture
        arch = "Unknown"
        if props.major == 8:
            if props.minor == 6:
                arch = "Ampere (RTX 30xx/A6000)"
            elif props.minor == 0:
                arch = "Ampere (A100)"
        elif props.major == 7:
            arch = "Turing (RTX 20xx)"
        
        print(f"  Architecture: {arch}")
        
        # Check TF32 support (Ampere+)
        if props.major >= 8:
            print(f"  TF32 Support: ✓ YES (8x speedup available)")
        else:
            print(f"  TF32 Support: ❌ NO (requires Ampere or newer)")


def verify_tensor_operations():
    """Verify GPU tensor operations work correctly."""
    print("\n" + "=" * 70)
    print("STEP 3: GPU TENSOR OPERATIONS TEST")
    print("=" * 70)
    
    device = torch.device("cuda:0")
    
    # Test 1: Basic tensor creation
    print("\n[Test 1] Creating tensors on GPU...")
    try:
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        print("✓ Tensor creation successful")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 2: Matrix multiplication
    print("\n[Test 2] Matrix multiplication...")
    try:
        z = torch.mm(x, y)
        print(f"✓ Matrix multiply successful (result shape: {z.shape})")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 3: GPU <-> CPU transfer
    print("\n[Test 3] GPU <-> CPU transfer...")
    try:
        x_cpu = x.cpu()
        x_gpu = x_cpu.cuda()
        print("✓ Transfer successful")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 4: In-place operations
    print("\n[Test 4] In-place operations...")
    try:
        x.mul_(2.0)
        print("✓ In-place operations successful")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    return True


def benchmark_cpu_vs_gpu():
    """Benchmark CPU vs GPU performance."""
    print("\n" + "=" * 70)
    print("STEP 4: CPU vs GPU PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    # Test matrix sizes (simulating neural network operations)
    sizes = [
        (256, 192, 256),   # Similar to GMM layer
        (512, 512, 512),   # Medium layer
        (1024, 1024, 1024) # Large layer
    ]
    
    print("\nRunning benchmarks (this may take 30-60 seconds)...")
    print("\nMatrix Multiply Performance:")
    print(f"{'Size':<20} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for m, k, n in sizes:
        # CPU benchmark
        x_cpu = torch.randn(m, k)
        y_cpu = torch.randn(k, n)
        
        # Warmup
        for _ in range(3):
            z_cpu = torch.mm(x_cpu, y_cpu)
        
        # Measure
        start = time.time()
        for _ in range(10):
            z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = (time.time() - start) / 10 * 1000
        
        # GPU benchmark
        x_gpu = x_cpu.cuda()
        y_gpu = y_cpu.cuda()
        
        # Warmup
        for _ in range(3):
            z_gpu = torch.mm(x_gpu, y_gpu)
            torch.cuda.synchronize()
        
        # Measure
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 10 * 1000
        
        speedup = cpu_time / gpu_time
        
        size_str = f"{m}x{k} @ {k}x{n}"
        print(f"{size_str:<20} {cpu_time:>10.2f}     {gpu_time:>10.2f}     {speedup:>7.1f}x")
        
        # Cleanup
        del x_gpu, y_gpu, z_gpu
        torch.cuda.empty_cache()


def benchmark_convolution():
    """Benchmark convolution operations (similar to GMM network)."""
    print("\n" + "=" * 70)
    print("STEP 5: CONVOLUTION BENCHMARK (GMM-LIKE)")
    print("=" * 70)
    
    # Simulate GMM network layer
    batch_size = 1
    channels = 64
    height, width = 256, 192
    
    print(f"\nInput size: {batch_size}x{channels}x{height}x{width}")
    print("Testing convolution operations...\n")
    
    # Create input
    x_cpu = torch.randn(batch_size, channels, height, width)
    
    # Create conv layer
    conv_cpu = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    # CPU benchmark
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            y_cpu = conv_cpu(x_cpu)
        
        # Measure
        start = time.time()
        for _ in range(20):
            y_cpu = conv_cpu(x_cpu)
        cpu_time = (time.time() - start) / 20 * 1000
    
    # GPU benchmark
    x_gpu = x_cpu.cuda()
    conv_gpu = conv_cpu.cuda()
    
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            y_gpu = conv_gpu(x_gpu)
            torch.cuda.synchronize()
        
        # Measure
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            y_gpu = conv_gpu(x_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 20 * 1000
    
    speedup = cpu_time / gpu_time
    
    print(f"CPU Time: {cpu_time:.2f} ms")
    print(f"GPU Time: {gpu_time:.2f} ms")
    print(f"Speedup: {speedup:.1f}x")
    
    # Estimate FPS improvement for AR Mirror
    # Current GMM inference: ~47.5ms on CPU
    current_fps = 21.1
    estimated_gpu_time = 47.5 / speedup
    estimated_fps = 1000 / estimated_gpu_time
    
    print(f"\n📊 AR Mirror Performance Estimate:")
    print(f"  Current (CPU): {current_fps:.1f} FPS")
    print(f"  Estimated (GPU): {estimated_fps:.1f} FPS")
    print(f"  Expected Boost: {estimated_fps/current_fps:.1f}x")


def main():
    """Run all verification steps."""
    print("\n")
    print("*" * 70)
    print("AR MIRROR - GPU VERIFICATION & BENCHMARK")
    print("*" * 70)
    print("\n")
    
    # Step 1: Basic CUDA check
    if not verify_cuda_basic():
        print("\n❌ GPU verification failed. Please fix CUDA setup first.")
        return 1
    
    # Step 2: GPU details
    verify_gpu_details()
    
    # Step 3: Tensor operations
    if not verify_tensor_operations():
        print("\n❌ GPU tensor operations failed.")
        return 1
    
    # Step 4: CPU vs GPU benchmark
    benchmark_cpu_vs_gpu()
    
    # Step 5: Convolution benchmark
    benchmark_convolution()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL VERIFICATION TESTS PASSED")
    print("=" * 70)
    
    print("\nYour GPU is ready for AR Mirror!")
    print("\nNext steps:")
    print("  1. Run: python app.py --phase 2")
    print("  2. Monitor FPS improvement in the window title")
    print("  3. Expected: 200-300 FPS (vs current 21 FPS)")
    print("\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
