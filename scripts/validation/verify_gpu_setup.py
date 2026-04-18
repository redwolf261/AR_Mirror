#!/usr/bin/env python3
"""
GPU Setup Verification Script
Comprehensive check for CUDA, GPU drivers, and PyTorch integration
"""

import sys
import subprocess
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{BLUE}{BOLD}{'='*70}{RESET}")
    print(f"{BLUE}{BOLD}{text.center(70)}{RESET}")
    print(f"{BLUE}{BOLD}{'='*70}{RESET}\n")


def print_success(text):
    """Print success message"""
    print(f"{GREEN}[OK] {text}{RESET}")


def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}[!] {text}{RESET}")


def print_error(text):
    """Print error message"""
    print(f"{RED}[X] {text}{RESET}")


def print_info(text, indent=2):
    """Print info message"""
    print(f"{' '*indent}{text}")


def check_gpu_hardware():
    """Check if NVIDIA GPU is detected by Windows"""
    print_header("1. HARDWARE DETECTION")
    
    try:
        # Check using wmic
        result = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        gpus = [line.strip() for line in result.stdout.split('\n') if line.strip() and line.strip() != 'Name']
        
        nvidia_found = False
        for gpu in gpus:
            if 'NVIDIA' in gpu.upper():
                print_success(f"NVIDIA GPU detected: {gpu}")
                nvidia_found = True
            else:
                print_info(f"Other GPU: {gpu}")
        
        if not nvidia_found:
            print_error("NVIDIA GPU not detected by Windows")
            print_info("Action: Enable discrete GPU in BIOS", 4)
            print_info("Guide: docs/GPU_BIOS_SETUP_GUIDE.md", 4)
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Failed to check GPU hardware: {e}")
        return False


def check_nvidia_driver():
    """Check NVIDIA driver status"""
    print_header("2. NVIDIA DRIVER")
    
    try:
        # Try to run nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            info = result.stdout.strip().split(',')
            if len(info) >= 3:
                print_success(f"GPU: {info[0].strip()}")
                print_success(f"Driver Version: {info[1].strip()}")
                print_success(f"VRAM: {info[2].strip()}")
                return True
        
        print_warning("nvidia-smi not found or failed")
        print_info("Action: Install/Update NVIDIA drivers", 4)
        print_info("Download: https://www.nvidia.com/download/index.aspx", 4)
        return False
        
    except FileNotFoundError:
        print_warning("nvidia-smi not found")
        print_info("Action: Install NVIDIA drivers", 4)
        return False
    except Exception as e:
        print_error(f"Driver check failed: {e}")
        return False


def check_pytorch_cuda():
    """Check PyTorch CUDA integration"""
    print_header("3. PYTORCH CUDA INTEGRATION")
    
    try:
        import torch
        
        print_success(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_success("CUDA is available!")
            print_success(f"CUDA version: {torch.version.cuda}")  # type: ignore
            print_success(f"cuDNN version: {torch.backends.cudnn.version()}")
            print_success(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print_success(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test CUDA computation
            try:
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
                print_success("CUDA computation test: PASSED")
            except Exception as e:
                print_error(f"CUDA computation test failed: {e}")
                return False
            
            return True
        else:
            print_error("CUDA is NOT available")
            print_info("Possible causes:", 4)
            print_info("1. GPU not enabled in BIOS", 6)
            print_info("2. NVIDIA drivers not installed", 6)
            print_info("3. CUDA Toolkit not installed", 6)
            print_info("4. PyTorch not compiled with CUDA support", 6)
            return False
            
    except ImportError:
        print_error("PyTorch not installed")
        print_info("Action: pip install torch torchvision", 4)
        return False
    except Exception as e:
        print_error(f"PyTorch check failed: {e}")
        return False


def check_cuda_toolkit():
    """Check CUDA Toolkit installation"""
    print_header("4. CUDA TOOLKIT")
    
    try:
        # Check for nvcc (CUDA compiler)
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Extract version from output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print_success(f"CUDA Toolkit: {line.strip()}")
                    return True
        
        print_warning("CUDA Toolkit not found (nvcc not in PATH)")
        print_info("This is optional but recommended for best performance", 4)
        print_info("Download: https://developer.nvidia.com/cuda-12-1-0-download-archive", 4)
        return False
        
    except FileNotFoundError:
        print_warning("CUDA Toolkit not installed")
        print_info("Optional: Install for development and debugging", 4)
        return False
    except Exception as e:
        print_warning(f"CUDA Toolkit check failed: {e}")
        return False


def check_tensorcore_support():
    """Check TensorCore (TF32) support"""
    print_header("5. TENSORCORE SUPPORT")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print_warning("Skipped (CUDA not available)")
            return False
        
        # Check compute capability
        capability = torch.cuda.get_device_capability(0)
        compute_capability = float(f"{capability[0]}.{capability[1]}")
        
        print_info(f"Compute Capability: {compute_capability}")
        
        # TF32 requires compute capability >= 8.0 (Ampere+)
        # RTX 2050 is Ampere architecture (8.6)
        if compute_capability >= 8.0:
            print_success("TensorCore (TF32) supported!")
            print_info("This enables automatic mixed precision for 2-3x speedup", 4)
            return True
        else:
            print_warning("TensorCore (TF32) not supported on this GPU")
            print_info("GPU will still work, but without TF32 acceleration", 4)
            return False
            
    except Exception as e:
        print_warning(f"TensorCore check failed: {e}")
        return False


def run_performance_test():
    """Run quick performance test"""
    print_header("6. PERFORMANCE TEST")
    
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print_warning("Skipped (CUDA not available)")
            return False
        
        # Test matrix multiplication performance
        size = 2048
        device = 'cuda'
        
        print_info(f"Testing {size}x{size} matrix multiplication...")
        
        # Warmup
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        for _ in range(3):
            _ = torch.matmul(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = torch.matmul(x, y)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        
        avg_time_ms = sum(times) / len(times) * 1000
        gflops = (2 * size**3) / (avg_time_ms / 1000) / 1e9
        
        print_success(f"Average time: {avg_time_ms:.2f} ms")
        print_success(f"Performance: {gflops:.1f} GFLOPS")
        
        # Compare with CPU
        print_info("Comparing with CPU...")
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        t0 = time.time()
        _ = torch.matmul(x_cpu, y_cpu)
        cpu_time_ms = (time.time() - t0) * 1000
        
        speedup = cpu_time_ms / avg_time_ms
        print_success(f"GPU Speedup: {speedup:.1f}x faster than CPU")
        
        return True
        
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False


def print_summary(results):
    """Print summary and next steps"""
    print_header("SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"Tests Passed: {passed}/{total}\n")
    
    for test, result in results.items():
        icon = "[OK]" if result else "[X]"
        color = GREEN if result else RED
        print(f"{color}{icon} {test}{RESET}")
    
    print(f"\n{BLUE}{'='*70}{RESET}\n")
    
    if all(results.values()):
        print(f"{GREEN}{BOLD}*** ALL CHECKS PASSED - GPU READY! ***{RESET}\n")
        print("Next steps:")
        print("  1. Run baseline validation:")
        print("     python phase2_validation.py")
        print()
        print("  2. Run performance benchmark:")
        print("     python benchmarks/week1_performance_test.py")
        print()
        print("  3. Start optimized pipeline:")
        print("     python app.py --phase2 --gpu")
        
    elif results.get('PyTorch CUDA', False):
        print(f"{YELLOW}{BOLD}[!] GPU PARTIALLY CONFIGURED{RESET}\n")
        print("Your GPU is working, but some optimizations are missing.")
        print("You can proceed with GPU acceleration, but consider:")
        
        if not results.get('CUDA Toolkit', True):
            print("  • Installing CUDA Toolkit for development tools")
        if not results.get('TensorCore Support', True):
            print("  • Your GPU may not support TF32 (this is OK)")
        
    else:
        print(f"{RED}{BOLD}[X] GPU NOT CONFIGURED{RESET}\n")
        print("Required actions:")
        
        if not results.get('Hardware Detection', False):
            print("  1. Enable discrete GPU in BIOS")
            print("     → See: docs/GPU_BIOS_SETUP_GUIDE.md")
        
        if not results.get('NVIDIA Driver', False):
            print("  2. Install NVIDIA drivers")
            print("     → https://www.nvidia.com/download/index.aspx")
        
        if not results.get('PyTorch CUDA', False):
            print("  3. Install CUDA Toolkit")
            print("     → https://developer.nvidia.com/cuda-12-1-0-download-archive")
        
        print("\nAfter fixing, re-run: python verify_gpu_setup.py")
    
    print(f"\n{BLUE}{'='*70}{RESET}\n")


def main():
    """Main verification flow"""
    print(f"\n{BOLD}GPU Setup Verification - AR Mirror Project{RESET}")
    print("This script will verify your GPU configuration for maximum performance\n")
    
    results = {}
    
    # Run all checks
    results['Hardware Detection'] = check_gpu_hardware()
    results['NVIDIA Driver'] = check_nvidia_driver()
    results['PyTorch CUDA'] = check_pytorch_cuda()
    results['CUDA Toolkit'] = check_cuda_toolkit()
    results['TensorCore Support'] = check_tensorcore_support()
    results['Performance Test'] = run_performance_test()
    
    # Print summary
    print_summary(results)
    
    # Return exit code
    return 0 if results['PyTorch CUDA'] else 1


if __name__ == "__main__":
    sys.exit(main())
