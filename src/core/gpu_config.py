"""GPU optimization configuration for AR Mirror.

This module provides GPU acceleration and optimization settings
for maximum performance on NVIDIA GPUs.

Features:
- Automatic CUDA detection and configuration
- TF32 acceleration (8x faster on Ampere+ GPUs)
- cuDNN autotuning for optimal kernels
- Memory management and monitoring
- Graceful CPU fallback
"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class GPUConfig:
    """GPU configuration and optimization settings."""
    
    _initialized = False
    _device = None
    
    @staticmethod
    def get_device(prefer_gpu: bool = True) -> torch.device:
        """
        Get the optimal device for inference.
        
        Args:
            prefer_gpu: If True, use GPU if available. If False, force CPU.
        
        Returns:
            torch.device: Selected device (cuda or cpu)
        """
        if GPUConfig._device is not None:
            return GPUConfig._device
        
        if prefer_gpu and torch.cuda.is_available():
            GPUConfig._device = torch.device("cuda:0")
            logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            GPUConfig._device = torch.device("cpu")
            if prefer_gpu:
                logger.warning("GPU not available, using CPU")
            else:
                logger.info("Using CPU (GPU disabled by user)")
        
        return GPUConfig._device
    
    @staticmethod
    def enable_optimizations() -> bool:
        """
        Enable all GPU optimizations.
        
        Returns:
            bool: True if GPU optimizations enabled, False if using CPU
        """
        if GPUConfig._initialized:
            return GPUConfig._device.type == "cuda"
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU optimizations")
            GPUConfig._initialized = True
            return False
        
        try:
            # Get device
            device = GPUConfig.get_device()
            
            # Enable TF32 for Ampere+ GPUs (RTX 2050 has Ampere architecture)
            # TF32: ~8x faster than FP32, minimal accuracy loss (<0.1%)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✓ Enabled TF32 precision (8x speedup on Ampere GPUs)")
            
            # Enable cuDNN autotuner
            # Benchmarks different algorithms, picks fastest for your specific input sizes
            # ~10-20% speedup after warmup (first few iterations)
            torch.backends.cudnn.benchmark = True
            logger.info("✓ Enabled cuDNN autotuner")
            
            # Disable determinism for maximum speed
            # Note: Results may vary slightly between runs (non-deterministic)
            torch.backends.cudnn.deterministic = False
            
            # Set default CUDA device
            torch.cuda.set_device(0)
            
            # Log GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✓ GPU: {gpu_name} ({gpu_memory:.2f} GB)")
            
            # Log memory status
            GPUConfig.log_memory_status()
            
            logger.info("✅ GPU optimizations enabled")
            GPUConfig._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable GPU optimizations: {e}")
            logger.warning("Falling back to CPU")
            GPUConfig._device = torch.device("cpu")
            GPUConfig._initialized = True
            return False
    
    @staticmethod
    def log_memory_status():
        """Log current GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, "
                       f"{reserved:.2f}GB reserved, {total:.2f}GB total")
        except Exception as e:
            logger.warning(f"Failed to get GPU memory status: {e}")
    
    @staticmethod
    def preallocate_memory(size_gb: float = 2.0):
        """
        Pre-allocate GPU memory to avoid fragmentation.
        
        This creates a large allocation and then frees it, which helps
        CUDA allocator manage memory more efficiently for future allocations.
        
        Args:
            size_gb: Amount of memory to pre-allocate in GB
        """
        if not torch.cuda.is_available():
            return
        
        try:
            # Create dummy tensor to reserve memory
            dummy = torch.empty(
                int(size_gb * 1024**3 / 4),  # 4 bytes per float32
                dtype=torch.float32,
                device="cuda"
            )
            del dummy
            torch.cuda.empty_cache()
            
            logger.info(f"✓ Pre-allocated {size_gb}GB GPU memory")
        except Exception as e:
            logger.warning(f"Failed to preallocate memory: {e}")
    
    @staticmethod
    def get_optimal_batch_size(
        model_size_mb: float,
        input_size_mb: float = 10,
        max_batch_size: int = 16
    ) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            model_size_mb: Size of the model in MB
            input_size_mb: Size of one input sample in MB
            max_batch_size: Maximum batch size to try
        
        Returns:
            int: Recommended batch size (1 if using CPU)
        """
        if not torch.cuda.is_available():
            return 1
        
        try:
            # Get available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**2
            available_memory = total_memory - allocated_memory
            
            # Reserve 20% for safety margin
            usable_memory = available_memory * 0.8
            
            # Calculate batch size
            # Memory needed = model + (batch_size * input_size) + (batch_size * output_size)
            # Assume output_size ~= input_size
            memory_per_sample = input_size_mb * 2
            available_for_batch = usable_memory - model_size_mb
            
            batch_size = int(available_for_batch / memory_per_sample)
            batch_size = max(1, min(batch_size, max_batch_size))
            
            logger.info(f"Optimal batch size: {batch_size} "
                       f"(based on {usable_memory:.0f}MB available memory)")
            
            return batch_size
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return 4  # Safe default
    
    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Get comprehensive GPU information.
        
        Returns:
            dict: GPU information including device, memory, compute capability
        """
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device": str(GPUConfig.get_device()),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,  # type: ignore
                "cudnn_version": torch.backends.cudnn.version(),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            })
        
        return info
    
    @staticmethod
    def reset():
        """Reset GPU configuration (for testing)."""
        GPUConfig._initialized = False
        GPUConfig._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_gpu_info():
    """Print formatted GPU information."""
    info = GPUConfig.get_info()
    
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    
    if info["cuda_available"]:
        print(f"✓ Device: {info['device']}")
        print(f"✓ GPU: {info['gpu_name']}")
        print(f"✓ CUDA Version: {info['cuda_version']}")
        print(f"✓ cuDNN Version: {info['cudnn_version']}")
        print(f"✓ Memory: {info['memory_total_gb']:.2f} GB total")
        print(f"✓ Compute Capability: {info['compute_capability']}")
        print(f"✓ TF32 Enabled: {info['tf32_enabled']}")
        print(f"✓ cuDNN Benchmark: {info['cudnn_benchmark']}")
    else:
        print(f"⚠ Device: {info['device']}")
        print("⚠ GPU: Not available (using CPU)")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test GPU configuration
    logging.basicConfig(level=logging.INFO)
    
    print_gpu_info()
    
    if torch.cuda.is_available():
        print("\nEnabling optimizations...")
        GPUConfig.enable_optimizations()
        
        print("\nMemory status:")
        GPUConfig.log_memory_status()
        
        print(f"\nOptimal batch size: {GPUConfig.get_optimal_batch_size(72)}")
