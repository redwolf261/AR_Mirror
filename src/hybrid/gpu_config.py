"""
GPU Configuration for Hybrid Pipeline

Delegates to the canonical implementation in src.core.gpu_config.
This module exists for backward compatibility — all hybrid pipeline code
that imports from src.hybrid.gpu_config will continue to work.
"""

from src.core.gpu_config import GPUConfig
from src.hybrid.gpu_acceleration import GPUAccelerator

import torch
import logging

logger = logging.getLogger(__name__)


class GPUConfiguration:
    """
    Manages GPU configuration and optimization for the pipeline.
    
    Compatibility wrapper around src.core.gpu_config.GPUConfig.
    
    Features:
    - Automatic device selection (CUDA/Metal/CPU)
    - Mixed precision support (float16/float32)
    - Memory monitoring
    - Performance profiling
    """
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize GPU configuration
        
        Args:
            force_cpu: Force CPU mode even if GPU is available
        """
        self.accelerator = GPUAccelerator()
        self.force_cpu = force_cpu
        
        # Determine device
        if force_cpu or not self.accelerator.is_available():
            self.device = torch.device('cpu')
            self.dtype_inference = torch.float32
            self.dtype_compute = torch.float32
            self.gpu_enabled = False
        else:
            self.device = torch.device('cuda')
            self.dtype_inference = torch.float32
            self.dtype_compute = torch.float32
            self.gpu_enabled = True
        
        # Device info
        self.device_name = self.accelerator.device_name
        self.device_type = self.accelerator.device_type
        
        # Performance tracking
        self.benchmarks = {
            'segmentation_cpu': 40.9,
            'segmentation_gpu': 15.0,
            'shape_estimation_cpu': 7.3,
            'shape_estimation_gpu': 4.5,
            'garment_encoding_cpu': 8.2,
            'garment_encoding_gpu': 5.5,
        }

    def enable_optimizations(self) -> bool:
        """Enable GPU optimizations via the canonical GPUConfig."""
        return GPUConfig.enable_optimizations()

    def get_device(self) -> torch.device:
        """Get the configured device."""
        return self.device

    def get_status(self) -> dict:
        """Get GPU status information."""
        return {
            'device': str(self.device),
            'device_name': self.device_name,
            'device_type': self.device_type,
            'gpu_enabled': self.gpu_enabled,
            'force_cpu': self.force_cpu,
        }


__all__ = ['GPUConfig', 'GPUConfiguration']
