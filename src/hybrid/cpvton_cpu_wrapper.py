#!/usr/bin/env python3
"""
CP-VTON Model Wrapper with CPU Support
Wraps the original CP-VTON networks to support CPU execution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cp-vton"))

import torch
import torch.nn as nn


def patch_cpvton_for_cpu():
    """Patch CP-VTON networks to support CPU execution"""
    import networks
    
    # Save original classes
    OriginalFeatureRegression = networks.FeatureRegression
    OriginalTpsGridGen = networks.TpsGridGen
    OriginalGMM = networks.GMM
    
    # Patched FeatureRegression
    class CPUFeatureRegression(OriginalFeatureRegression):
        def __init__(self, input_nc=512, output_dim=6, use_cuda=None):
            # Force use_cuda based on actual availability
            actual_use_cuda = torch.cuda.is_available() if use_cuda is None else (use_cuda and torch.cuda.is_available())
            super().__init__(input_nc, output_dim, actual_use_cuda)
    
    # Patched TpsGridGen
    class CPUTpsGridGen(OriginalTpsGridGen):
        def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=None):
            # Force use_cuda based on actual availability
            actual_use_cuda = torch.cuda.is_available() if use_cuda is None else (use_cuda and torch.cuda.is_available())
            super().__init__(out_h, out_w, use_regular_grid, grid_size, reg_factor, actual_use_cuda)
    
    # Patched GMM
    class CPUGMM(OriginalGMM):
        def __init__(self, opt):
            # Temporarily patch the classes
            networks.FeatureRegression = CPUFeatureRegression
            networks.TpsGridGen = CPUTpsGridGen
            super().__init__(opt)
            # Restore original classes
            networks.FeatureRegression = OriginalFeatureRegression
            networks.TpsGridGen = OriginalTpsGridGen
    
    return CPUGMM


def load_gmm_cpu_safe(checkpoint_path, device='cpu'):
    """
    Load GMM model with CPU support
    
    Args:
        checkpoint_path: Path to GMM checkpoint
        device: 'cpu' or 'cuda'
        
    Returns:
        Loaded GMM model
    """
    class Args:
        fine_height = 256
        fine_width = 192
        grid_size = 5
    
    opt = Args()
    
    # Get CPU-safe GMM class
    CPUGMM = patch_cpvton_for_cpu()
    
    # Create model
    model = CPUGMM(opt)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    return model


# Test
if __name__ == "__main__":
    print("Testing CPU-safe GMM loading...")
    
    try:
        model = load_gmm_cpu_safe("cp-vton/checkpoints/gmm_train_new/gmm_final.pth", device='cpu')
        print("✓ GMM loaded successfully on CPU!")
        
        # Test forward pass
        import torch
        agnostic = torch.randn(1, 22, 256, 192)
        cloth_mask = torch.randn(1, 1, 256, 192)
        
        with torch.no_grad():
            grid, theta = model(agnostic, cloth_mask)
        
        print(f"✓ Forward pass successful!")
        print(f"  Grid shape: {grid.shape}")
        print(f"  Theta shape: {theta.shape}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
