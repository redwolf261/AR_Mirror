#!/usr/bin/env python3
"""Validate VITON loader and show what it's loading"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app import load_viton_cloth
import numpy as np

print("\n" + "="*80)
print("VITON LOADER VALIDATION")
print("="*80 + "\n")

# Load first VITON pair from dataset
dataset_root = "dataset/train"
test_filename = "00000_00.jpg"

print(f"[1] Testing load_viton_cloth('{dataset_root}', '{test_filename}')")
print()

cloth_path = os.path.join(dataset_root, "cloth", test_filename)
mask_path = os.path.join(dataset_root, "cloth-mask", test_filename)

print(f"  Cloth image:  {cloth_path}")
print(f"    ├─ Exists: {'✓' if os.path.exists(cloth_path) else '✗'}")
if os.path.exists(cloth_path):
    print(f"    └─ Size: {os.path.getsize(cloth_path) / 1024:.1f} KB")

print(f"  Cloth mask:   {mask_path}")
print(f"    ├─ Exists: {'✓' if os.path.exists(mask_path) else '✗'}")
if os.path.exists(mask_path):
    print(f"    └─ Size: {os.path.getsize(mask_path) / 1024:.1f} KB")

print()
print("[2] Loading with proper VITON loader...")

cloth_rgb, cloth_mask = load_viton_cloth(dataset_root, test_filename)

if cloth_rgb is None or cloth_mask is None:
    print("  ✗ FAILED to load")
    sys.exit(1)

print()
print("  ✓ LOADED SUCCESSFULLY")
print()
print(f"  Cloth RGB:")
print(f"    ├─ Shape: {cloth_rgb.shape}")
print(f"    ├─ Dtype: {cloth_rgb.dtype}")
print(f"    ├─ Range: [{cloth_rgb.min():.3f}, {cloth_rgb.max():.3f}]")
print(f"    └─ Type: float32 [0,1] ✓")
print()
print(f"  Cloth Mask:")
print(f"    ├─ Shape: {cloth_mask.shape}")
print(f"    ├─ Dtype: {cloth_mask.dtype}")
print(f"    ├─ Unique values: {np.unique(cloth_mask)}")
print(f"    ├─ Foreground pixels: {(cloth_mask > 0).sum()} ({100*(cloth_mask > 0).sum() / cloth_mask.size:.1f}%)")
print(f"    └─ Type: binary float32 [0,1] ✓")
print()
print("[3] Validation Assertions")
print()

try:
    assert cloth_rgb.ndim == 3 and cloth_rgb.shape[2] == 3
    print("  ✓ cloth_rgb has 3 color channels")
    
    assert cloth_mask.ndim == 3 and cloth_mask.shape[2] == 1
    print("  ✓ cloth_mask has 1 channel")
    
    assert cloth_rgb.dtype == np.float32
    print("  ✓ cloth_rgb is float32")
    
    assert cloth_mask.dtype == np.float32
    print("  ✓ cloth_mask is float32")
    
    assert 0.0 <= cloth_rgb.min() and cloth_rgb.max() <= 1.0
    print("  ✓ cloth_rgb in range [0,1]")
    
    assert set(np.unique(cloth_mask)).issubset({0.0, 1.0})
    print("  ✓ cloth_mask is binary (only 0.0 and 1.0)")
    
    print()
    print("="*80)
    print("✅ ALL VITON LOADER VALIDATIONS PASSED")
    print("="*80)
    print()
    print("Alpha compositing formula ready:")
    print("  composite = cloth_rgb * cloth_mask + background_rgb * (1 - cloth_mask)")
    print()
    
except AssertionError as e:
    print(f"\n✗ ASSERTION FAILED: {e}")
    sys.exit(1)
