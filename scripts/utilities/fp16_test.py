#!/usr/bin/env python3
"""
C2: FP16 Conversion Test

Compares GMM quality and performance between FP32 and FP16.
Measures: checkpoint size, memory, latency, and visual quality (L1 diff).

Usage:
    python scripts/utilities/fp16_test.py
"""

import sys
import os
import json
import time
import gc
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register CUDA DLLs
_nvidia_dir = PROJECT_ROOT / "ar" / "Lib" / "site-packages" / "nvidia"
if _nvidia_dir.exists():
    for _bin in _nvidia_dir.glob("*/bin"):
        if _bin.is_dir() and str(_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")


def main():
    import torch
    import torch.nn.functional as F
    
    sys.path.insert(0, str(PROJECT_ROOT / "cp-vton"))
    from networks import GMM
    
    print("=" * 70)
    print("FP16 CONVERSION TEST — GMM Model")
    print("=" * 70)
    
    ckpt_path = PROJECT_ROOT / "cp-vton" / "checkpoints" / "gmm_train_new" / "gmm_final.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    
    class Args:
        fine_height = 256
        fine_width = 192
        grid_size = 5
    
    # --- Load FP32 model ---
    print("\n[1/4] Loading FP32 model...")
    model_fp32 = GMM(Args())
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_fp32.load_state_dict(ckpt)
    model_fp32.to(device).eval()
    for p in model_fp32.parameters():
        p.requires_grad = False
    
    fp32_param_bytes = sum(p.numel() * p.element_size() for p in model_fp32.parameters())
    print(f"  FP32 param memory: {fp32_param_bytes / 1024 / 1024:.1f} MB")
    
    # --- Load FP16 model ---
    print("[2/4] Loading FP16 model...")
    model_fp16 = GMM(Args())
    model_fp16.load_state_dict(ckpt)
    model_fp16.half().to(device).eval()
    for p in model_fp16.parameters():
        p.requires_grad = False
    
    fp16_param_bytes = sum(p.numel() * p.element_size() for p in model_fp16.parameters())
    print(f"  FP16 param memory: {fp16_param_bytes / 1024 / 1024:.1f} MB")
    print(f"  Size reduction: {(1 - fp16_param_bytes / fp32_param_bytes) * 100:.0f}%")
    
    # --- Save FP16 checkpoint to measure disk size ---
    fp16_ckpt_path = PROJECT_ROOT / "output" / "gmm_fp16.pth"
    fp16_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_fp16.state_dict(), fp16_ckpt_path)
    fp16_disk_mb = fp16_ckpt_path.stat().st_size / 1024 / 1024
    fp32_disk_mb = ckpt_path.stat().st_size / 1024 / 1024
    print(f"  Disk: FP32={fp32_disk_mb:.1f} MB → FP16={fp16_disk_mb:.1f} MB ({(1-fp16_disk_mb/fp32_disk_mb)*100:.0f}% reduction)")
    
    # --- Latency comparison ---
    # FP16 on CPU is very slow (no native FP16 ALUs) — keep runs low
    num_runs = 100 if device == 'cuda' else 5
    print(f"\n[3/4] Latency comparison ({num_runs} runs)...")
    if device == 'cpu':
        print("  NOTE: FP16 on CPU is emulated and slow. FP16 is a GPU-only optimization.")
    
    # Generate fixed test inputs
    agnostic = torch.randn(1, 22, 256, 192, device=device)
    mask = torch.randn(1, 1, 256, 192, device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model_fp32(agnostic, mask)
            model_fp16(agnostic.half(), mask.half())
    
    # FP32 latency
    fp32_times = []
    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            grid_fp32, theta_fp32 = model_fp32(agnostic, mask)
        if device == 'cuda':
            torch.cuda.synchronize()
        fp32_times.append(time.perf_counter() - t0)
    
    # FP16 latency
    agnostic_h = agnostic.half()
    mask_h = mask.half()
    fp16_times = []
    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            grid_fp16, theta_fp16 = model_fp16(agnostic_h, mask_h)
        if device == 'cuda':
            torch.cuda.synchronize()
        fp16_times.append(time.perf_counter() - t0)
    
    fp32_avg = np.mean(fp32_times) * 1000
    fp16_avg = np.mean(fp16_times) * 1000
    speedup = fp32_avg / fp16_avg if fp16_avg > 0 else 0
    
    print(f"  FP32 avg: {fp32_avg:.2f} ms")
    print(f"  FP16 avg: {fp16_avg:.2f} ms")
    print(f"  Speedup:  {speedup:.2f}×")
    
    # --- Quality comparison ---
    print("\n[4/4] Quality comparison...")
    
    # Run both on same input and compare outputs
    with torch.no_grad():
        grid_fp32, theta_fp32 = model_fp32(agnostic, mask)
        grid_fp16, theta_fp16 = model_fp16(agnostic_h, mask_h)
    
    # Compare grids (the actual warp field)
    grid_diff = (grid_fp32.float() - grid_fp16.float()).abs()
    grid_l1 = grid_diff.mean().item()
    grid_max = grid_diff.max().item()
    
    # Compare theta (affine params)
    theta_diff = (theta_fp32.float() - theta_fp16.float()).abs()
    theta_l1 = theta_diff.mean().item()
    theta_max = theta_diff.max().item()
    
    # Compare warped outputs (grid_sample requires float32 on CPU)
    cloth_t = torch.randn(1, 3, 256, 192, device=device)
    with torch.no_grad():
        warped_fp32 = F.grid_sample(cloth_t, grid_fp32, padding_mode='border', align_corners=True)
        # FP16 grid_sample on CPU needs float32 cast
        warped_fp16 = F.grid_sample(cloth_t, grid_fp16.float(), padding_mode='border', align_corners=True)
    
    output_diff = (warped_fp32.float() - warped_fp16.float()).abs()
    output_l1 = output_diff.mean().item()
    output_max = output_diff.max().item()
    
    # In 0-255 space for intuition
    output_l1_255 = output_l1 * 255
    
    print(f"  Grid L1: {grid_l1:.6f} (max: {grid_max:.6f})")
    print(f"  Theta L1: {theta_l1:.6f} (max: {theta_max:.6f})")
    print(f"  Output L1: {output_l1:.6f} ({output_l1_255:.2f}/255)")
    print(f"  Output max diff: {output_max:.6f} ({output_max*255:.2f}/255)")
    
    # --- Verdict ---
    quality_ok = output_l1_255 < 1.0  # Less than 1 intensity level difference
    size_ok = fp16_disk_mb < fp32_disk_mb * 0.65  # At least 35% reduction
    speed_ok = fp16_avg <= fp32_avg * 1.1  # No more than 10% slower
    
    all_pass = quality_ok and size_ok and speed_ok
    
    print(f"\n  Quality (L1 < 1.0/255): {'PASS' if quality_ok else 'FAIL'} ({output_l1_255:.2f})")
    print(f"  Size (≥35% reduction):  {'PASS' if size_ok else 'FAIL'} ({(1-fp16_disk_mb/fp32_disk_mb)*100:.0f}%)")
    print(f"  Speed (≤10% slower):    {'PASS' if speed_ok else 'FAIL'} ({speedup:.2f}×)")
    print(f"\n  VERDICT: {'✅ FP16 VIABLE' if all_pass else '❌ FP16 NOT VIABLE'}")
    
    # --- Report ---
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': device,
        'size': {
            'fp32_disk_mb': round(fp32_disk_mb, 1),
            'fp16_disk_mb': round(fp16_disk_mb, 1),
            'reduction_pct': round((1 - fp16_disk_mb / fp32_disk_mb) * 100, 1),
            'fp32_param_mb': round(fp32_param_bytes / 1024 / 1024, 1),
            'fp16_param_mb': round(fp16_param_bytes / 1024 / 1024, 1),
        },
        'latency': {
            'fp32_avg_ms': round(float(fp32_avg), 2),
            'fp16_avg_ms': round(float(fp16_avg), 2),
            'speedup': round(float(speedup), 2),
        },
        'quality': {
            'grid_l1': float(round(grid_l1, 6)),
            'theta_l1': float(round(theta_l1, 6)),
            'output_l1_0to1': float(round(output_l1, 6)),
            'output_l1_0to255': float(round(output_l1_255, 2)),
            'output_max_diff_0to255': float(round(output_max * 255, 2)),
        },
        'verdict': {
            'quality_pass': bool(quality_ok),
            'size_pass': bool(size_ok),
            'speed_pass': bool(speed_ok),
            'fp16_viable': bool(all_pass),
        }
    }
    
    output_path = PROJECT_ROOT / "output" / "fp16_test.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")
    
    # Cleanup FP16 checkpoint
    if fp16_ckpt_path.exists():
        os.remove(fp16_ckpt_path)


if __name__ == "__main__":
    main()
