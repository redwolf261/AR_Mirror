#!/usr/bin/env python3
"""
C1: Model Memory Profiler

Measures actual RSS consumed by each model in the AR Mirror pipeline.
Simulates a 1GB mobile RAM budget and reports what fits.

Usage:
    python scripts/utilities/memory_profiler.py
"""

import sys
import os
import json
import time
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register CUDA DLLs
_nvidia_dir = PROJECT_ROOT / "ar" / "Lib" / "site-packages" / "nvidia"
if _nvidia_dir.exists():
    for _bin in _nvidia_dir.glob("*/bin"):
        if _bin.is_dir() and str(_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")


def get_rss_mb():
    """Get current process RSS in MB (cross-platform)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    # Windows fallback using os module
    try:
        import subprocess
        pid = os.getpid()
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True, timeout=5
        )
        # Parse: "python.exe","1234","Console","1","123,456 K"
        line = result.stdout.strip()
        if line:
            parts = line.split('"')
            # Memory is the last quoted field, e.g. '123,456 K'
            mem_str = parts[-2].replace(',', '').replace(' K', '').strip()
            return int(mem_str) / 1024  # KB to MB
    except Exception:
        pass
    return 0.0


def profile_model(name: str, load_fn, inference_fn=None, num_runs=10):
    """Profile a single model's memory footprint.
    
    Returns dict with baseline_mb, loaded_mb, delta_mb, peak_inference_mb.
    """
    gc.collect()
    time.sleep(0.5)
    baseline = get_rss_mb()
    
    # Load
    t0 = time.time()
    obj = load_fn()
    load_time = time.time() - t0
    gc.collect()
    time.sleep(0.3)
    loaded = get_rss_mb()
    delta = loaded - baseline
    
    # Inference (optional)
    peak_inference = loaded
    avg_latency = 0
    if inference_fn and obj is not None:
        latencies = []
        for _ in range(num_runs):
            t0 = time.time()
            inference_fn(obj)
            latencies.append(time.time() - t0)
            current = get_rss_mb()
            peak_inference = max(peak_inference, current)
        avg_latency = sum(latencies) / len(latencies) * 1000  # ms
    
    # Cleanup
    del obj
    gc.collect()
    time.sleep(0.3)
    after_cleanup = get_rss_mb()
    
    return {
        'name': name,
        'baseline_mb': round(baseline, 1),
        'loaded_mb': round(loaded, 1),
        'delta_mb': round(delta, 1),
        'peak_inference_mb': round(peak_inference, 1),
        'load_time_s': round(load_time, 2),
        'avg_inference_ms': round(avg_latency, 1),
        'after_cleanup_mb': round(after_cleanup, 1),
    }


def main():
    import numpy as np
    
    print("=" * 70)
    print("MEMORY PROFILER — AR Mirror Models")
    print("=" * 70)
    
    results = []
    baseline_rss = get_rss_mb()
    print(f"\nBaseline process RSS: {baseline_rss:.1f} MB\n")
    
    # --- 1. Pose Landmarker ---
    print("[1/4] Profiling Pose Landmarker...")
    try:
        def load_pose():
            from src.core.body_aware_fitter import BodyAwareGarmentFitter
            return BodyAwareGarmentFitter()
        
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        def infer_pose(fitter):
            fitter.extract_body_measurements(dummy_frame)
        
        r = profile_model("PoseLandmarker + BodyFitter", load_pose, infer_pose, num_runs=10)
        results.append(r)
        print(f"      Delta: {r['delta_mb']:.1f} MB | Inference: {r['avg_inference_ms']:.1f} ms")
    except Exception as e:
        print(f"      ERROR: {e}")
        results.append({'name': 'PoseLandmarker', 'error': str(e)})
    
    # --- 2. GMM Model ---
    print("[2/4] Profiling GMM Model...")
    try:
        def load_gmm():
            import torch
            sys.path.insert(0, str(PROJECT_ROOT / "cp-vton"))
            from networks import GMM
            class Args:
                fine_height = 256
                fine_width = 192
                grid_size = 5
            model = GMM(Args())
            ckpt = torch.load(
                PROJECT_ROOT / "cp-vton" / "checkpoints" / "gmm_train_new" / "gmm_final.pth",
                map_location='cpu', weights_only=False
            )
            model.load_state_dict(ckpt)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model
        
        def infer_gmm(model):
            import torch
            with torch.no_grad():
                agnostic = torch.randn(1, 22, 256, 192)
                cloth_mask = torch.randn(1, 1, 256, 192)
                model(agnostic, cloth_mask)
        
        r = profile_model("GMM (FP32)", load_gmm, infer_gmm, num_runs=10)
        results.append(r)
        print(f"      Delta: {r['delta_mb']:.1f} MB | Inference: {r['avg_inference_ms']:.1f} ms")
    except Exception as e:
        print(f"      ERROR: {e}")
        results.append({'name': 'GMM', 'error': str(e)})
    
    # --- 3. GMM FP16 ---
    print("[3/4] Profiling GMM (FP16)...")
    try:
        def load_gmm_fp16():
            import torch
            sys.path.insert(0, str(PROJECT_ROOT / "cp-vton"))
            from networks import GMM
            class Args:
                fine_height = 256
                fine_width = 192
                grid_size = 5
            model = GMM(Args())
            ckpt = torch.load(
                PROJECT_ROOT / "cp-vton" / "checkpoints" / "gmm_train_new" / "gmm_final.pth",
                map_location='cpu', weights_only=False
            )
            model.load_state_dict(ckpt)
            model.half()  # FP16
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model
        
        def infer_gmm_fp16(model):
            import torch
            with torch.no_grad():
                agnostic = torch.randn(1, 22, 256, 192).half()
                cloth_mask = torch.randn(1, 1, 256, 192).half()
                model(agnostic, cloth_mask)
        
        r = profile_model("GMM (FP16)", load_gmm_fp16, infer_gmm_fp16, num_runs=10)
        results.append(r)
        print(f"      Delta: {r['delta_mb']:.1f} MB | Inference: {r['avg_inference_ms']:.1f} ms")
    except Exception as e:
        print(f"      ERROR: {e}")
        results.append({'name': 'GMM FP16', 'error': str(e)})
    
    # --- 4. LiveBodySegmenter ---
    print("[4/4] Profiling Body Segmenter...")
    try:
        def load_segmenter():
            from src.core.live_pose_converter import LiveBodySegmenter
            return LiveBodySegmenter()
        
        dummy_rgb = np.random.rand(256, 192, 3).astype(np.float32)
        def infer_seg(seg):
            seg.segment(dummy_rgb)
        
        r = profile_model("LiveBodySegmenter", load_segmenter, infer_seg, num_runs=10)
        results.append(r)
        print(f"      Delta: {r['delta_mb']:.1f} MB | Inference: {r['avg_inference_ms']:.1f} ms")
    except Exception as e:
        print(f"      ERROR: {e}")
        results.append({'name': 'LiveBodySegmenter', 'error': str(e)})
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    total_delta = sum(r.get('delta_mb', 0) for r in results if 'error' not in r)
    mobile_budget = 1024  # 1 GB
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'baseline_rss_mb': round(baseline_rss, 1),
        'models': results,
        'summary': {
            'total_model_delta_mb': round(total_delta, 1),
            'mobile_budget_mb': mobile_budget,
            'fits_in_budget': total_delta < mobile_budget,
            'headroom_mb': round(mobile_budget - total_delta, 1),
        }
    }
    
    print(json.dumps(report, indent=2))
    
    output_path = PROJECT_ROOT / "output" / "memory_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
