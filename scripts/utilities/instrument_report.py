#!/usr/bin/env python3
"""
Instrumentation Report — Headless stability profiler

Runs the AR Mirror pipeline for N frames, collects all instrumentation data,
and writes a JSON report to disk. Also measures FPS overhead from instrumentation.

Usage:
    python scripts/utilities/instrument_report.py --frames 300
    python scripts/utilities/instrument_report.py --frames 300 --output output/stability_report.json
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-register NVIDIA CUDA DLLs
_nvidia_dir = PROJECT_ROOT / "ar" / "Lib" / "site-packages" / "nvidia"
if _nvidia_dir.exists():
    for _bin in _nvidia_dir.glob("*/bin"):
        if _bin.is_dir() and str(_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

import cv2


def run_report(num_frames: int, output_path: str, phase: int = 2):
    """Run the pipeline for num_frames and collect instrumentation data.
    
    Args:
        num_frames: Number of frames to capture and analyze.
        output_path: Path to write the JSON report.
        phase: Pipeline phase (0, 1, 2).
    """
    print("=" * 70)
    print(f"INSTRUMENTATION REPORT — {num_frames} frames")
    print("=" * 70)
    
    # --- Initialize components ---
    print("\n[1/3] Initializing pipeline...")
    
    # Body-aware fitter (contains landmark logger)
    body_fitter = None
    try:
        from src.core.body_aware_fitter import BodyAwareGarmentFitter
        body_fitter = BodyAwareGarmentFitter()
        print("      [OK] Body-aware fitter + landmark logger")
    except Exception as e:
        print(f"      [WARN] Body fitter not available: {e}")
    
    # Phase 2 pipeline (contains transform logger + GPD)
    pipeline = None
    if phase == 2:
        try:
            from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
            pipeline = Phase2NeuralPipeline(
                device='auto', enable_tom=False,
                batch_size=1, enable_optimizations=True
            )
            print("      [OK] Phase 2 pipeline + transform logger + GPD")
        except Exception as e:
            print(f"      [WARN] Phase 2 not available: {e}")
    
    # --- Open camera ---
    print("\n[2/3] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("      [ERROR] Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      [OK] Camera at {w}x{h}")
    
    # --- Load a garment for warping (if dataset available) ---
    cloth_rgb, cloth_mask = None, None
    from src.app.rendering import load_viton_cloth
    pairs_file = PROJECT_ROOT / "dataset" / "train_pairs.txt"
    if pairs_file.exists():
        with open(pairs_file, 'r') as f:
            line = f.readline().strip().split()
            if len(line) >= 2:
                dataset_root = str(PROJECT_ROOT / "dataset" / "train")
                cloth_rgb, cloth_mask = load_viton_cloth(dataset_root, line[1])
                if cloth_rgb is not None:
                    print(f"      [OK] Loaded garment: {line[1]}")
    if cloth_rgb is None:
        print("      [WARN] No garment loaded — transform/GPD will be empty")
    
    # --- Run ---
    print(f"\n[3/3] Running {num_frames} frames...\n")
    frame_times = deque(maxlen=num_frames)
    start = time.time()
    
    for i in range(num_frames):
        t0 = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print(f"      Camera lost at frame {i}")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Pose detection (populates landmark logger)
        measurements = None
        if body_fitter:
            measurements = body_fitter.extract_body_measurements(frame)
        
        # Neural warp (populates transform logger + GPD)
        if pipeline and measurements and cloth_rgb is not None:
            try:
                h_f, w_f = frame.shape[:2]
                landmarks = measurements['landmarks']
                mp_dict = {}
                for idx in range(len(landmarks)):
                    lm = landmarks[idx]
                    mp_dict[idx] = {'x': lm.x, 'y': lm.y, 'visibility': lm.visibility}
                
                person_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # Feed landmark displacement into GPD
                lm_disp = body_fitter.landmark_logger.get_last_displacement()
                pipeline.gpd_metric._static_threshold = 3.0  # align thresholds
                
                result = pipeline.warp_garment(
                    person_rgb, cloth_rgb, cloth_mask,
                    mp_dict, body_mask=measurements.get('body_mask')
                )
                # GPD is updated inside warp_garment() — no double-count
            except Exception as e:
                if i == 0:
                    print(f"      [WARN] Warp error: {e}")
        
        frame_time = time.time() - t0
        frame_times.append(frame_time)
        
        # Progress bar
        if (i + 1) % 10 == 0 or i == num_frames - 1:
            pct = int((i + 1) / num_frames * 40)
            bar = "[" + "=" * pct + " " * (40 - pct) + "]"
            fps_now = 1.0 / np.mean(list(frame_times)[-30:]) if frame_times else 0
            print(f"\r  {bar} {i+1}/{num_frames} | FPS: {fps_now:.1f}", end="", flush=True)
    
    total_time = time.time() - start
    actual_frames = len(frame_times)
    cap.release()
    
    # --- Collect all stats ---
    print(f"\n\nCollecting metrics from {actual_frames} frames...\n")
    
    report = {
        'run_info': {
            'frames': actual_frames,
            'duration_s': round(total_time, 2),
            'resolution': f"{w}x{h}",
            'phase': phase,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'fps': {
            'avg_fps': round(actual_frames / total_time, 1) if total_time > 0 else 0,
            'avg_latency_ms': round(float(np.mean(list(frame_times))) * 1000, 1) if frame_times else 0,
            'p95_latency_ms': round(float(np.percentile(list(frame_times), 95)) * 1000, 1) if len(frame_times) > 5 else 0,
        },
    }
    
    # Landmark stability
    if body_fitter:
        report['landmark_stability'] = body_fitter.landmark_logger.get_stats()
    else:
        report['landmark_stability'] = {'error': 'body_fitter not available'}
    
    # Transform + GPD
    if pipeline:
        report['gmm_transform'] = pipeline.transform_logger.get_stats()
        report['gpd'] = pipeline.gpd_metric.get_stats()
    else:
        report['gmm_transform'] = {'error': 'pipeline not available'}
        report['gpd'] = {'error': 'pipeline not available'}
    
    # --- Write report ---
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # --- Print summary ---
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='AR Mirror — Stability Instrumentation Report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--frames', type=int, default=300,
                        help='Number of frames to capture (default: 300)')
    parser.add_argument('--output', type=str, default='output/stability_report.json',
                        help='Output path for the JSON report')
    parser.add_argument('--phase', type=int, default=2, choices=[0, 1, 2],
                        help='Pipeline phase (default: 2)')
    
    args = parser.parse_args()
    run_report(args.frames, args.output, args.phase)


if __name__ == "__main__":
    main()
