#!/usr/bin/env python3
"""
D2: Continuous Run Profiler — 10-minute endurance test.

Monitors:
  - Memory growth (RSS at 30s intervals)
  - FPS stability (rolling 30s windows)
  - Transform drift (θ variance trend)
  - Crash count per 1000 frames
  - GPD trend over time

Usage:
    python scripts/utilities/continuous_run.py --duration 600
    python scripts/utilities/continuous_run.py --duration 60   # 1-minute quick test
"""

import sys
import os
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_nvidia_dir = PROJECT_ROOT / "ar" / "Lib" / "site-packages" / "nvidia"
if _nvidia_dir.exists():
    for _bin in _nvidia_dir.glob("*/bin"):
        if _bin.is_dir() and str(_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

import cv2


def get_rss_mb():
    """Get process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        pid = os.getpid()
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True, timeout=5
        )
        line = result.stdout.strip()
        if line:
            parts = line.split('"')
            mem_str = parts[-2].replace(',', '').replace(' K', '').strip()
            return int(mem_str) / 1024
    except Exception:
        pass
    return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='D2: Continuous Run Profiler')
    parser.add_argument('--duration', type=int, default=600,
                        help='Duration in seconds (default: 600 = 10 minutes)')
    parser.add_argument('--output', type=str, default='output/continuous_run.json')
    args = parser.parse_args()
    
    duration = args.duration
    
    print("=" * 70)
    print(f"CONTINUOUS RUN — {duration}s ({duration/60:.0f} min)")
    print("=" * 70)
    
    # Initialize
    print("\n[1/3] Initializing...")
    from src.core.body_aware_fitter import BodyAwareGarmentFitter
    body_fitter = BodyAwareGarmentFitter()
    print("      [OK] Body fitter")
    
    pipeline = None
    try:
        from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
        pipeline = Phase2NeuralPipeline(
            device='auto', enable_tom=False,
            batch_size=1, enable_optimizations=True
        )
        print("      [OK] Neural pipeline")
    except Exception as e:
        print(f"      [WARN] Pipeline not available: {e}")
    
    # Load garment
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
                    print(f"      [OK] Garment: {line[1]}")
    
    # Camera
    print("\n[2/3] Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      [OK] Camera at {w}x{h}")
    
    # Run
    print(f"\n[3/3] Running for {duration}s...\n")
    
    # Tracking
    rss_samples = []         # (timestamp_s, rss_mb)
    fps_windows = []         # (timestamp_s, fps)
    crash_log = []           # (frame_idx, error)
    frame_count = 0
    total_crashes = 0
    
    rss_interval = 30        # Sample RSS every 30s
    fps_window = deque(maxlen=90)  # ~3s at 30 FPS
    
    start_time = time.time()
    last_rss_time = start_time
    initial_rss = get_rss_mb()
    rss_samples.append((0, round(initial_rss, 1)))
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break
        
        t0 = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("  Camera lost!")
            break
        
        frame = cv2.flip(frame, 1)
        
        try:
            measurements = body_fitter.extract_body_measurements(frame)
            
            if pipeline and measurements and cloth_rgb is not None:
                landmarks = measurements['landmarks']
                mp_dict = {}
                for idx in range(len(landmarks)):
                    lm = landmarks[idx]
                    mp_dict[idx] = {'x': lm.x, 'y': lm.y, 'visibility': lm.visibility}
                
                person_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                result = pipeline.warp_garment(
                    person_rgb, cloth_rgb, cloth_mask,
                    mp_dict, body_mask=measurements.get('body_mask')
                )
        except Exception as e:
            total_crashes += 1
            if len(crash_log) < 20:
                crash_log.append((frame_count, str(e)[:200]))
        
        frame_time = time.time() - t0
        fps_window.append(frame_time)
        frame_count += 1
        
        # FPS sample (every 30 frames)
        if frame_count % 30 == 0:
            current_fps = 1.0 / np.mean(list(fps_window)) if fps_window else 0
            fps_windows.append((round(elapsed, 1), round(float(current_fps), 1)))
        
        # RSS sample
        now = time.time()
        if now - last_rss_time >= rss_interval:
            rss = get_rss_mb()
            rss_samples.append((round(elapsed, 1), round(rss, 1)))
            last_rss_time = now
        
        # Progress
        if frame_count % 100 == 0:
            pct = int(elapsed / duration * 40)
            bar = "[" + "=" * pct + " " * (40 - pct) + "]"
            current_fps = 1.0 / np.mean(list(fps_window)) if fps_window else 0
            rss = rss_samples[-1][1] if rss_samples else 0
            print(f"\r  {bar} {elapsed:.0f}s/{duration}s | "
                  f"FPS: {current_fps:.1f} | RSS: {rss:.0f}MB | "
                  f"frames: {frame_count} | crashes: {total_crashes}",
                  end="", flush=True)
    
    # Final RSS
    final_rss = get_rss_mb()
    rss_samples.append((round(time.time() - start_time, 1), round(final_rss, 1)))
    
    cap.release()
    total_time = time.time() - start_time
    
    # Collect instrumentation stats
    print(f"\n\nCollecting final metrics...")
    
    landmark_stats = body_fitter.landmark_logger.get_stats()
    transform_stats = pipeline.transform_logger.get_stats() if pipeline else {}
    gpd_stats = pipeline.gpd_metric.get_stats() if pipeline else {}
    
    # Compute memory growth
    if len(rss_samples) >= 2:
        memory_growth = rss_samples[-1][1] - rss_samples[0][1]
    else:
        memory_growth = 0
    
    # FPS stability
    fps_values = [f[1] for f in fps_windows]
    fps_stability = {
        'min': round(float(min(fps_values)), 1) if fps_values else 0,
        'max': round(float(max(fps_values)), 1) if fps_values else 0,
        'mean': round(float(np.mean(fps_values)), 1) if fps_values else 0,
        'std': round(float(np.std(fps_values)), 2) if fps_values else 0,
        'p5': round(float(np.percentile(fps_values, 5)), 1) if len(fps_values) > 2 else 0,
    }
    
    # Build report
    report = {
        'run_info': {
            'target_duration_s': duration,
            'actual_duration_s': round(total_time, 1),
            'total_frames': frame_count,
            'resolution': f'{w}x{h}',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'stability': {
            'total_crashes': total_crashes,
            'crashes_per_1000': round(total_crashes / max(frame_count, 1) * 1000, 2),
            'crash_free': total_crashes == 0,
        },
        'memory': {
            'initial_rss_mb': rss_samples[0][1] if rss_samples else 0,
            'final_rss_mb': rss_samples[-1][1] if rss_samples else 0,
            'growth_mb': round(memory_growth, 1),
            'growth_per_minute': round(memory_growth / (total_time / 60), 2) if total_time > 0 else 0,
            'timeline': rss_samples,
        },
        'fps': fps_stability,
        'fps_timeline': fps_windows,
        'landmark_stability': landmark_stats,
        'gmm_transform': transform_stats,
        'gpd': gpd_stats,
        'crash_log': crash_log[:10],
    }
    
    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("CONTINUOUS RUN RESULTS")
    print("=" * 70)
    print(f"  Duration:       {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Frames:         {frame_count}")
    print(f"  Crashes:        {total_crashes} ({report['stability']['crashes_per_1000']}/1000)")
    print(f"  Memory growth:  {memory_growth:+.1f} MB ({report['memory']['growth_per_minute']:+.1f} MB/min)")
    print(f"  FPS:            {fps_stability['mean']:.1f} avg ± {fps_stability['std']:.1f} "
          f"(min={fps_stability['min']}, p5={fps_stability['p5']})")
    print(f"  GPD (static):   {gpd_stats.get('gpd_rgb_static', 'N/A')}")
    print(f"  θ drift:        {transform_stats.get('theta_frame_drift', 'N/A')}")
    print(f"\nReport saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
