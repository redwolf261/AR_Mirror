#!/usr/bin/env python3
"""
D1: Real-World Robustness Tests

Tests 6 adversarial/edge-case scenarios to build a failure taxonomy.
Each scenario is synthetically simulated from webcam frames.

Scenarios:
  1. No person (empty frame)
  2. Partial occlusion (hand over torso)
  3. Extreme lighting (overexposed / underexposed)
  4. Rapid movement (frame-to-frame blur)
  5. Multi-person confusion
  6. Off-center / edge-of-frame pose

Usage:
    python scripts/utilities/robustness_test.py
    python scripts/utilities/robustness_test.py --frames 50
"""

import sys
import os
import json
import time
import traceback
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register CUDA DLLs
_nvidia_dir = PROJECT_ROOT / "ar" / "Lib" / "site-packages" / "nvidia"
if _nvidia_dir.exists():
    for _bin in _nvidia_dir.glob("*/bin"):
        if _bin.is_dir() and str(_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

import cv2


# ─── Scenario Simulators ────────────────────────────────────────────

def scenario_normal(frame):
    """Baseline: unmodified frame."""
    return frame

def scenario_no_person(frame):
    """Empty scene — solid color."""
    return np.full_like(frame, 128, dtype=np.uint8)

def scenario_partial_occlusion(frame):
    """Simulate hand/object over torso region."""
    h, w = frame.shape[:2]
    occluded = frame.copy()
    # Draw a dark rectangle over the center-torso area
    cx, cy = w // 2, h // 2
    cv2.rectangle(occluded, (cx - 80, cy - 100), (cx + 80, cy + 60), (30, 30, 30), -1)
    return occluded

def scenario_overexposed(frame):
    """Extreme brightness — blown-out highlights."""
    bright = cv2.convertScaleAbs(frame, alpha=2.5, beta=80)
    return bright

def scenario_underexposed(frame):
    """Extreme darkness — crushed shadows."""
    dark = cv2.convertScaleAbs(frame, alpha=0.3, beta=-30)
    return dark

def scenario_motion_blur(frame):
    """Simulate rapid horizontal movement blur."""
    kernel_size = 25
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size) / kernel_size
    return cv2.filter2D(frame, -1, kernel)

def scenario_off_center(frame):
    """Person at edge of frame — crop and pad."""
    h, w = frame.shape[:2]
    # Shift frame right by 40% — left side is black
    shifted = np.zeros_like(frame)
    offset = int(w * 0.4)
    shifted[:, offset:] = frame[:, :w - offset]
    return shifted


SCENARIOS = {
    'normal':            scenario_normal,
    'no_person':         scenario_no_person,
    'partial_occlusion': scenario_partial_occlusion,
    'overexposed':       scenario_overexposed,
    'underexposed':      scenario_underexposed,
    'motion_blur':       scenario_motion_blur,
    'off_center':        scenario_off_center,
}


# ─── Test Runner ────────────────────────────────────────────────────

def run_scenario(body_fitter, pipeline, cloth_rgb, cloth_mask,
                 scenario_name, scenario_fn, frames, cap):
    """Run a single scenario and collect metrics."""
    results = {
        'scenario': scenario_name,
        'frames_tested': 0,
        'pose_detections': 0,
        'pose_failures': 0,
        'warp_successes': 0,
        'warp_failures': 0,
        'crashes': 0,
        'crash_details': [],
        'avg_fps': 0,
        'avg_confidence': 0,
    }
    
    confidences = []
    frame_times = []
    
    for i in range(frames):
        ret, raw_frame = cap.read()
        if not ret:
            break
        
        raw_frame = cv2.flip(raw_frame, 1)
        frame = scenario_fn(raw_frame)
        results['frames_tested'] += 1
        
        t0 = time.time()
        try:
            # Pose detection
            measurements = body_fitter.extract_body_measurements(frame)
            
            if measurements is None:
                results['pose_failures'] += 1
            else:
                results['pose_detections'] += 1
                confidences.append(body_fitter.last_confidence)
                
                # Neural warp
                if pipeline and cloth_rgb is not None:
                    try:
                        h_f, w_f = frame.shape[:2]
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
                        results['warp_successes'] += 1
                    except Exception as e:
                        results['warp_failures'] += 1
                        if len(results['crash_details']) < 3:
                            results['crash_details'].append(str(e)[:200])
        except Exception as e:
            results['crashes'] += 1
            if len(results['crash_details']) < 3:
                results['crash_details'].append(f"CRASH: {str(e)[:200]}")
        
        frame_times.append(time.time() - t0)
    
    if frame_times:
        results['avg_fps'] = round(1.0 / np.mean(frame_times), 1)
    if confidences:
        results['avg_confidence'] = round(float(np.mean(confidences)), 3)
    
    # Compute rates
    total = results['frames_tested']
    if total > 0:
        results['detection_rate'] = round(results['pose_detections'] / total, 3)
        results['crash_rate'] = round(results['crashes'] / total, 4)
        warp_attempts = results['warp_successes'] + results['warp_failures']
        results['warp_success_rate'] = round(results['warp_successes'] / warp_attempts, 3) if warp_attempts > 0 else 0
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='D1: Robustness Test Suite')
    parser.add_argument('--frames', type=int, default=30,
                        help='Frames per scenario (default: 30)')
    parser.add_argument('--output', type=str, default='output/robustness_report.json')
    args = parser.parse_args()
    
    print("=" * 70)
    print("ROBUSTNESS TEST SUITE — 7 Scenarios")
    print("=" * 70)
    
    # Initialize
    print("\n[1/3] Initializing...")
    from src.core.body_aware_fitter import BodyAwareGarmentFitter
    body_fitter = BodyAwareGarmentFitter()
    
    pipeline = None
    try:
        from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
        pipeline = Phase2NeuralPipeline(
            device='auto', enable_tom=False,
            batch_size=1, enable_optimizations=True
        )
    except Exception as e:
        print(f"  [WARN] Pipeline not available: {e}")
    
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
    
    # Camera
    print("[2/3] Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return
    
    # Run scenarios
    print(f"\n[3/3] Running {len(SCENARIOS)} scenarios × {args.frames} frames each...\n")
    all_results = []
    
    for i, (name, fn) in enumerate(SCENARIOS.items(), 1):
        print(f"  [{i}/{len(SCENARIOS)}] {name}...", end=" ", flush=True)
        body_fitter.landmark_smoother.reset()
        body_fitter._consecutive_static = 0
        body_fitter._locked_measurements = None
        
        result = run_scenario(body_fitter, pipeline, cloth_rgb, cloth_mask,
                              name, fn, args.frames, cap)
        all_results.append(result)
        
        status = "✓" if result['crashes'] == 0 else "✗ CRASHES"
        print(f"{status} | detect={result.get('detection_rate', 0):.0%} "
              f"warp={result.get('warp_success_rate', 0):.0%} "
              f"crashes={result['crashes']} FPS={result['avg_fps']}")
    
    cap.release()
    
    # Build failure taxonomy
    taxonomy = {
        'total_scenarios': len(SCENARIOS),
        'total_frames': sum(r['frames_tested'] for r in all_results),
        'total_crashes': sum(r['crashes'] for r in all_results),
        'scenarios_with_crashes': sum(1 for r in all_results if r['crashes'] > 0),
        'scenarios_zero_detection': sum(1 for r in all_results if r.get('detection_rate', 0) == 0),
        'worst_detection_scenario': min(all_results, key=lambda r: r.get('detection_rate', 0))['scenario'],
        'crash_free': sum(r['crashes'] for r in all_results) == 0,
    }
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'frames_per_scenario': args.frames,
        'taxonomy': taxonomy,
        'scenarios': all_results,
    }
    
    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("FAILURE TAXONOMY")
    print("=" * 70)
    print(json.dumps(taxonomy, indent=2))
    print(f"\nFull report: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
