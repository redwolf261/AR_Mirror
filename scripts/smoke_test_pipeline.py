#!/usr/bin/env python3
"""
Headless smoke test for Phase 2 pipeline.
Runs a single warp_garment call with synthetic inputs and prints timing.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import time

print("=== Phase 2 Pipeline Smoke Test ===\n")

# 1. Load pipeline
t0 = time.perf_counter()
from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline  # noqa: E402
pipeline = Phase2NeuralPipeline(device="cuda", enable_tom=False)
load_time = time.perf_counter() - t0
print(f"[OK] Pipeline loaded in {load_time:.1f}s")

de = pipeline.depth_estimator
print(f"     Depth backend  : {de.backend if de else 'none'}")
sr = pipeline.smpl_reconstructor
print(f"     SMPL available : {sr.is_available if sr else False}")
print(f"     SMPL-X stub    : {type(sr).__name__ if sr else 'none'}")
print()

# 2. Synthetic inputs
person = np.random.rand(256, 192, 3).astype(np.float32)
cloth  = np.random.rand(256, 192, 3).astype(np.float32)
mask   = np.random.rand(256, 192).astype(np.float32)

# Minimal 33-joint landmark dict
lm = {i: {"x": 0.5, "y": 0.5, "visibility": 0.95} for i in range(33)}
# Reasonable shoulder/hip positions
for idx, (x, y) in [(11, (0.40, 0.35)), (12, (0.60, 0.35)),
                    (23, (0.40, 0.55)), (24, (0.60, 0.55))]:
    lm[idx] = {"x": x, "y": y, "visibility": 0.95}

# 3. Warm-up + timed run
print("Running warp_garment (warm-up) ...")
pipeline.warp_garment(person, cloth, mask, lm)

times = []
for i in range(5):
    t0 = time.perf_counter()
    result = pipeline.warp_garment(person, cloth, mask, lm)
    times.append(time.perf_counter() - t0)

avg_ms = sum(times) / len(times) * 1000
print(f"[OK] warp_garment x5 avg : {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS)")
print(f"     warped_cloth shape   : {result.warped_cloth.shape}")
print(f"     depth_proxy          : {result.depth_proxy:.4f}")
print(f"     quality_score        : {result.quality_score:.3f}")
print(f"     stage timings (ms)   :")
for k, v in result.timings.items():
    print(f"       {k:<30} {v*1000:.1f}")

print()
print("=== ALL OK ===")
