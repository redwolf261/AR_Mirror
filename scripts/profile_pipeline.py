#!/usr/bin/env python3
"""
Pipeline stage profiler — breaks down where the 74ms warp_garment goes.
Runs 20 warm frames and prints a per-stage timed table.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import time
import logging
logging.basicConfig(level=logging.WARNING)  # suppress init noise

from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline

# ── synthetic inputs ──────────────────────────────────────────────────
person = np.random.rand(256, 192, 3).astype(np.float32)
cloth  = np.random.rand(256, 192, 3).astype(np.float32)
mask   = np.random.rand(256, 192).astype(np.float32)
# Pre-computed body_mask as in production (skips segmentation model call)
body_mask = (np.random.rand(256, 192) > 0.3).astype(np.float32)
lm = {i: {"x": 0.5, "y": 0.5, "visibility": 0.95} for i in range(33)}
for idx, (x, y) in [(11, (0.40, 0.35)), (12, (0.60, 0.35)),
                    (23, (0.40, 0.55)), (24, (0.60, 0.55))]:
    lm[idx] = {"x": x, "y": y, "visibility": 0.95}

print("Loading pipeline (suppressed)...")
pipeline = Phase2NeuralPipeline(device="cuda", enable_tom=False)
print("Loaded.\n")

# ── warm-up ───────────────────────────────────────────────────────────
for _ in range(3):
    pipeline.warp_garment(person, cloth, mask, lm, body_mask=body_mask)

# ── timed runs ────────────────────────────────────────────────────────
N = 20
stage_totals: dict = {}
wall_times = []

for _ in range(N):
    t0 = time.perf_counter()
    result = pipeline.warp_garment(person, cloth, mask, lm, body_mask=body_mask)
    wall_times.append(time.perf_counter() - t0)
    for k, v in result.timings.items():
        stage_totals[k] = stage_totals.get(k, 0.0) + v

avg_wall = sum(wall_times) / N * 1000
stage_avgs = {k: v / N * 1000 for k, v in stage_totals.items()}
accounted = sum(stage_avgs.values())
overhead = avg_wall - accounted

print(f"{'Stage':<35} {'avg ms':>8}  {'% total':>8}")
print("-" * 55)
for stage, ms in sorted(stage_avgs.items(), key=lambda x: -x[1]):
    pct = ms / avg_wall * 100
    print(f"  {stage:<33} {ms:>8.1f}  {pct:>7.1f}%")
print(f"  {'[unaccounted overhead]':<33} {overhead:>8.1f}  {overhead/avg_wall*100:>7.1f}%")
print("-" * 55)
print(f"  {'TOTAL wall time':<33} {avg_wall:>8.1f}  {'100.0':>7}%")
print(f"\n  Effective FPS: {1000/avg_wall:.1f}")
