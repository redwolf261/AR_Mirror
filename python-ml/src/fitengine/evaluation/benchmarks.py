"""
FitEngine evaluation benchmarks.

Metrics:
    adjacent_accuracy(pred, gt):  abs(pred_idx - gt_idx) <= 1
    nonadjacent_error_rate:       fraction where abs(pred_idx - gt_idx) > 1

Phase gates (from configs/default.yaml):
    PHASE1_GATE:
        jacket_adj_acc          >= 0.80
        nonadjacent_error_rate  <= 0.10

    PRODUCTION_GATE:
        jacket_adj_acc          >= 0.88
        collar_adj_acc          >= 0.85
        nonadjacent_error_rate  <= 0.05

Benchmark suites (stub targets for Phase 2, Month 3):
    noise_stability_curve      - adj_acc vs. increasing noise σ
    silhouette_ablation        - performance with mask only (no keypoints)
    height_ablation            - performance without height input
    occlusion_stress_test      - random rectangle occlusions at 10/25/50 % area
    pilot_data_eval            - run against annotated PilotLogger JSONL file

CLI:
    python -m fitengine.evaluation.benchmarks \\
        --checkpoint checkpoints/best.pt \\
        --val-h5 data/val.h5 \\
        [--pilot-log data/pilot.jsonl] \\
        [--gate phase1|production]

# TODO (Phase 2, Month 3): implement after PoseDataset and trainer are live.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

# ---------------------------------------------------------------------------
# Gate constants (mirrors configs/default.yaml)
# ---------------------------------------------------------------------------

PHASE1_GATE: dict[str, float] = {
    "jacket_adj_acc": 0.80,
    "nonadjacent_error_rate": 0.10,
}

PRODUCTION_GATE: dict[str, float] = {
    "jacket_adj_acc": 0.88,
    "collar_adj_acc": 0.85,
    "nonadjacent_error_rate": 0.05,
}


# ---------------------------------------------------------------------------
# Core metric (usable in Phase 1 for manual spot-checks)
# ---------------------------------------------------------------------------

def adjacent_accuracy(
    pred_indices: Sequence[int],
    gt_indices: Sequence[int],
) -> float:
    """
    Fraction of predictions within ±1 class index of ground truth.

    Args:
        pred_indices: Predicted class indices (int).
        gt_indices:   Ground-truth class indices (int).

    Returns:
        Float in [0, 1].
    """
    pred = np.asarray(pred_indices, dtype=int)
    gt = np.asarray(gt_indices, dtype=int)
    if len(pred) == 0:
        return 0.0
    return float(np.mean(np.abs(pred - gt) <= 1))


def nonadjacent_error_rate(
    pred_indices: Sequence[int],
    gt_indices: Sequence[int],
) -> float:
    """Fraction of predictions more than 1 class index away from ground truth."""
    pred = np.asarray(pred_indices, dtype=int)
    gt = np.asarray(gt_indices, dtype=int)
    if len(pred) == 0:
        return 0.0
    return float(np.mean(np.abs(pred - gt) > 1))


def check_gate(metrics: dict[str, float], gate: dict[str, float]) -> dict[str, bool]:
    """
    Check each metric against its gate threshold.

    Returns:
        Dict mapping metric name to pass/fail bool.  True = passing.
    """
    results: dict[str, bool] = {}
    for key, threshold in gate.items():
        val = metrics.get(key, 0.0)
        if key == "nonadjacent_error_rate":
            results[key] = val <= threshold          # lower is better
        else:
            results[key] = val >= threshold          # higher is better
    return results


# ---------------------------------------------------------------------------
# Phase 2 benchmark suites — stubs
# ---------------------------------------------------------------------------

def noise_stability_curve(
    checkpoint,
    val_h5,
    sigma_values=(0.0, 0.02, 0.05, 0.10, 0.20),
) -> dict:
    """
    Evaluate jacket_adj_acc at increasing joint-noise levels.

    # TODO (Phase 2, Month 3)
    """
    raise NotImplementedError("noise_stability_curve is a Phase 2 benchmark target.")


def silhouette_ablation(checkpoint, val_h5) -> dict:
    """
    Evaluate accuracy using silhouette width curves only (zero keypoints).

    # TODO (Phase 2, Month 3)
    """
    raise NotImplementedError("silhouette_ablation is a Phase 2 benchmark target.")


def height_ablation(checkpoint, val_h5) -> dict:
    """
    Evaluate accuracy with height_norm zeroed out.

    # TODO (Phase 2, Month 3)
    """
    raise NotImplementedError("height_ablation is a Phase 2 benchmark target.")


def occlusion_stress_test(
    checkpoint,
    val_h5,
    occlusion_fracs=(0.10, 0.25, 0.50),
) -> dict:
    """
    Apply random rectangle occlusions and measure degradation.

    # TODO (Phase 2, Month 3)
    """
    raise NotImplementedError("occlusion_stress_test is a Phase 2 benchmark target.")


def pilot_data_eval(checkpoint, pilot_log_path) -> dict:
    """
    Evaluate model on annotated PilotLogger JSONL ground-truth labels.

    # TODO (Phase 2, Month 3)
    """
    raise NotImplementedError("pilot_data_eval is a Phase 2 benchmark target.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FitEngine evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-h5", required=True)
    parser.add_argument("--pilot-log", default=None)
    parser.add_argument(
        "--gate", choices=["phase1", "production"], default="phase1"
    )
    args = parser.parse_args()
    raise SystemExit("benchmarks.py not yet fully implemented (Phase 2 target).")
