#!/usr/bin/env python3
"""Compute measurement and size accuracy against ground-truth records.

Input format: JSONL file where each row includes predicted + actual values.
Expected flexible keys (any subset):
- predicted.shoulder_cm / actual.shoulder_cm
- predicted.chest_cm / actual.chest_cm
- predicted.waist_cm / actual.waist_cm
- predicted.torso_cm / actual.torso_cm
- predicted.size / actual.size

Fallback flat keys are also supported:
- predicted_shoulder_cm, actual_shoulder_cm, ...
- predicted_size, actual_size
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

METRICS = ("shoulder_cm", "chest_cm", "waist_cm", "torso_cm")


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _get_nested(row: Dict[str, object], dotted_key: str) -> object:
    parts = dotted_key.split(".")
    node: object = row
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _pick(row: Dict[str, object], candidates: Iterable[str]) -> object:
    for key in candidates:
        if "." in key:
            value = _get_nested(row, key)
            if value is not None:
                return value
        elif key in row and row[key] is not None:
            return row[key]
    return None


def _bin_index(confidence_0_100: float, bin_size: int = 10) -> int:
    c = max(0.0, min(100.0, confidence_0_100))
    return int(c // bin_size)


def evaluate(records: List[Dict[str, object]]) -> Dict[str, object]:
    errors: Dict[str, List[float]] = defaultdict(list)
    abs_errors: Dict[str, List[float]] = defaultdict(list)

    size_total = 0
    size_correct = 0

    conf_bins: Dict[int, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "correct": 0.0})

    for row in records:
        for metric in METRICS:
            pred = _to_float(
                _pick(
                    row,
                    (
                        f"predicted.{metric}",
                        f"prediction.{metric}",
                        f"pred.{metric}",
                        f"predicted_{metric}",
                        f"prediction_{metric}",
                    ),
                )
            )
            actual = _to_float(
                _pick(
                    row,
                    (
                        f"actual.{metric}",
                        f"ground_truth.{metric}",
                        f"truth.{metric}",
                        f"actual_{metric}",
                        f"ground_truth_{metric}",
                    ),
                )
            )
            if pred is None or actual is None:
                continue

            err = pred - actual
            errors[metric].append(err)
            abs_errors[metric].append(abs(err))

        pred_size = _pick(row, ("predicted.size", "prediction.size", "predicted_size", "prediction_size", "size_pred"))
        actual_size = _pick(row, ("actual.size", "ground_truth.size", "actual_size", "ground_truth_size", "size_actual"))
        if isinstance(pred_size, str) and isinstance(actual_size, str):
            size_total += 1
            correct = int(pred_size.strip().upper() == actual_size.strip().upper())
            size_correct += correct

            conf = _to_float(_pick(row, ("predicted.confidence", "prediction.confidence", "confidence", "size_confidence")))
            if conf is not None:
                # Support either 0..1 or 0..100
                if conf <= 1.0:
                    conf *= 100.0
                b = _bin_index(conf)
                conf_bins[b]["count"] += 1.0
                conf_bins[b]["correct"] += float(correct)

    metric_summary: Dict[str, object] = {}
    for metric in METRICS:
        e = errors[metric]
        ae = abs_errors[metric]
        if not e:
            metric_summary[metric] = {"n": 0, "mae": None, "rmse": None, "bias": None}
            continue

        n = len(e)
        mae = sum(ae) / n
        rmse = math.sqrt(sum(v * v for v in e) / n)
        bias = sum(e) / n
        metric_summary[metric] = {
            "n": n,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "bias": round(bias, 3),
        }

    reliability = []
    for b in sorted(conf_bins.keys()):
        c = conf_bins[b]["count"]
        if c <= 0:
            continue
        acc = conf_bins[b]["correct"] / c
        lo = b * 10
        hi = lo + 10
        reliability.append(
            {
                "bin": f"{lo}-{hi}",
                "count": int(c),
                "empirical_accuracy": round(acc, 4),
            }
        )

    return {
        "records": len(records),
        "measurement_metrics": metric_summary,
        "size": {
            "n": size_total,
            "top1_accuracy": round(size_correct / size_total, 4) if size_total else None,
        },
        "confidence_reliability": reliability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AR Mirror measurement/size accuracy from JSONL ground-truth data")
    parser.add_argument("input", type=Path, help="Path to JSONL validation file")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON report path")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows: List[Dict[str, object]] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    report = evaluate(rows)
    print(json.dumps(report, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
