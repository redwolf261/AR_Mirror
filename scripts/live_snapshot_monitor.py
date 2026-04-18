#!/usr/bin/env python3
"""Continuous monitor for AR Mirror snapshot API.

Pulls /api/snapshot at a fixed interval and writes:
- latest_frame.jpg
- latest_state.json
- timestamped history files (bounded by --keep)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _fetch_json(url: str, timeout: float = 6.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 (local dev endpoint)
        data = resp.read().decode("utf-8", errors="replace")
    return json.loads(data)


def _safe_num(value: Any) -> float | None:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if not (n == n):
        return None
    return n


def _summarize(state: dict[str, Any]) -> str:
    fit = state.get("fit_engine") or {}
    session = fit.get("session") or {}
    readiness = fit.get("readiness") or {}
    meas = state.get("measurements") or {}

    chest = _safe_num(meas.get("chest_cm"))
    shoulder = _safe_num(meas.get("shoulder_cm"))
    torso = _safe_num(meas.get("torso_cm"))
    fps = _safe_num(state.get("fps"))

    parts = [
        f"step={session.get('step', 'idle')}",
        f"ready={bool(readiness.get('measurement_ready'))}",
        f"fps={fps:.1f}" if fps is not None else "fps=--",
        f"chest={chest:.1f}" if chest is not None else "chest=--",
        f"shoulder={shoulder:.1f}" if shoulder is not None else "shoulder=--",
        f"torso={torso:.1f}" if torso is not None else "torso=--",
    ]
    return " | ".join(parts)


def _prune_history(folder: Path, keep: int) -> None:
    if keep <= 0:
        return
    for prefix in ("state_", "frame_"):
        items = sorted(folder.glob(f"{prefix}*"))
        overflow = len(items) - keep
        if overflow > 0:
            for old in items[:overflow]:
                try:
                    old.unlink(missing_ok=True)
                except OSError:
                    pass


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[monitor] API: {args.api_url}")
    print(f"[monitor] Output: {out_dir}")
    print(f"[monitor] Interval: {args.interval:.2f}s")

    while True:
        t0 = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((t0 % 1.0) * 1000)
        suffix = f"{stamp}_{ms:03d}"

        try:
            snap = _fetch_json(args.api_url)
            state = snap.get("state") or {}
            frame_b64 = snap.get("frame_jpeg_b64") or ""

            latest_state_path = out_dir / "latest_state.json"
            latest_frame_path = out_dir / "latest_frame.jpg"
            hist_state_path = out_dir / f"state_{suffix}.json"
            hist_frame_path = out_dir / f"frame_{suffix}.jpg"

            latest_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            hist_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

            if frame_b64:
                frame_bytes = base64.b64decode(frame_b64)
                latest_frame_path.write_bytes(frame_bytes)
                hist_frame_path.write_bytes(frame_bytes)

            _prune_history(out_dir, args.keep)
            print(f"[monitor] ok {suffix} | {_summarize(state)}")

        except urllib.error.URLError as exc:
            print(f"[monitor] network error: {exc}")
        except json.JSONDecodeError as exc:
            print(f"[monitor] bad JSON: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[monitor] error: {exc}")

        if args.once:
            return 0

        elapsed = time.time() - t0
        sleep_for = max(0.0, args.interval - elapsed)
        time.sleep(sleep_for)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live snapshot monitor for AR Mirror")
    parser.add_argument(
        "--api-url",
        default="http://localhost:5051/api/snapshot",
        help="Snapshot endpoint URL",
    )
    parser.add_argument(
        "--out-dir",
        default="logs/live-monitor",
        help="Directory for snapshot artifacts",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=180,
        help="Max history files per type to keep",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one snapshot and exit",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
