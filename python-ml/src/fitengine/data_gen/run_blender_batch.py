"""
Subprocess launcher for N parallel Blender renders — Phase 2 / Month 3.

Usage:
    python fitengine/data_gen/run_blender_batch.py --n 5000 --workers 1

# TODO (Phase 2, Month 3)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def launch_batch(n: int, workers: int, output_dir: str) -> None:
    """
    Launch `workers` Blender subprocess renders covering `n` subjects total.

    # TODO (Phase 2, Month 3)
    """
    raise NotImplementedError(
        "run_blender_batch.py is a Phase 2 / Month 3 implementation target."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output",  type=str, default="data/blender_out/")
    args = parser.parse_args()
    launch_batch(args.n, args.workers, args.output)
