"""
FitEngine ONNX + WASM exporter — Phase 3 SaaS export pipeline.

Produces:
    exports/
        regressor.onnx      - DualViewRegressor, opset 17, dynamic batch
        classifier.onnx     - FitEngineClassifierBundle, opset 17, dynamic batch
        regressor.wasm      - ONNX Runtime Web WASM artefact
        regressor.data      - companion weights file for WASM
        classifier.wasm
        classifier.data

Validation:
    Max abs error vs PyTorch reference < 1e-4 on 512 random inputs.

CLI:
    python -m fitengine.exporter \\
        --checkpoint checkpoints/best.pt \\
        --output exports/ \\
        --targets onnx wasm

# TODO (Phase 3, Month 4): implement after Phase 2 training reaches Gate 5
#       (jacket_adj_acc >= 0.88 on held-out real set).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def export_onnx(
    checkpoint: str | Path,
    output_dir: str | Path,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> dict[str, Path]:
    """
    Export DualViewRegressor and FitEngineClassifierBundle to ONNX.

    Returns:
        {'regressor': Path, 'classifier': Path}

    # TODO (Phase 3, Month 4)
    """
    raise NotImplementedError(
        "export_onnx is a Phase 3 implementation target."
    )


def export_wasm(
    onnx_paths: dict[str, Path],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Convert ONNX models to ONNX Runtime Web WASM bundles.

    Returns:
        {'regressor_wasm': Path, 'regressor_data': Path,
         'classifier_wasm': Path, 'classifier_data': Path}

    Prerequisites:
        npm install -g onnxruntime-web
        npx ort-wasm-pack --model regressor.onnx --output exports/

    # TODO (Phase 3, Month 4)
    """
    raise NotImplementedError(
        "export_wasm is a Phase 3 implementation target."
    )


def validate_onnx(
    onnx_path: Path,
    checkpoint: Path,
    n_samples: int = 512,
    tolerance: float = 1e-4,
) -> bool:
    """
    Verify max absolute error between ONNX and PyTorch outputs < tolerance.

    # TODO (Phase 3, Month 4)
    """
    raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export FitEngine models")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--output", default="exports/", help="Output directory")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["onnx", "wasm"],
        default=["onnx"],
        help="Export targets",
    )
    args = parser.parse_args()
    raise SystemExit("Exporter not yet implemented (Phase 3 target).")
