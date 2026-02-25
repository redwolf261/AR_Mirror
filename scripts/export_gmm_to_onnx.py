#!/usr/bin/env python3
"""
Export trained GMM checkpoint → models/gmm_model.onnx

Usage:
    .\\ar\\Scripts\\python.exe scripts\\export_gmm_to_onnx.py

No arguments needed. Reads from:
    cp-vton/checkpoints/gmm_train_new/gmm_final.pth

Writes to:
    models/gmm_model.onnx

After export, verify with:
    .\\ar\\Scripts\\python.exe scripts\\export_gmm_to_onnx.py --verify
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cp-vton"))  # makes networks.py importable directly

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "output" / "logs" / "export_gmm.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("export_gmm")

# ── paths ────────────────────────────────────────────────────────────────────
CHECKPOINT = ROOT / "cp-vton" / "checkpoints" / "gmm_train_new" / "gmm_final.pth"
ONNX_OUT   = ROOT / "models" / "gmm_model.onnx"

# GMM input dimensions (must match training config)
# inputA: agnostic representation (pose 18ch + shape 1ch + head 3ch = 22ch) @ 192×256
# inputB: cloth MASK (1ch — binary silhouette, NOT RGB cloth)              @ 192×256
INPUT_H, INPUT_W = 256, 192
INPUT_A_CH = 22   # pose+shape+head agnostic
INPUT_B_CH = 1    # cloth mask (binary)
GRID_SIZE  = 5    # TPS control-point grid (inferred from regression.linear output_dim=50)


def build_gmm() -> torch.nn.Module:
    """Instantiate GMM from cp-vton/networks.py with correct architecture params.

    Confirmed from checkpoint inspection:
      extractionA: input_nc=22, ngf=64
      extractionB: input_nc=1,  ngf=64   (cloth MASK, not RGB)
      grid_size=5  (regression.linear.weight shape[0] = 50 = 2 * 5^2)
    """
    import networks as _nets  # type: ignore  (cp-vton/ is on sys.path)

    import argparse
    opt = argparse.Namespace(
        grid_size=GRID_SIZE,
        fine_height=INPUT_H,
        fine_width=INPUT_W,
        use_cuda=False,   # CPU export; TpsGridGen will use cpu tensors
    )
    model = _nets.GMM(opt)
    log.info(f"GMM instantiated  (grid_size={GRID_SIZE}, H={INPUT_H}, W={INPUT_W})")
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> torch.nn.Module:
    log.info(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        log.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    log.info(f"  Checkpoint loaded  ({len(state)} tensors)")
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_a = torch.randn(1, INPUT_A_CH, INPUT_H, INPUT_W)
    dummy_b = torch.randn(1, INPUT_B_CH, INPUT_H, INPUT_W)   # cloth MASK (1ch)

    log.info(f"Tracing GMM with inputs:  agnostic={tuple(dummy_a.shape)}  cloth_mask={tuple(dummy_b.shape)}")

    # Ensure TpsGridGen runs on CPU during export (use_cuda=False in build_gmm)
    model.eval()
    with torch.no_grad():
        # Sanity-check forward pass before exporting
        grid_out, theta_out = model(dummy_a, dummy_b)
        log.info(f"Forward pass OK  grid={tuple(grid_out.shape)}  theta={tuple(theta_out.shape)}")

    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        (dummy_a, dummy_b),
        str(output_path),
        input_names=["agnostic", "cloth_mask"],
        output_names=["grid", "theta"],
        dynamic_axes={
            "agnostic":   {0: "batch"},
            "cloth_mask": {0: "batch"},
            "grid":       {0: "batch"},
            "theta":      {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / 1_048_576
    log.info(f"ONNX export complete  →  {output_path}  ({size_mb:.1f} MB, {elapsed:.1f}s)")


def _add_torch_lib_to_path() -> None:
    """Add PyTorch's bundled CUDA DLLs to PATH so ORT CUDA EP can find cublasLt etc."""
    import os
    try:
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            current = os.environ.get("PATH", "")
            s = str(torch_lib)
            if s not in current:
                os.environ["PATH"] = s + os.pathsep + current
                log.info(f"Added torch/lib to PATH: {s}")
    except Exception:
        pass


def verify_onnx(onnx_path: Path) -> None:
    """Run ONNX model via onnxruntime and compare to PyTorch output."""
    log.info("─── Verification ───────────────────────────────────────")

    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping ORT verify")
        return

    # Inject CUDA DLLs before creating the session
    _add_torch_lib_to_path()

    # ── PyTorch reference ──
    model = load_checkpoint(build_gmm(), CHECKPOINT)
    dummy_a = torch.randn(1, INPUT_A_CH, INPUT_H, INPUT_W)
    dummy_b = torch.randn(1, INPUT_B_CH, INPUT_H, INPUT_W)   # cloth mask (1ch)

    with torch.no_grad():
        pt_grid, pt_theta = model(dummy_a, dummy_b)
    pt_np = pt_grid.numpy()

    # ── ORT inference ──
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    active = sess.get_providers()[0]
    log.info(f"ORT execution provider: {active}")

    ort_outputs = sess.run(None, {
        "agnostic":   dummy_a.numpy(),
        "cloth_mask": dummy_b.numpy(),
    })
    ort_np = ort_outputs[0]   # grid output

    max_diff = float(np.abs(pt_np - ort_np).max())
    mean_diff = float(np.abs(pt_np - ort_np).mean())
    log.info(f"PyTorch vs ORT  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

    if max_diff < 1e-3:
        log.info("✓ Verification PASSED — outputs match within tolerance")
    else:
        log.warning(f"⚠ Verification WARNING — max diff {max_diff:.5f} > 1e-3")

    # ── Latency benchmark (20 warm + 100 timed) ──
    log.info("Running latency benchmark (100 iterations)…")
    bench_inputs = {"agnostic": dummy_a.numpy(), "cloth_mask": dummy_b.numpy()}
    for _ in range(20):
        sess.run(None, bench_inputs)

    t0 = time.perf_counter()
    for _ in range(100):
        sess.run(None, bench_inputs)
    ms_per_frame = (time.perf_counter() - t0) / 100 * 1000
    log.info(f"ORT inference latency: {ms_per_frame:.2f} ms/frame  ({1000/ms_per_frame:.0f} FPS theoretical)")
    log.info("────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="Export GMM checkpoint to ONNX")
    parser.add_argument("--verify", action="store_true",
                        help="After export, run ORT verification + latency benchmark")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT),
                        help=f"Path to .pth checkpoint (default: {CHECKPOINT})")
    parser.add_argument("--output", default=str(ONNX_OUT),
                        help=f"Output .onnx path (default: {ONNX_OUT})")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path  = Path(args.output)

    log.info("══════════════════════════════════════════════════════")
    log.info("  GMM → ONNX Export")
    log.info(f"  checkpoint : {ckpt_path}")
    log.info(f"  output     : {out_path}")
    log.info(f"  input size : {INPUT_A_CH}ch × {INPUT_H}×{INPUT_W}")
    log.info("══════════════════════════════════════════════════════")

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = load_checkpoint(build_gmm(), ckpt_path)
    export_onnx(model, out_path)

    if args.verify:
        verify_onnx(out_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
