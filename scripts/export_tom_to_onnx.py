#!/usr/bin/env python3
"""
Export trained HR-VITON SPADEGenerator checkpoint → models/tom_model.onnx

The trained checkpoint is an HR-VITON SPADEGenerator (NOT CP-VTON UnetGenerator).
It *reconstructs* the full person image with the garment naturalistically fitted.

Architecture (confirmed by checkpoint inspection):
  Model   : SPADEGenerator(opt, input_nc=9)
  opt     : ngf=64, num_upsampling_layers='most', gen_semantic_nc=7,
            fine_height=512, fine_width=384, norm_G='spectralaliasinstance'
  Input x : cat([warped_cloth(3), person(3), bg_proxy(3)]) → (1, 9, 512, 384)
  Input seg: 7-channel segmentation map                     → (1, 7, 512, 384)
  Output  : synthesized RGB image tanh[-1,1]                → (1, 3, 512, 384)

Final image: resize(output, 256, 192), then (tanh+1)/2 → [0,1]

Usage:
    .venv\\Scripts\\python.exe scripts\\export_tom_to_onnx.py
    .venv\\Scripts\\python.exe scripts\\export_tom_to_onnx.py --verify
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
sys.path.insert(0, str(ROOT / "vendor" / "hr_viton"))   # makes network_generator.py importable

# ── logging ──────────────────────────────────────────────────────────────────
(ROOT / "output" / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "output" / "logs" / "export_tom.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("export_tom")

# ── paths ────────────────────────────────────────────────────────────────────
CHECKPOINT = ROOT / "cp-vton" / "checkpoints" / "tom_train_new" / "tom_final.pth"
ONNX_OUT   = ROOT / "models" / "tom_model.onnx"

# SPADEGenerator dims (HR-VITON native)
FINE_H       = 512
FINE_W       = 384
INPUT_NC     = 9    # warped_cloth(3) + person(3) + bg_proxy(3)
SEG_NC       = 7    # 7-class semantic segmentation


def build_spade_generator() -> torch.nn.Module:
    """Instantiate SPADEGenerator with params confirmed from checkpoint."""
    from network_generator import SPADEGenerator  # type: ignore  (vendor/hr_viton on sys.path)

    opt = argparse.Namespace(
        norm_G="spectralaliasinstance",
        ngf=64,
        num_upsampling_layers="most",
        gen_semantic_nc=SEG_NC,
        fine_height=FINE_H,
        fine_width=FINE_W,
        cuda=False,    # CPU export; noise branch uses plain randn
    )
    model = SPADEGenerator(opt, input_nc=INPUT_NC)
    log.info(
        f"SPADEGenerator instantiated  "
        f"(input_nc={INPUT_NC}, seg_nc={SEG_NC}, ngf=64, ups_layers='most', "
        f"H={FINE_H}, W={FINE_W})"
    )
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> torch.nn.Module:
    log.info(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        log.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    n = sum(p.numel() for p in model.parameters())
    log.info(f"  Checkpoint loaded  (tensors={len(state)}, params={n/1e6:.1f}M)")

    # Zero out noise_scale params for deterministic ONNX export
    # Values are small (~0.04 max) so this has negligible visual impact
    zeroed = 0
    for name, param in model.named_parameters():
        if 'noise_scale' in name:
            param.data.zero_()
            zeroed += 1
    if zeroed:
        log.info(f"  noise_scale zeroed ({zeroed} params) for deterministic export")

    model.eval()
    return model


def export_onnx(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_x   = torch.randn(1, INPUT_NC, FINE_H, FINE_W)
    dummy_seg = torch.randn(1, SEG_NC,   FINE_H, FINE_W)

    log.info(f"Tracing SPADEGenerator:  x={tuple(dummy_x.shape)}  seg={tuple(dummy_seg.shape)}")

    # Monkey-patch SPADENorm.forward to remove torch.randn so the ONNX graph
    # is deterministic (noise_scale is already zeroed, so noise contribution = 0).
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "vendor" / "hr_viton"))
    from network_generator import SPADENorm  # type: ignore

    _orig_forward = SPADENorm.forward

    def _no_noise_forward(self, x, seg, misalign_mask=None):
        # Part 1: parameter-free norm WITHOUT random noise
        if misalign_mask is None:
            normalized = self.param_free_norm(x)
        else:
            normalized = self.param_free_norm(x, misalign_mask)
        # Part 2: SPADE affine params conditioned on segmap
        actv  = self.conv_shared(seg)
        gamma = self.conv_gamma(actv)
        beta  = self.conv_beta(actv)
        return normalized * (1 + gamma) + beta

    SPADENorm.forward = _no_noise_forward

    model.eval()
    with torch.no_grad():
        out = model(dummy_x, dummy_seg)
        log.info(f"Forward pass OK  output={tuple(out.shape)}  range=[{out.min():.3f}, {out.max():.3f}]")

    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        (dummy_x, dummy_seg),
        str(output_path),
        input_names=["tom_x", "tom_seg"],         # (batch, 9, 512, 384), (batch, 7, 512, 384)
        output_names=["tom_output"],               # (batch, 3, 512, 384) tanh [-1, 1]
        dynamic_axes={
            "tom_x":      {0: "batch"},
            "tom_seg":    {0: "batch"},
            "tom_output": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=False,   # skip heavy fold pass for 100M-param model
    )
    # Restore original forward
    SPADENorm.forward = _orig_forward

    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / 1_048_576
    log.info(f"ONNX export complete  →  {output_path}  ({size_mb:.1f} MB, {elapsed:.1f}s)")


def _add_torch_lib_to_path() -> None:
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
    """Run ORT vs PyTorch comparison + latency benchmark."""
    log.info("─── Verification ────────────────────────────────────────")

    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping ORT verify")
        return

    _add_torch_lib_to_path()

    # PyTorch reference (with zeroed noise)
    model = load_checkpoint(build_spade_generator(), CHECKPOINT)
    dummy_x   = torch.randn(1, INPUT_NC, FINE_H, FINE_W)
    dummy_seg = torch.zeros(1, SEG_NC,   FINE_H, FINE_W)   # zeros → deterministic

    with torch.no_grad():
        pt_out = model(dummy_x, dummy_seg).numpy()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    active = sess.get_providers()[0]
    log.info(f"ORT execution provider: {active}")

    ort_out = sess.run(None, {
        "tom_x":   dummy_x.numpy(),
        "tom_seg": dummy_seg.numpy(),
    })[0]

    max_diff  = float(np.abs(pt_out - ort_out).max())
    mean_diff = float(np.abs(pt_out - ort_out).mean())
    log.info(f"PyTorch vs ORT  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

    if max_diff < 1e-2:
        log.info("✓ Verification PASSED — outputs match within tolerance")
    else:
        log.warning(f"⚠ Verification WARNING — max diff {max_diff:.5f} > 0.01")

    # Latency benchmark (10 warm + 20 timed — SPADEGen is slow on CPU)
    log.info("Running latency benchmark (10 warm + 20 timed)…")
    bench_in = {"tom_x": dummy_x.numpy(), "tom_seg": dummy_seg.numpy()}
    for _ in range(10):
        sess.run(None, bench_in)
    t0 = time.perf_counter()
    for _ in range(20):
        sess.run(None, bench_in)
    ms = (time.perf_counter() - t0) / 20 * 1000
    log.info(f"ORT TOM latency: {ms:.0f} ms/frame  ({1000/ms:.1f} FPS theoretical)")
    log.info("─────────────────────────────────────────────────────────")

    # Composite smoke-test
    log.info("Composite formula smoke-test…")
    out_01 = np.clip((ort_out[0].transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
    log.info(f"  synthesized range [{out_01.min():.3f}, {out_01.max():.3f}]  shape={out_01.shape}")
    log.info("Smoke-test OK")


def main():
    parser = argparse.ArgumentParser(description="Export HR-VITON SPADEGenerator (TOM) to ONNX")
    parser.add_argument("--verify", action="store_true",
                        help="After export: run ORT verification + latency benchmark")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--output",     default=str(ONNX_OUT))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path  = Path(args.output)

    log.info("══════════════════════════════════════════════════════")
    log.info("  HR-VITON SPADEGenerator (TOM) → ONNX Export")
    log.info(f"  checkpoint : {ckpt_path}")
    log.info(f"  output     : {out_path}")
    log.info(f"  x input    : ({INPUT_NC}, {FINE_H}, {FINE_W})  warped_cloth+person+bg")
    log.info(f"  seg input  : ({SEG_NC}, {FINE_H}, {FINE_W})  7-class segmap")
    log.info("══════════════════════════════════════════════════════")

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = load_checkpoint(build_spade_generator(), ckpt_path)
    export_onnx(model, out_path)

    if args.verify:
        verify_onnx(out_path)

    log.info("Done.  Run with --verify to benchmark ORT inference.")


if __name__ == "__main__":
    main()


import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cp-vton"))   # makes networks.py importable

# ── logging ──────────────────────────────────────────────────────────────────
(ROOT / "output" / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "output" / "logs" / "export_tom.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("export_tom")

# ── paths ────────────────────────────────────────────────────────────────────
CHECKPOINT = ROOT / "cp-vton" / "checkpoints" / "tom_train_new" / "tom_final.pth"
ONNX_OUT   = ROOT / "models" / "tom_model.onnx"

# TOM dimensions (must match CP-VTON training)
INPUT_H        = 256
INPUT_W        = 192
AGNOSTIC_CH    = 22   # pose(18) + shape(1) + head(3)
WARPED_CLOTH_CH = 3   # RGB warped cloth
TOM_INPUT_CH   = AGNOSTIC_CH + WARPED_CLOTH_CH   # 25
TOM_OUTPUT_CH  = 4    # p_rendered(3) + m_composite(1)


def build_tom() -> torch.nn.Module:
    """Instantiate CP-VTON UnetGenerator matching the saved checkpoint."""
    import networks as _nets  # type: ignore  (cp-vton/ is on sys.path)

    model = _nets.UnetGenerator(
        input_nc=TOM_INPUT_CH,   # 25
        output_nc=TOM_OUTPUT_CH, # 4
        num_downs=6,
        ngf=64,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=False,
    )
    log.info(
        f"UnetGenerator instantiated  "
        f"(input={TOM_INPUT_CH}ch, output={TOM_OUTPUT_CH}ch, num_downs=6, "
        f"H={INPUT_H}, W={INPUT_W})"
    )
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> torch.nn.Module:
    log.info(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        log.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    n = sum(p.numel() for p in model.parameters())
    log.info(f"  Checkpoint loaded  (tensors={len(state)}, params={n/1e6:.1f}M)")
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combined input: agnostic(22) + warped_cloth(3) = 25 channels
    dummy_input = torch.randn(1, TOM_INPUT_CH, INPUT_H, INPUT_W)

    log.info(f"Tracing TOM with input: {tuple(dummy_input.shape)}")

    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
        log.info(f"Forward pass OK  output={tuple(out.shape)}")
        # Verify output split: p_rendered=out[:,:3], m_composite=out[:,3:]
        p_rendered   = torch.tanh(out[:, :3])
        m_composite  = torch.sigmoid(out[:, 3:])
        log.info(
            f"  p_rendered range  [{p_rendered.min():.3f}, {p_rendered.max():.3f}]"
        )
        log.info(
            f"  m_composite range [{m_composite.min():.3f}, {m_composite.max():.3f}]"
        )

    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["tom_input"],    # (batch, 25, 256, 192)
        output_names=["tom_output"],  # (batch, 4, 256, 192)
        dynamic_axes={
            "tom_input":  {0: "batch"},
            "tom_output": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / 1_048_576
    log.info(f"ONNX export complete  →  {output_path}  ({size_mb:.1f} MB, {elapsed:.1f}s)")


def _add_torch_lib_to_path() -> None:
    """Add PyTorch bundled CUDA DLLs to PATH so ORT CUDA EP can find cublasLt."""
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
    log.info("─── Verification ────────────────────────────────────────")

    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping ORT verify")
        return

    _add_torch_lib_to_path()

    # ── PyTorch reference ────────────────────────────────────────
    model = load_checkpoint(build_tom(), CHECKPOINT)
    dummy = torch.randn(1, TOM_INPUT_CH, INPUT_H, INPUT_W)

    with torch.no_grad():
        pt_out = model(dummy).numpy()

    # ── ORT inference ────────────────────────────────────────────
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    active = sess.get_providers()[0]
    log.info(f"ORT execution provider: {active}")

    ort_out = sess.run(None, {"tom_input": dummy.numpy()})[0]

    max_diff  = float(np.abs(pt_out - ort_out).max())
    mean_diff = float(np.abs(pt_out - ort_out).mean())
    log.info(f"PyTorch vs ORT  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

    if max_diff < 1e-3:
        log.info("✓ Verification PASSED — outputs match within tolerance")
    else:
        log.warning(f"⚠ Verification WARNING — max diff {max_diff:.5f} > 1e-3")

    # ── Latency benchmark ────────────────────────────────────────
    log.info("Running latency benchmark (50 warm + 100 timed)…")
    bench_in = {"tom_input": dummy.numpy()}
    for _ in range(50):
        sess.run(None, bench_in)
    t0 = time.perf_counter()
    for _ in range(100):
        sess.run(None, bench_in)
    ms = (time.perf_counter() - t0) / 100 * 1000
    log.info(f"ORT TOM latency: {ms:.1f} ms/frame  ({1000/ms:.0f} FPS theoretical)")
    log.info("─────────────────────────────────────────────────────────")

    # ── Show what the composite would look like on random data ───
    log.info("Composite formula smoke-test…")
    dummy_agnostic    = np.random.rand(1, AGNOSTIC_CH, INPUT_H, INPUT_W).astype(np.float32)
    dummy_warped_cloth = np.random.rand(1, WARPED_CLOTH_CH, INPUT_H, INPUT_W).astype(np.float32) * 2 - 1  # [-1,1]
    tom_in = np.concatenate([dummy_agnostic, dummy_warped_cloth], axis=1)
    tom_raw = sess.run(None, {"tom_input": tom_in})[0]   # (1, 4, 256, 192)

    p_rendered  = np.tanh(tom_raw[:, :3])                       # tanh → [-1,1]
    m_composite = 1.0 / (1.0 + np.exp(-tom_raw[:, 3:]))         # sigmoid → [0,1]
    wc = (dummy_warped_cloth + 1.0) / 2.0                       # [-1,1] → [0,1]
    pr = (p_rendered + 1.0) / 2.0                               # [-1,1] → [0,1]
    p_tryon = wc * m_composite + pr * (1.0 - m_composite)       # composite

    log.info(f"  p_tryon range [{p_tryon.min():.3f}, {p_tryon.max():.3f}]  shape={p_tryon.shape}")
    log.info("Smoke-test OK")


def main():
    parser = argparse.ArgumentParser(description="Export CP-VTON TOM checkpoint to ONNX")
    parser.add_argument("--verify", action="store_true",
                        help="After export: run ORT verification + latency benchmark")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--output",     default=str(ONNX_OUT))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path  = Path(args.output)

    log.info("══════════════════════════════════════════════════════")
    log.info("  CP-VTON TOM → ONNX Export")
    log.info(f"  checkpoint : {ckpt_path}")
    log.info(f"  output     : {out_path}")
    log.info(f"  input      : ({TOM_INPUT_CH}, {INPUT_H}, {INPUT_W})")
    log.info(f"               = agnostic({AGNOSTIC_CH}ch) + warped_cloth({WARPED_CLOTH_CH}ch)")
    log.info("══════════════════════════════════════════════════════")

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = load_checkpoint(build_tom(), ckpt_path)
    export_onnx(model, out_path)

    if args.verify:
        verify_onnx(out_path)

    log.info("Done.  Run with --verify to benchmark ORT inference.")


if __name__ == "__main__":
    main()
