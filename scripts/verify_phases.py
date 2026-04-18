"""Quick verification that all phase backends are in place."""
import pathlib, importlib, json, subprocess, sys

print("=== PHASE BACKEND VERIFICATION ===\n")

# Phase A – ESRGAN
try:
    # torchvision ≥ 0.17 removed functional_tensor; apply compat shim before basicsr import
    import sys, types as _types
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        import torchvision.transforms.functional as _TF
        _ft = _types.ModuleType("torchvision.transforms.functional_tensor")
        for _n in dir(_TF):
            if not _n.startswith("__"):
                setattr(_ft, _n, getattr(_TF, _n))
        sys.modules["torchvision.transforms.functional_tensor"] = _ft
    import realesrgan, basicsr  # noqa: F401
    print("Phase A ESRGAN:     OK  – realesrgan + basicsr installed")
except Exception as e:
    print(f"Phase A ESRGAN:     FAIL – {e}")

weights = pathlib.Path("models/RealESRGAN_x4plus.pth")
if weights.exists():
    print(f"  weights:          EXISTS ({round(weights.stat().st_size/1e6)}MB)")
else:
    print("  weights:          not yet downloaded (auto-downloads on first garment load)")

# Phase B – SMPL vert segmentation
seg_path = pathlib.Path("models/smpl_vert_segmentation.json")
if seg_path.exists():
    seg = json.loads(seg_path.read_text())
    total = sum(len(v) for v in seg.values())
    print(f"Phase B SMPL seg:   OK  – {len(seg)} parts, {total} vertices ({seg_path.stat().st_size} bytes)")
else:
    print("Phase B SMPL seg:   MISS – models/smpl_vert_segmentation.json not found")

# Phase C – RVM ONNX + onnxruntime
rvm_path = pathlib.Path("models/rvm_mobilenetv3_fp32.onnx")
if rvm_path.exists():
    print(f"Phase C RVM model:  OK  – {round(rvm_path.stat().st_size/1e6, 1)} MB")
else:
    print("Phase C RVM model:  MISS")

try:
    import onnxruntime as ort  # noqa: F401
    eps = [e for e in ort.get_available_providers() if "CUDA" in e or e == "CPUExecutionProvider"]
    print(f"  onnxruntime:      {ort.__version__}  providers: {eps}")
except Exception as e:
    print(f"  onnxruntime:      FAIL – {e}")

# Phase D – AdaptiveEMA (pure numpy, no deps)
print("Phase D AdaptiveEMA: OK  – pure numpy, always active")

# Phase E – OOTDiffusion
if importlib.util.find_spec("diffusers"):
    import diffusers  # noqa: F401
    print(f"Phase E OOTDiff:    OK  – diffusers {diffusers.__version__}")  # type: ignore[attr-defined]
else:
    print("Phase E OOTDiff:    SKIP – diffusers not installed (optional, 6 GB model)")

print()
print("=== DELETED MODULES CHECK ===")
root = pathlib.Path(__file__).resolve().parents[1]
dead_paths = {
    "src/core/semantic_parser.py": "src.core.semantic_parser",
    "src/core/catvton_prewarper.py": "src.core.catvton_prewarper",
    "src/core/densepose_converter.py": "src.core.densepose_converter",
    "src/core/smplx_body_reconstruction.py": "src.core.smplx_body_reconstruction",
    "src/core/parsing_backends.py": "src.core.parsing_backends",
    "src/pipelines/diffusion_renderer.py": "src.pipelines.diffusion_renderer",
}
for rel, module in dead_paths.items():
    exists = (root / rel).exists()
    print(f"  {'STILL PRESENT' if exists else 'GONE':15s}: {module}")

print()
print("=== SYNTAX CHECK ===")
targets = ["app.py", "src/app/rendering.py", "src/pipelines/phase2_neural_pipeline.py",
           "src/core/esrgan_upscaler.py", "src/core/rvm_matting.py"]
r = subprocess.run([sys.executable, "-m", "py_compile"] + targets, capture_output=True, text=True)
if r.returncode == 0:
    print("  All files: syntax OK")
else:
    print("  ERRORS:\n" + r.stderr)
