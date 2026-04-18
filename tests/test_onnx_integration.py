"""ONNX integration smoke checks for dependency and tooling contracts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _read_text_any(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def test_onnx_runtime_dependency_is_declared() -> None:
    root_reqs = _read_text_any(ROOT / "requirements.txt").lower()
    ar_reqs = _read_text_any(ROOT / "requirements-ar.txt").lower()
    ml_reqs = _read_text_any(ROOT / "python-ml" / "requirements.txt").lower()

    assert "onnxruntime" in root_reqs
    assert "onnxruntime" in ar_reqs
    assert "onnxruntime" in ml_reqs


def test_onnx_export_script_has_required_entrypoints() -> None:
    export_script = (ROOT / "scripts" / "export_tom_to_onnx.py").read_text(encoding="utf-8")

    assert "def export_onnx(" in export_script
    assert "def verify_onnx(" in export_script
    assert "torch.onnx.export(" in export_script
    assert "if __name__ == \"__main__\":" in export_script


def test_runtime_task_assets_exist() -> None:
    assert (ROOT / "pose_landmarker_lite.task").exists()
    assert (ROOT / "hand_landmarker.task").exists()
