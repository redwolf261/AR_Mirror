"""Deterministic ONNX validation checks for export and dependency readiness."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_tom_export_script_references_expected_onnx_output() -> None:
    script = (ROOT / "scripts" / "export_tom_to_onnx.py").read_text(encoding="utf-8")

    assert "tom_model.onnx" in script
    assert "CHECKPOINT" in script
    assert "ONNX_OUT" in script


def test_tom_export_script_supports_verify_flag() -> None:
    script = (ROOT / "scripts" / "export_tom_to_onnx.py").read_text(encoding="utf-8")

    assert "--verify" in script
    assert "def verify_onnx(" in script
    assert "onnxruntime" in script


def test_python_ml_requirements_include_onnx_stack() -> None:
    reqs = (ROOT / "python-ml" / "requirements.txt").read_text(encoding="utf-8").lower()

    assert "onnx" in reqs
    assert "onnxruntime" in reqs
