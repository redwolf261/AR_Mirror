"""Repository benchmark-harness smoke tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_stress_harness_files_exist() -> None:
    assert (ROOT / "tests" / "stress_test_pipeline.py").exists()
    assert (ROOT / "tests" / "stress" / "stress_test_production.py").exists()


def test_validation_report_contains_performance_terms() -> None:
    report = (ROOT / "VALIDATION_RESULTS.md").read_text(encoding="utf-8")

    assert "Performance" in report
    assert "FPS" in report


def test_auto_calibrator_uses_cma_optimizer_not_legacy_learning_rate() -> None:
    calibrator = (ROOT / "auto_calibrator.py").read_text(encoding="utf-8")

    assert "class _CMAOptimizer" in calibrator
    assert "_LEARN_RATE" not in calibrator


def test_python_ml_pytest_config_exists() -> None:
    assert (ROOT / "python-ml" / "pytest.ini").exists()
