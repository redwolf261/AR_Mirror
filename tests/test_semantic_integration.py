"""Integration checks after semantic parser deprecation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_src_core_init_declares_core_exports() -> None:
    text = (ROOT / "src" / "core" / "__init__.py").read_text(encoding="utf-8")

    assert "'DepthEstimator'" in text
    assert "'FrameSynchronizer'" in text


def test_core_init_keeps_optional_semantic_import_guarded() -> None:
    text = (ROOT / "src" / "core" / "__init__.py").read_text(encoding="utf-8")

    assert "from .semantic_parser import" in text
    assert "except ImportError" in text


def test_removed_semantic_files_stay_absent() -> None:
    assert not (ROOT / "src" / "core" / "semantic_parser.py").exists()
    assert not (ROOT / "src" / "core" / "parsing_backends.py").exists()


def test_app_keeps_semantic_placeholder_only() -> None:
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "self.semantic_parser = None" in app_text
    assert "from src.core.semantic_parser import" not in app_text
