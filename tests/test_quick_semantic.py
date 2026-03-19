"""Quick checks for semantic-stack deprecation in the current runtime."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_legacy_semantic_modules_are_absent() -> None:
    assert not (ROOT / "src" / "core" / "semantic_parser.py").exists()
    assert not (ROOT / "src" / "core" / "parsing_backends.py").exists()


def test_rendering_module_documents_semantic_removal() -> None:
    rendering = (ROOT / "src" / "app" / "rendering.py").read_text(encoding="utf-8")

    assert "SemanticParser removed" in rendering


def test_app_keeps_semantic_parser_placeholder_without_instantiation() -> None:
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "self.semantic_parser = None" in app_text
    assert "SemanticParser(" not in app_text
