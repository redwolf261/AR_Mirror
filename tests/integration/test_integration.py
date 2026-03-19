"""Repository integration smoke tests for current AR Mirror architecture."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]


def test_core_runtime_files_exist() -> None:
    required = [
        ROOT / "app.py",
        ROOT / "web_server.py",
        ROOT / "docker-compose.dev.yml",
        ROOT / "backend" / "package.json",
        ROOT / "web-ui" / "src" / "App.tsx",
        ROOT / "web-ui" / "src" / "main.tsx",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    assert not missing, f"Missing required runtime files: {missing}"


def test_backend_dev_script_is_available() -> None:
    package_json = ROOT / "backend" / "package.json"
    data = json.loads(package_json.read_text(encoding="utf-8"))
    scripts = data.get("scripts", {})

    assert "start:dev" in scripts
    assert scripts["start:dev"].strip() != ""


def test_dev_compose_uses_backend_start_dev() -> None:
    compose_text = (ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")
    assert "npm" in compose_text
    assert "start:dev" in compose_text


def test_runtime_phase_cli_is_limited_to_supported_modes() -> None:
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "choices=[0, 2]" in app_text
    assert "Phase 1 is deprecated" in app_text


def test_web_ui_ts_source_of_truth() -> None:
    assert (ROOT / "web-ui" / "src" / "App.tsx").exists()
    assert (ROOT / "web-ui" / "src" / "main.tsx").exists()
    assert not (ROOT / "web-ui" / "src" / "App.js").exists()
    assert not (ROOT / "web-ui" / "src" / "main.js").exists()
