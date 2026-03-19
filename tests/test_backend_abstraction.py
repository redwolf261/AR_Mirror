"""Backend boundary checks for the current AR Mirror architecture."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent.parent


def test_legacy_semantic_backend_files_are_removed() -> None:
    assert not (ROOT / "src" / "core" / "semantic_parser.py").exists()
    assert not (ROOT / "src" / "core" / "parsing_backends.py").exists()


def test_backend_package_scripts_cover_dev_runtime() -> None:
    package_json = ROOT / "backend" / "package.json"
    data = json.loads(package_json.read_text(encoding="utf-8"))
    scripts = data.get("scripts", {})

    assert "start" in scripts
    assert "start:dev" in scripts
    assert scripts["start:dev"].strip() != ""


def test_web_server_route_contract_is_present() -> None:
    web_server = (ROOT / "web_server.py").read_text(encoding="utf-8")

    assert "@app.route(\"/stream\")" in web_server
    assert "@app.route(\"/api/state\")" in web_server
    assert "@app.route(\"/api/params\", methods=[\"GET\"])" in web_server
    assert "@app.route(\"/api/garments\")" in web_server
    assert "@app.route(\"/api/garment\", methods=[\"POST\"])" in web_server


def test_app_runtime_uses_web_server_wrapper() -> None:
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "from web_server import WebServer" in app_text
    assert "self.semantic_parser = None" in app_text
