"""Repository-native unit checks for garment inventory integrity."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "config" / "garment_inventory.json"


def _load_inventory() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def test_inventory_file_exists() -> None:
    assert INVENTORY.exists(), "Missing config/garment_inventory.json"


def test_inventory_has_garments() -> None:
    data = _load_inventory()
    garments = data.get("garments", [])
    assert isinstance(garments, list)
    assert len(garments) > 0


def test_garment_schema_is_valid() -> None:
    garments = _load_inventory()["garments"]

    for garment in garments:
        assert garment.get("sku")
        assert garment.get("name")
        assert garment.get("type")
        assert garment.get("category")

        color = garment.get("color")
        assert isinstance(color, list)
        assert len(color) == 3
        assert all(isinstance(c, int) and 0 <= c <= 255 for c in color)

        sizes = garment.get("sizes")
        assert isinstance(sizes, list)
        assert len(sizes) > 0

        price = garment.get("price")
        assert isinstance(price, (int, float))
        assert float(price) > 0


def test_sku_values_are_unique() -> None:
    garments = _load_inventory()["garments"]
    skus = [g["sku"] for g in garments]
    assert len(skus) == len(set(skus)), "Duplicate SKU values found in inventory"
