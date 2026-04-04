import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import trimesh
    from trimesh.visual.texture import TextureVisuals, SimpleMaterial
except Exception as exc:
    raise SystemExit(f"trimesh unavailable: {exc}")

ROOT = Path(__file__).resolve().parents[1]


def build_glb(src_image: Path, out_glb: Path) -> None:
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_image).convert("RGBA")

    verts = np.array(
        [
            [-0.5, 0.8, 0.0],
            [0.5, 0.8, 0.0],
            [0.5, -0.8, 0.0],
            [-0.5, -0.8, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    material = SimpleMaterial(image=img)
    visual = TextureVisuals(uv=uv, image=img, material=material)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)
    mesh.export(out_glb)


def ensure_inventory_entries() -> None:
    inv_path = ROOT / "assets" / "garments" / "garment_inventory.json"
    inventory = json.loads(inv_path.read_text(encoding="utf-8"))

    template_sizes = {
        "S": {"shoulder_cm": 42.0, "chest_cm": 46.0, "length_cm": 62.0, "ease_shoulder": 2.0, "ease_chest": 4.0},
        "M": {"shoulder_cm": 44.0, "chest_cm": 50.0, "length_cm": 65.0, "ease_shoulder": 2.0, "ease_chest": 4.0},
        "L": {"shoulder_cm": 46.0, "chest_cm": 54.0, "length_cm": 68.0, "ease_shoulder": 2.0, "ease_chest": 4.0},
        "XL": {"shoulder_cm": 48.0, "chest_cm": 58.0, "length_cm": 70.0, "ease_shoulder": 2.0, "ease_chest": 4.0},
    }

    new_entries = [
        {
            "sku": "GLB-001",
            "brand": "ARMirror3D",
            "name": "GLB Classic Tee (Baked)",
            "category": "t-shirt",
            "color": "Black",
            "price_inr": 999,
            "available_sizes": ["S", "M", "L", "XL"],
            "sizes": template_sizes,
            "image_path": "garment_assets/GLB-001",
        },
        {
            "sku": "GLB-002",
            "brand": "ARMirror3D",
            "name": "GLB Studio Tee (Baked)",
            "category": "t-shirt",
            "color": "Black",
            "price_inr": 999,
            "available_sizes": ["S", "M", "L", "XL"],
            "sizes": template_sizes,
            "image_path": "garment_assets/GLB-002",
        },
    ]

    existing = {item.get("sku") for item in inventory if isinstance(item, dict)}
    changed = False
    for entry in new_entries:
        if entry["sku"] not in existing:
            inventory.append(entry)
            changed = True

    if changed:
        inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")


def main() -> None:
    build_glb(
        ROOT / "garment_assets" / "TSH-001" / "14274_00.jpg",
        ROOT / "garment_assets" / "GLB-001" / "model.glb",
    )
    build_glb(
        ROOT / "garment_assets" / "TSH-001" / "image.png",
        ROOT / "garment_assets" / "GLB-002" / "model.glb",
    )
    ensure_inventory_entries()
    print("ok: generated GLB-001 and GLB-002")


if __name__ == "__main__":
    main()
