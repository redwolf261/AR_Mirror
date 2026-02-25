#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AR Mirror -- Model Download Script
==================================
Downloads all optional models that improve garment wrapping quality.

Models downloaded
-----------------
1. SCHP-LIP (schp_lip.onnx)       -- 20-class human body parsing for
                                     occlusion-aware compositing.
                                     Enables hair/face layering on top of garment.

2. HR-VITON GMM  (hr_viton_gmm.pth) -- Superior TPS garment warping model.
   HR-VITON TOM  (hr_viton_tom.pth) -- Try-On Module for final synthesis.

3. DensePose ONNX (optional)        -- Per-pixel 3D body surface UV map.
                                      Enables true UV-space wrapping.

Usage
-----
    # Download all models:
    python scripts/download_models.py

    # Download specific model only:
    python scripts/download_models.py --schp
    python scripts/download_models.py --hrviton
    python scripts/download_models.py --densepose
"""

import argparse
import os
import sys
import urllib.request
import subprocess
from pathlib import Path

# ── make sure project root is on path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────── helpers ─────────────────────────────────────

def _progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(int(count * block_size * 100 / total_size), 100)
        bar = "#" * (pct // 4) + "-" * (25 - pct // 4)
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  ")
        sys.stdout.flush()


def _download(url: str, dest: Path, label: str) -> bool:
    """Download *url* to *dest* with a progress bar.  Returns True on success."""
    if dest.exists():
        print(f"  [OK] Already present: {dest.name}")
        return True
    print(f"  Downloading {label} ...")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\n  [OK] Saved to {dest}  ({dest.stat().st_size / 1_048_576:.1f} MB)")
        return True
    except Exception as exc:
        print(f"\n  [FAIL] Download failed: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def _hf_download(repo_id: str, filename: str, dest: Path, label: str) -> bool:
    """Download *filename* from a HuggingFace repo."""
    if dest.exists():
        print(f"  [OK] Already present: {dest.name}")
        return True
    print(f"  Downloading {label} from HuggingFace ({repo_id}/{filename}) ...")
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
        )
        if Path(local) != dest:
            import shutil
            shutil.copy2(local, dest)
        print(f"  [OK] Saved to {dest}  ({dest.stat().st_size / 1_048_576:.1f} MB)")
        return True
    except ImportError:
        print("  huggingface_hub not installed -- installing ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "huggingface_hub", "-q"])
        return _hf_download(repo_id, filename, dest, label)
    except Exception as exc:
        print(f"  [FAIL] HuggingFace download failed: {exc}")
        return False
def _gdrive_download(file_id: str, dest: Path, label: str) -> bool:
    """Download from Google Drive using gdown."""
    if dest.exists():
        print(f"  [OK] Already present: {dest.name}")
        return True
    print(f"  Downloading {label} from Google Drive ...")
    try:
        import gdown
    except ImportError:
        print("  gdown not installed -- installing ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown  # type: ignore
    try:
        gdown.download(id=file_id, output=str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  [OK] Saved to {dest}  ({dest.stat().st_size / 1_048_576:.1f} MB)")
            return True
        print(f"  [FAIL] File too small or missing after download.")
        return False
    except Exception as exc:
        print(f"  [FAIL] Google Drive download failed: {exc}")
        if dest.exists():
            dest.unlink()
        return False



def download_schp() -> bool:
    """
    Download SCHP-LIP and export to ONNX.

    Strategy:
    1. Try to get a pre-exported ONNX directly (fastest).
    2. Fall back to downloading the .pth and exporting it ourselves.
    """
    onnx_path = MODELS_DIR / "schp_lip.onnx"
    if onnx_path.exists():
        print(f"  [OK] schp_lip.onnx already present.")
        return True

    print("\n" + "=" * 60)
    print("SCHP-LIP  (Self-Correction Human Parsing)")
    print("=" * 60)

    # ── Step 1: try pre-built ONNX from HuggingFace ──────────────────────────
    # Several community conversions exist on HuggingFace.
    HF_CANDIDATES = [
        ("gokaygokay/SCHP-LIP", "schp_lip.onnx"),
        ("levihsu/schp",         "exp-schp-201908261155-lip.onnx"),
    ]
    for repo, fname in HF_CANDIDATES:
        ok = _hf_download(repo, fname, onnx_path, "SCHP-LIP ONNX")
        if ok and onnx_path.exists():
            print("  [OK] SCHP ONNX ready.")
            return True

    # ── Step 2: download .pth + convert ourselves ─────────────────────────────
    print("\n  Pre-built ONNX not found -- converting from official .pth checkpoint ...")
    pth_path = MODELS_DIR / "schp_lip.pth"
    PTH_URL = (
        "https://github.com/GoGoDuck912/Self-Correction-Human-Parsing"
        "/releases/download/v1.0/exp-schp-201908261155-lip.pth"
    )
    SCHP_GDRIVE_ID = "1k4dllHpu_3GurxkP7ue-fRiPDLVOxQBb"  # official Google Drive
    ok_pth = _download(PTH_URL, pth_path, "SCHP-LIP .pth checkpoint")
    if not ok_pth:
        ok_pth = _gdrive_download(SCHP_GDRIVE_ID, pth_path, "SCHP-LIP .pth (Drive)")
    if not ok_pth:
        print("  [FAIL] Could not obtain SCHP checkpoint -- skipping.")
        return False

    print("  Converting SCHP-LIP .pth → ONNX ...")
    # Run the project's own conversion script
    convert_script = ROOT / "scripts" / "convert_schp_to_onnx.py"
    if convert_script.exists():
        try:
            subprocess.check_call(
                [sys.executable, str(convert_script),
                 "--input",  str(pth_path),
                 "--output", str(onnx_path)],
                cwd=str(ROOT),
            )
            if onnx_path.exists():
                print(f"  [OK] ONNX saved: {onnx_path}")
                return True
        except subprocess.CalledProcessError as exc:
            print(f"  [FAIL] Conversion failed: {exc}")

    # ── Step 3: minimal inline conversion using torchvision ResNet-101 ────────
    print("  Attempting inline conversion using ResNet-101 backbone ...")
    try:
        return _inline_schp_export(pth_path, onnx_path)
    except Exception as exc:
        print(f"  [FAIL] Inline conversion failed: {exc}")
        return False


def _inline_schp_export(pth_path: Path, onnx_path: Path) -> bool:
    """Export SCHP-LIP checkpoint to ONNX using a ResNet-101 decoder architecture."""
    import torch
    import torch.nn as nn

    print("  Building SCHP-LIP architecture (ResNet-101 + ASPP decoder) ...")

    # Import SCHP architecture from the project vendor directory if available
    vendor_schp = ROOT / "vendor" / "Self-Correction-Human-Parsing"
    if vendor_schp.exists():
        sys.path.insert(0, str(vendor_schp))
        try:
            from networks import init_model  # SCHP's own network factory
            model = init_model("resnet101", num_classes=20, pretrained=None)
        except ImportError:
            model = None
    else:
        model = None

    if model is None:
        print("  SCHP vendor source not found -- cloning from GitHub ...")
        try:
            subprocess.check_call(
                ["git", "clone", "--depth=1",
                 "https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git",
                 str(vendor_schp)],
                cwd=str(ROOT),
            )
            sys.path.insert(0, str(vendor_schp))
            from networks import init_model  # type: ignore[import]
            model = init_model("resnet101", num_classes=20, pretrained=None)
        except Exception as exc:
            print(f"  [FAIL] Could not obtain SCHP architecture: {exc}")
            return False

    # Load weights
    state_dict = torch.load(str(pth_path), map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Export
    dummy = torch.zeros(1, 3, 512, 384)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"  [OK] ONNX exported to {onnx_path}")
    return True


# ─────────────────────────────── HR-VITON ────────────────────────────────────

def download_hrviton() -> bool:
    """Download HR-VITON GMM and TOM checkpoints."""
    print("\n" + "=" * 60)
    print("HR-VITON  (High-Resolution Virtual Try-On)")
    print("=" * 60)

    gmm_path = MODELS_DIR / "hr_viton_gmm.pth"
    tom_path  = MODELS_DIR / "hr_viton_tom.pth"

    # Official Google Drive IDs from the HR-VITON paper repository
    # https://github.com/sangyun884/HR-VITON
    GMM_GDRIVE_ID = "1T5_YDSOfPR6RYvW5bD3tcQe8LZVkpS7e"   # mtviton.pth
    TOM_GDRIVE_ID = "1ZPJJQKtB_-hFwGpUG2OiqGYZ8O-l7fxA"   # gen.pth

    ok_gmm = _gdrive_download(GMM_GDRIVE_ID, gmm_path, "HR-VITON GMM (mtviton.pth)")
    ok_tom = _gdrive_download(TOM_GDRIVE_ID, tom_path, "HR-VITON TOM (gen.pth)")

    if not ok_gmm or not ok_tom:
        # Try HuggingFace mirrors as fallback
        print("  Google Drive failed -- trying HuggingFace mirrors ...")
        HF_MIRRORS = [
            ("minar2020/HR-VITON",    "mtviton.pth", "gen.pth"),
            ("humanlab/HR-VITON",     "mtviton.pth", "gen.pth"),
            ("kabachuha/HR-VITON",    "mtviton.pth", "gen.pth"),
        ]
        for repo, gmm_f, tom_f in HF_MIRRORS:
            if not ok_gmm:
                ok_gmm = _hf_download(repo, gmm_f, gmm_path, f"HR-VITON GMM ({repo})")
            if not ok_tom:
                ok_tom = _hf_download(repo, tom_f, tom_path, f"HR-VITON TOM ({repo})")
            if ok_gmm and ok_tom:
                break

    if not ok_gmm or not ok_tom:
        print("\n  [NOTE] Could not download HR-VITON automatically.")
        print("  Manual download:")
        print("    1. GMM: https://drive.google.com/file/d/1T5_YDSOfPR6RYvW5bD3tcQe8LZVkpS7e")
        print(f"       Save as: {gmm_path}")
        print("    2. TOM: https://drive.google.com/file/d/1ZPJJQKtB_-hFwGpUG2OiqGYZ8O-l7fxA")
        print(f"       Save as: {tom_path}")

    return ok_gmm and ok_tom


# ─────────────────────────────── DensePose ONNX ──────────────────────────────

def download_densepose() -> bool:
    """Download a lightweight DensePose ONNX checkpoint."""
    print("\n" + "=" * 60)
    print("DensePose  (per-pixel 3D body UV -- optional)")
    print("=" * 60)

    dp_path = MODELS_DIR / "densepose_rcnn_R_50_FPN.onnx"

    # Community ONNX export of DensePose R-50-FPN
    ok = _hf_download(
        "facebook/densepose",
        "densepose_rcnn_R_50_FPN_s1x.onnx",
        dp_path,
        "DensePose R-50-FPN ONNX",
    )
    if not ok:
        print("  DensePose ONNX not available on HuggingFace yet.")
        print("  You can build it yourself with:")
        print("    pip install detectron2")
        print("    python scripts/export_densepose_onnx.py")
    return ok


# ─────────────────────────────── dependencies ────────────────────────────────

def install_dependencies():
    """Install Python packages needed by the new features."""
    pkgs = [
        "huggingface_hub",
        "trimesh",           # 3D mesh rendering (WrappedGarmentMesh)
        "diffusers",         # CatVTON pre-warper
        "transformers",      # CatVTON
        "accelerate",        # CatVTON (model cpu offload)
    ]
    print("\n" + "=" * 60)
    print("Installing / verifying Python dependencies")
    print("=" * 60)
    for pkg in pkgs:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  Installing {pkg} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
            )
            print(f"  [OK] {pkg} installed")


# ─────────────────────────────── main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download AR Mirror wrapping models")
    parser.add_argument("--schp",       action="store_true", help="Download SCHP-LIP only")
    parser.add_argument("--hrviton",    action="store_true", help="Download HR-VITON only")
    parser.add_argument("--densepose",  action="store_true", help="Download DensePose only")
    parser.add_argument("--deps",       action="store_true", help="Install dependencies only")
    parser.add_argument("--all",        action="store_true", help="Download everything (default)")
    args = parser.parse_args()

    # Default = all
    run_all = args.all or not (args.schp or args.hrviton or args.densepose or args.deps)

    results = {}

    if run_all or args.deps:
        install_dependencies()

    if run_all or args.schp:
        results["SCHP-LIP"]    = download_schp()

    if run_all or args.hrviton:
        results["HR-VITON"]    = download_hrviton()

    if run_all or args.densepose:
        results["DensePose"]   = download_densepose()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "[OK]  ready" if ok else "[FAIL]  failed / unavailable"
        print(f"  {name:20s}  {status}")

    have_schp    = (MODELS_DIR / "schp_lip.onnx").exists()
    have_gmm     = (MODELS_DIR / "hr_viton_gmm.pth").exists()
    have_tom     = (MODELS_DIR / "hr_viton_tom.pth").exists()

    print()
    if have_schp and have_gmm and have_tom:
        print("  All primary models ready -- garment wrapping is fully activated.")
    else:
        missing = []
        if not have_schp: missing.append("schp_lip.onnx")
        if not have_gmm:  missing.append("hr_viton_gmm.pth")
        if not have_tom:  missing.append("hr_viton_tom.pth")
        print(f"  Missing: {', '.join(missing)}")
        print("  The app will still run using the improved geometric fallback.")
    print()


if __name__ == "__main__":
    main()
