#!/usr/bin/env python3
"""
GMM ONNX Validation on VITON Test Set
======================================
Loads dataset/test_pairs.txt, builds the 22-channel GMM agnostic input from
OpenPose JSON + parse-agnostic mask, runs gmm_model.onnx, and computes:
  - SSIM  (structural similarity vs. target cloth)
  - L1    (pixel distance)
  - warp coverage (fraction of cloth pixels transferred)

Saves N visual result grids to output/figures/gmm_validation/

Usage (paste into terminal, no timeout):
    .\\ar\\Scripts\\python.exe scripts\\validate_gmm.py
    .\\ar\\Scripts\\python.exe scripts\\validate_gmm.py --pairs 200 --save-grids 20
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows stdout (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── logging ──────────────────────────────────────────────────────────────────
log_dir = ROOT / "output" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(stream=open(os.devnull, "w", encoding="utf-8")
                              if False else __import__("sys").stdout),
        logging.FileHandler(log_dir / "gmm_validation.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("validate_gmm")

# ── paths ─────────────────────────────────────────────────────────────────────
TEST_DIR   = ROOT / "dataset" / "test"
PAIRS_TXT  = ROOT / "dataset" / "test_pairs.txt"
ONNX_PATH  = ROOT / "models" / "gmm_model.onnx"
FIG_DIR    = ROOT / "output" / "figures" / "gmm_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# GMM dimensions
GMM_H, GMM_W = 256, 192
# Original dataset resolution
DATA_H, DATA_W = 1024, 768


# ── agnostic assembly ─────────────────────────────────────────────────────────

def _gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float = 4.0) -> np.ndarray:
    """Single Gaussian blob centred at (cx, cy) in a (h,w) map."""
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    hm = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)


# COCO-25 keypoint index → body part grouping used by cp-vton
# 18-channel pose heatmaps: joints 0-17 (COCO subset without feet)
_COCO25_TO_POSE18 = list(range(18))  # joints 0-17 map 1-to-1

def build_agnostic(
    openpose_json: Path,
    parse_agnostic_png: Path,
    agnostic_jpg: Path,
) -> np.ndarray:
    """
    Assemble 22-channel GMM agnostic at (GMM_H, GMM_W):
      channels  0-17 : Gaussian pose heatmaps (18 joints)
      channel  18    : body shape (torso pixel mask from parse-agnostic)
      channels 19-21 : head RGB crop from agnostic masked image

    Returns float32 array (22, GMM_H, GMM_W), values in [-1, 1].
    """
    # ── pose heatmaps ──────────────────────────────────────────────────────
    with open(openpose_json) as f:
        kp_data = json.load(f)

    heatmaps = np.zeros((18, GMM_H, GMM_W), dtype=np.float32)
    if kp_data.get("people"):
        raw = kp_data["people"][0]["pose_keypoints_2d"]  # flat [x,y,c, ...]
        for i, dst in enumerate(_COCO25_TO_POSE18):
            x_norm = raw[i * 3]     / DATA_W  # normalised [0,1]
            y_norm = raw[i * 3 + 1] / DATA_H
            conf   = raw[i * 3 + 2]
            if conf < 0.1:
                continue
            cx = x_norm * GMM_W
            cy = y_norm * GMM_H
            heatmaps[dst] = _gaussian_heatmap(GMM_H, GMM_W, cx, cy, sigma=4.0)

    # ── body shape (channel 18) ────────────────────────────────────────────
    parse = cv2.imread(str(parse_agnostic_png))  # (1024,768,3) uint8
    # torso labels in parse-agnostic-v3.2: values 85 and 220 (upper/lower body)
    gray = parse[:, :, 0]
    body = np.isin(gray, [85, 220]).astype(np.float32)
    body_small = cv2.resize(body, (GMM_W, GMM_H), interpolation=cv2.INTER_AREA)
    body_small = body_small * 2.0 - 1.0  # [0,1] → [-1,1]

    # ── head (channels 19-21) ─────────────────────────────────────────────
    agnostic_img = cv2.imread(str(agnostic_jpg))  # (1024,768,3) BGR uint8
    agnostic_rgb = cv2.cvtColor(agnostic_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    agnostic_small = cv2.resize(agnostic_rgb, (GMM_W, GMM_H), interpolation=cv2.INTER_AREA)
    # Isolate head only (top 25% of frame, zeroing torso area)
    head_region = np.zeros_like(agnostic_small)
    head_h = int(GMM_H * 0.25)
    head_region[:head_h] = agnostic_small[:head_h]
    head_ch = head_region.transpose(2, 0, 1)  # (3,H,W)

    # ── stack ──────────────────────────────────────────────────────────────
    agnostic = np.concatenate([
        heatmaps,                          # (18, H, W)
        body_small[np.newaxis],            # ( 1, H, W)
        head_ch,                           # ( 3, H, W)
    ], axis=0)  # (22, H, W)

    return agnostic.astype(np.float32)


# ── metrics ───────────────────────────────────────────────────────────────────

def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Fast SSIM approximation without scikit-image dependency."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a  = np.sqrt(((a - mu_a) ** 2).mean())
    sigma_b  = np.sqrt(((b - mu_b) ** 2).mean())
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2) / \
           ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a ** 2 + sigma_b ** 2 + C2))
    return float(np.clip(ssim, -1.0, 1.0))


def warp_coverage(warped_mask: np.ndarray, cloth_mask: np.ndarray) -> float:
    """Fraction of cloth pixels successfully transferred (IoU-style)."""
    w = (warped_mask > 0.3).astype(np.float32)
    c = (cloth_mask  > 0.3).astype(np.float32)
    inter = (w * c).sum()
    union = np.clip(c.sum(), 1.0, None)
    return float(inter / union)


# ── visualisation ─────────────────────────────────────────────────────────────

def make_grid(person_img, cloth_img, warped_cloth, warped_mask, pair_name: str) -> np.ndarray:
    """
    5-panel grid: person | cloth | warped_cloth | warped_mask | overlay
    All panels at GMM resolution (256×192).
    """
    H, W = GMM_H, GMM_W

    def to_u8(x):
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    person_small = cv2.resize(person_img, (W, H), interpolation=cv2.INTER_AREA)
    cloth_small  = cv2.resize(cloth_img,  (W, H), interpolation=cv2.INTER_AREA)

    mask_3ch = np.stack([warped_mask] * 3, axis=-1)
    person_f = person_small.astype(np.float32) / 255.0 if person_small.dtype == np.uint8 else person_small
    overlay  = person_f * (1.0 - mask_3ch) + warped_cloth * mask_3ch

    panels = [
        person_small,
        cloth_small,
        to_u8(warped_cloth),
        to_u8(np.stack([warped_mask] * 3, axis=-1)),
        to_u8(overlay),
    ]
    grid = np.concatenate(panels, axis=1)

    # Add label strip
    label_strip = np.zeros((24, grid.shape[1], 3), dtype=np.uint8)
    labels = ["person", "cloth", "warped", "mask", "overlay"]
    for i, lbl in enumerate(labels):
        cv2.putText(label_strip, lbl, (i * W + 4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(label_strip, pair_name, (2, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 220, 80), 1)

    return np.concatenate([grid, label_strip], axis=0)


# ── ORT session ───────────────────────────────────────────────────────────────

def load_ort_session():
    os.environ["PATH"] = (
        str(Path(__import__("torch").__file__).parent / "lib")
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    ep = sess.get_providers()[0]
    log.info(f"ORT session ready  EP={ep}")
    return sess


def gmm_infer(sess, agnostic: np.ndarray, cloth: np.ndarray, cloth_mask: np.ndarray) -> tuple:
    """Run one GMM inference.

    ONNX I/O (verified):
      inputs : agnostic (1,22,H,W), cloth_mask (1,1,H,W)
      outputs: grid (1,H,W,2) in [-1,1]

    Returns (warped_cloth, grid_2hw) where grid_2hw is (2,H,W) for downstream remap.
    """
    H, W = GMM_H, GMM_W
    cloth_r  = cv2.resize(cloth,      (W, H), interpolation=cv2.INTER_LINEAR)
    mask_r   = cv2.resize(cloth_mask, (W, H), interpolation=cv2.INTER_AREA)
    mask_r   = mask_r if mask_r.ndim == 2 else mask_r[:, :, 0]

    agnostic_in   = agnostic[np.newaxis].astype(np.float32)              # (1,22,H,W)
    cloth_mask_in = mask_r[np.newaxis, np.newaxis].astype(np.float32)    # (1,1,H,W)

    # output: grid (1,H,W,2) in [-1,1] — already at full GMM resolution
    grid = sess.run(["grid"], {"agnostic": agnostic_in, "cloth_mask": cloth_mask_in})[0]  # (1,H,W,2)

    map_x = ((grid[0, :, :, 0] + 1.0) * 0.5 * (W - 1)).astype(np.float32)
    map_y = ((grid[0, :, :, 1] + 1.0) * 0.5 * (H - 1)).astype(np.float32)

    warped_cloth = cv2.remap(cloth_r, map_x, map_y,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    # Pack maps as (2,H,W) so caller can remap the mask with the same grid
    grid_2hw = np.stack([map_x, map_y], axis=0)
    return np.clip(warped_cloth, 0.0, 1.0), grid_2hw  # grid_2hw: (2,H,W) as pixel coords


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate GMM ONNX on VITON test set")
    parser.add_argument("--pairs",      type=int, default=500, help="Number of pairs to evaluate (default: 500)")
    parser.add_argument("--save-grids", type=int, default=30,  help="Number of visual grids to save (default: 30)")
    parser.add_argument("--batch",      type=int, default=1,   help="ORT batch size (default: 1)")
    args = parser.parse_args()

    if not ONNX_PATH.exists():
        log.error(f"ONNX model not found: {ONNX_PATH}")
        log.error("Run: .\\ar\\Scripts\\python.exe scripts\\export_gmm_to_onnx.py --verify")
        sys.exit(1)

    # ── load pairs ────────────────────────────────────────────────────────
    pairs = []
    with open(PAIRS_TXT) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    pairs = pairs[: args.pairs]
    log.info("============================================================")
    log.info(f"Evaluating {len(pairs)} pairs  (save_grids={args.save_grids})")
    log.info("============================================================")

    # ── ORT session ───────────────────────────────────────────────────────
    sess = load_ort_session()

    # ── per-pair evaluation ───────────────────────────────────────────────
    ssim_scores, l1_scores, cov_scores = [], [], []
    times = []
    grids_saved = 0
    skipped = 0

    for idx, (person_id, cloth_id) in enumerate(pairs):
        # Derive filenames (strip extension if present, force .jpg / .png)
        pid = person_id.replace(".jpg", "").replace(".png", "")
        cid = cloth_id.replace(".jpg", "").replace(".png", "")

        json_path    = TEST_DIR / "openpose_json"           / f"{pid}_keypoints.json"
        parse_path   = TEST_DIR / "image-parse-agnostic-v3.2" / f"{pid}.png"
        ag_path      = TEST_DIR / "agnostic-v3.2"          / f"{pid}.jpg"
        cloth_path   = TEST_DIR / "cloth"                   / f"{cid}.jpg"
        cmask_path   = TEST_DIR / "cloth-mask"              / f"{cid}.jpg"
        person_path  = TEST_DIR / "image"                   / f"{pid}.jpg"

        # Skip missing pairs gracefully
        for p in [json_path, parse_path, ag_path, cloth_path, cmask_path, person_path]:
            if not p.exists():
                log.warning(f"  [{idx+1:4d}] SKIP — missing: {p.name}")
                skipped += 1
                break
        else:
            pass
        missing = any(not p.exists() for p in [json_path, parse_path, ag_path, cloth_path, cmask_path, person_path])
        if missing:
            continue

        try:
            t0 = time.perf_counter()

            # Build agnostic
            agnostic = build_agnostic(json_path, parse_path, ag_path)

            # Load cloth in [0,1] RGB
            cloth_bgr = cv2.imread(str(cloth_path))
            cloth_rgb = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            cloth_mask_raw = cv2.imread(str(cmask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            cloth_mask = (cloth_mask_raw > 127).astype(np.float32)
            cloth_mask_small = cv2.resize(cloth_mask, (GMM_W, GMM_H), interpolation=cv2.INTER_AREA)

            # GMM inference
            warped_cloth, grid_2hw = gmm_infer(sess, agnostic, cloth_rgb, cloth_mask)  # grid_2hw: (2,GMM_H,GMM_W)

            # Warp cloth mask through same grid (already pixel coords)
            map_x_f = grid_2hw[0]
            map_y_f = grid_2hw[1]
            warped_mask = cv2.remap(cloth_mask_small, map_x_f, map_y_f,
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

            # ── metrics (compare warped cloth vs. target cloth at GMM res) ──
            cloth_small = cv2.resize(cloth_rgb, (GMM_W, GMM_H), interpolation=cv2.INTER_AREA)
            s = ssim_simple(warped_cloth, cloth_small)
            l = float(np.abs(warped_cloth - cloth_small).mean())
            c = warp_coverage(warped_mask, cloth_mask_small)

            ssim_scores.append(s)
            l1_scores.append(l)
            cov_scores.append(c)

            # ── per-pair log (every 50 pairs) ──────────────────────────────
            if (idx + 1) % 50 == 0 or idx == 0:
                log.info(
                    f"  [{idx+1:4d}/{len(pairs)}]  "
                    f"SSIM={np.mean(ssim_scores):.4f}  "
                    f"L1={np.mean(l1_scores):.4f}  "
                    f"Cov={np.mean(cov_scores):.3f}  "
                    f"t={np.mean(times):.1f}ms"
                )

            # ── save grid ──────────────────────────────────────────────────
            if grids_saved < args.save_grids:
                person_bgr = cv2.imread(str(person_path))
                person_rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                grid = make_grid(person_rgb, cloth_rgb, warped_cloth, warped_mask,
                                 f"{pid}+{cid}")
                out_path = FIG_DIR / f"val_{idx:04d}_{pid}.jpg"
                cv2.imwrite(str(out_path), cv2.cvtColor((grid * 255.0).astype(np.uint8) if grid.dtype != np.uint8 else grid, cv2.COLOR_RGB2BGR))
                grids_saved += 1

        except Exception as e:
            log.warning(f"  [{idx+1:4d}] ERROR {pid}: {e}")
            skipped += 1
            continue

    # ── final report ──────────────────────────────────────────────────────
    n = len(ssim_scores)
    log.info("=" * 60)
    log.info("  GMM VALIDATION RESULTS")
    log.info("=" * 60)
    log.info(f"  Pairs evaluated : {n}  (skipped: {skipped})")
    log.info(f"  SSIM            : {np.mean(ssim_scores):.4f}  +/- {np.std(ssim_scores):.4f}")
    log.info(f"  L1  distance    : {np.mean(l1_scores):.4f}  +/- {np.std(l1_scores):.4f}")
    log.info(f"  Warp coverage   : {np.mean(cov_scores):.4f}  +/- {np.std(cov_scores):.4f}")
    log.info(f"  Avg latency     : {np.mean(times):.1f} ms/pair")
    log.info(f"  Grids saved     : {grids_saved}  ->  {FIG_DIR}")
    log.info("=" * 60)

    # ── save numeric results as JSON ──────────────────────────────────────
    import json as _json
    results = {
        "n_pairs": n,
        "skipped": skipped,
        "ssim":  {"mean": float(np.mean(ssim_scores)),  "std": float(np.std(ssim_scores))},
        "l1":    {"mean": float(np.mean(l1_scores)),    "std": float(np.std(l1_scores))},
        "coverage": {"mean": float(np.mean(cov_scores)), "std": float(np.std(cov_scores))},
        "latency_ms": {"mean": float(np.mean(times)), "std": float(np.std(times))},
    }
    out_json = ROOT / "output" / "logs" / "gmm_validation_results.json"
    with open(out_json, "w") as f:
        _json.dump(results, f, indent=2)
    log.info(f"  Metrics saved   : {out_json}")


if __name__ == "__main__":
    main()
