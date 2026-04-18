"""
render.py — Canonical stick-figure renderer.

Converts body-33 keypoints → grayscale image.

Design goals:
  1. Output is structurally similar to a dilated binary segmentation mask, so
     fine-tuning on real photos only requires head/classifier adaptation.
  2. Deterministic: same kp array → same image every time.
  3. Fast: pure NumPy/PIL, no GPU needed (called in DataLoader workers).

Output: np.ndarray [1, size, size] float32, values in [0.0, 1.0].
        Channel-first for direct use as a PyTorch single-channel tensor.

Usage:
    from fitengine.render import render_body_image, measure_min_torso_px
    img = render_body_image(kp33, size=128)   # [1,128,128] float32
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

# Body-33 skeleton: (joint_a, joint_b)
# Matches visualize.py exactly so drawing is consistent.
_SKELETON = [
    (0,  27),  # nose → neck
    (27, 5),   # neck → left_shoulder
    (27, 6),   # neck → right_shoulder
    (5,  7),   # l_shoulder → l_elbow
    (7,  9),   # l_elbow → l_wrist
    (6,  8),   # r_shoulder → r_elbow
    (8,  10),  # r_elbow → r_wrist
    (5,  11),  # l_shoulder → l_hip
    (6,  12),  # r_shoulder → r_hip
    (11, 12),  # l_hip → r_hip
    (11, 13),  # l_hip → l_knee
    (13, 15),  # l_knee → l_ankle
    (12, 14),  # r_hip → r_knee
    (14, 16),  # r_knee → r_ankle
    (32, 11),  # mid_hip → l_hip
    (32, 12),  # mid_hip → r_hip
]

# Thickness as fraction of image width.
# At 128px → ~8px (torso) / ~5px (limbs).
_TORSO_BONES  = {(5, 11), (6, 12), (11, 12), (27, 5), (27, 6), (32, 11), (32, 12)}
_TORSO_THICK  = 0.060   # fraction of image width
_LIMB_THICK   = 0.038

_CONF_MIN     = 0.25
_HEAD_RADIUS  = 0.055   # fraction of image width


def render_body_image(
    kp33:      np.ndarray,          # [33, 3] float32, (x, y, conf) normalised [0,1]
    size:      int = 128,
    conf_min:  float = _CONF_MIN,
) -> np.ndarray:
    """
    Render a body-33 pose as a grayscale canonical stick figure.

    Returns float32 array [1, size, size], values in [0, 1].
    Joints with confidence < conf_min are skipped.
    """
    img  = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    def _px(idx: int) -> Optional[tuple[int, int]]:
        kp = kp33[idx]
        if kp[2] < conf_min:
            return None
        return (
            max(0, min(size - 1, int(round(kp[0] * size)))),
            max(0, min(size - 1, int(round(kp[1] * size)))),
        )

    torso_thick = max(2, int(round(_TORSO_THICK * size)))
    limb_thick  = max(1, int(round(_LIMB_THICK  * size)))
    head_r      = max(2, int(round(_HEAD_RADIUS  * size)))

    # Draw bones (thicker torso bones first so limbs draw on top)
    for pass_torso in (True, False):
        for a, b in _SKELETON:
            is_torso = (min(a,b), max(a,b)) in {(min(x,y), max(x,y)) for x,y in _TORSO_BONES}
            if is_torso != pass_torso:
                continue
            pa, pb = _px(a), _px(b)
            if pa is None or pb is None:
                continue
            thick = torso_thick if is_torso else limb_thick
            _draw_capsule(draw, pa, pb, thick)

    # Head circle (centred on nose, joint 0)
    pnose = _px(0)
    if pnose is None:
        # fall back to joint 27 (neck) slightly shifted up
        pneck = _px(27)
        if pneck is not None:
            pnose = (pneck[0], max(0, pneck[1] - head_r * 2))
    if pnose is not None:
        x, y = pnose
        draw.ellipse(
            (x - head_r, y - head_r, x + head_r, y + head_r),
            fill=255,
        )

    arr = np.asarray(img, dtype=np.float32) / 255.0   # [size, size]
    return arr[np.newaxis]                              # [1, size, size]


def _draw_capsule(
    draw:  ImageDraw.ImageDraw,
    pa:    tuple[int, int],
    pb:    tuple[int, int],
    thick: int,
) -> None:
    """
    Draw a filled capsule (thick line with rounded ends) between two points.
    PIL doesn't have a native thick-line primitive, so we approximate with
    drawing a polygon for the body and circles at each end.
    """
    r = thick // 2
    ax, ay = pa
    bx, by = pb

    # Circles at endpoints
    draw.ellipse((ax - r, ay - r, ax + r, ay + r), fill=255)
    draw.ellipse((bx - r, by - r, bx + r, by + r), fill=255)

    # Rectangle body — offset perpendicular to the line direction
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ox = -dy / length * r
    oy =  dx / length * r

    poly = [
        (ax + ox, ay + oy),
        (bx + ox, by + oy),
        (bx - ox, by - oy),
        (ax - ox, ay - oy),
    ]
    draw.polygon(poly, fill=255)


# ---------------------------------------------------------------------------
# Diagnostic: measure min torso pixel width across N synthetic samples
# ---------------------------------------------------------------------------

def measure_min_torso_px(n_samples: int = 2000, size: int = 128, seed: int = 42) -> dict:
    """
    Generate n_samples synthetic skeletons, render each, measure the minimum
    non-zero column width in the torso region (rows 20–65% of image height).
    Returns {"min_px": int, "p5_px": float, "mean_px": float}.

    Callers should assert min_px >= 4; if not, increase 'size'.
    """
    from fitengine.data_gen.synthetic import SyntheticDataGenerator

    gen    = SyntheticDataGenerator(seed=seed)
    # generate_batch returns features only; we need raw keypoints.
    # Access private methods directly (same package) to get kp_front.
    bodies = gen._sample_bodies(n_samples)   # [N, 5] — h, sh, ch, wa, ne (cm)

    widths: list[int] = []
    rng = gen._rng

    for row in bodies:
        h_cm, sh_cm, ch_cm, wa_cm, _ = row
        h_over_w   = float(np.clip(rng.normal(1.78, 0.12), 1.40, 2.10))
        fill_front = float(rng.uniform(0.55, 0.88))
        hip_circ   = wa_cm + float(rng.normal(12.0, 3.0))
        hip_span   = hip_circ / np.pi
        hip_half   = hip_span / 2.0

        kp_f = gen._place_skeleton(h_cm, sh_cm, hip_half, fill_front, h_over_w, view="front")
        img  = render_body_image(kp_f, size=size)[0]   # [size, size]

        # Torso band: rows 20%–65% of image height
        r0 = int(0.20 * size)
        r1 = int(0.65 * size)
        col_max  = img[r0:r1, :].max(axis=0)
        lit_cols = int((col_max > 0.05).sum())
        widths.append(lit_cols)

    a = np.array(widths)
    return {
        "min_px":  int(a.min()),
        "p5_px":   float(np.percentile(a, 5)),
        "mean_px": float(a.mean()),
        "size":    size,
    }


if __name__ == "__main__":
    import json, sys
    n    = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    result = measure_min_torso_px(n_samples=n, size=size)
    print(json.dumps(result, indent=2))
    verdict = "OK" if result["min_px"] >= 4 else f"TOO THIN — use size >= {size * 2}"
    print(f"\nVerdict: {verdict}")
