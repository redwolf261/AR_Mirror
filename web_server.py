"""
web_server.py — Lightweight debug/iteration web server for AR Mirror.

Exposes:
  GET /stream          — MJPEG live video of processed frames
  GET /api/state       — JSON: current measurements, FPS, active garment
  GET /api/params      — JSON: current tunable parameters
  POST /api/params     — JSON: update one or more parameters at runtime
  GET /api/garments    — JSON: list of available garments
  POST /api/garment    — JSON: { "name": "xxx.jpg" }  — switch active garment

Usage (from tryon_selector.py):
    from web_server import WebServer
    ws = WebServer()
    ws.start()                          # starts Flask in background thread
    # inside render loop:
    ws.push_frame(frame)
    ws.push_state(fps, sel, meas)
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2
import numpy as np
import uuid
import base64
import socket

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Shared state  (thread-safe via locks)
# ─────────────────────────────────────────────────────────────────

_frame_lock  = threading.Lock()
_latest_jpeg: Optional[bytes] = None      # MJPEG frame bytes

_state_lock  = threading.Lock()
_state: Dict[str, Any] = {}

_ws_thread: Optional[threading.Thread] = None

_params_lock = threading.Lock()
_params: Dict[str, Any] = {
    # ── Visual overlay ───────────────────────────────────────────
    "show_skeleton":       True,
    "show_garment_box":    True,
    "render_tryon_overlay": True,
    # ── Garment placement ────────────────────────────────────────
    "torso_x_pad_pct":     0.25,      # extra width each side (fraction)
    "target_w_min_pct":    0.55,      # min garment width as fraction of frame
    "target_w_scale":      1.30,      # scale factor on raw torso box width
    "target_h_scale":      1.10,      # scale factor on raw torso box height
    "shoulder_y_offset_px": 8,        # pixels above shoulder lm for shirt_top
    # ── Mask blending ────────────────────────────────────────────
    "mask_blend_body":     0.70,       # body mask weight in wm blend (0=ignore mask, 1=hard clip)
    "mask_dilation_px":    31,         # body mask dilation kernel size
    # ── Pipeline timing ──────────────────────────────────────────
    "depth_skip_n":        5,          # run depth every N frames
    "meas_ttl":            5,          # re-measure every N frames
    "user_height_cm":      170.0,      # explicit scale calibration input
    "calibration_square_cm": 10.0,     # physical red-square side length
}

# Param type registry so web UI shows sliders / toggles correctly
PARAM_META: Dict[str, Dict] = {
    "show_skeleton":         {"type": "bool",  "label": "Show skeleton"},
    "show_garment_box":      {"type": "bool",  "label": "Show garment box"},
    "render_tryon_overlay":  {"type": "bool",  "label": "Render try-on overlay"},
    "torso_x_pad_pct":       {"type": "float", "min": 0.0, "max": 0.5,  "step": 0.01, "label": "Torso x-pad (%)"},
    "target_w_min_pct":      {"type": "float", "min": 0.2, "max": 0.9,  "step": 0.01, "label": "Garment min width (%)"},
    "target_w_scale":        {"type": "float", "min": 0.8, "max": 2.0,  "step": 0.05, "label": "Width scale"},
    "target_h_scale":        {"type": "float", "min": 0.8, "max": 2.0,  "step": 0.05, "label": "Height scale"},
    "shoulder_y_offset_px":  {"type": "int",   "min": -15, "max": 30,   "step": 1,    "label": "Shoulder Y offset (px)"},
    "mask_blend_body":       {"type": "float", "min": 0.0, "max": 1.0,  "step": 0.05, "label": "Body mask blend"},
    "mask_dilation_px":      {"type": "int",   "min": 1,   "max": 101,  "step": 2,    "label": "Mask dilation (px)"},
    "depth_skip_n":          {"type": "int",   "min": 1,   "max": 60,   "step": 1,    "label": "Depth skip N frames"},
    "meas_ttl":              {"type": "int",   "min": 1,   "max": 30,   "step": 1,    "label": "Measure TTL frames"},
    "user_height_cm":        {"type": "float", "min": 130, "max": 240,  "step": 1,    "label": "User height (cm)"},
    "calibration_square_cm": {"type": "float", "min": 2,   "max": 50,   "step": 0.5,  "label": "Calibration square (cm)"},
}

# callbacks registered by tryon_selector to react to param changes
_param_callbacks: list = []
# callback for garment switch
_garment_callback = None
# garment list provider
_garment_list_fn = None

_fitengine_lock = threading.Lock()
_fitengine_session: Dict[str, Any] = {
    "active": False,
    "session_id": None,
    "step": "idle",
    "height_cm": None,
    "front_captured": False,
    "side_captured": False,
    "front_snapshot": None,
    "side_snapshot": None,
    "truth_metrics": None,
    "result": None,
    "warnings": [],
    "updated_at": time.time(),
}

_signal_lock = threading.Lock()
_pose_signal_ema: Dict[str, float] = {
    "anchor_x": float("nan"),
    "anchor_y": float("nan"),
    "scale": float("nan"),
}

_metrics_lock = threading.Lock()
_perf_history: Dict[str, deque] = {
    "fps": deque(maxlen=300),
    "latency_ms": deque(maxlen=300),
    "ts_ms": deque(maxlen=300),
}

# ── Social Share Store (in-memory, max 20 entries, 15-min TTL) ────────────
_share_lock = threading.Lock()
_share_store: Dict[str, Dict[str, Any]] = {}  # share_id → payload
_SHARE_TTL_S = 900  # 15 minutes
_SHARE_MAX   = 20


def _local_ip() -> str:
    """Best-effort local LAN IP so QR codes work from phone on same WiFi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def _purge_expired_shares() -> None:
    now = time.time()
    expired = [k for k, v in _share_store.items() if now - v["created_at"] > _SHARE_TTL_S]
    for k in expired:
        _share_store.pop(k, None)


def _create_share_entry(flask_port: int) -> Dict[str, Any]:
    """Capture current frame, generate QR, build share payload."""
    with _frame_lock:
        frame_bytes = _latest_jpeg

    # Generate share ID
    share_id   = uuid.uuid4().hex[:12]
    local_ip   = _local_ip()
    share_url  = f"http://{local_ip}:{flask_port}/api/share/view/{share_id}"
    nfc_payload = share_url  # NDEF URI payload for NFC writer apps

    # Build preview b64 (latest frame or placeholder)
    preview_b64 = ""
    if frame_bytes:
        preview_b64 = base64.b64encode(frame_bytes).decode()
    else:
        # 1×1 transparent PNG placeholder
        preview_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScAAAAAElFTkSuQmCC"

    # Generate QR code pointing to share URL
    qr_b64 = ""
    try:
        import qrcode  # type: ignore
        from io import BytesIO
        qr = qrcode.QRCode(version=2, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=2)
        qr.add_data(share_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        qr_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as _qr_err:
        log.warning("[Share] qrcode not available: %s", _qr_err)

    payload: Dict[str, Any] = {
        "share_id":    share_id,
        "share_url":   share_url,
        "nfc_payload": nfc_payload,
        "qr_b64":      qr_b64,
        "preview_b64": preview_b64,
        "created_at":  time.time(),
        "expires_at":  time.time() + _SHARE_TTL_S,
    }

    with _share_lock:
        _purge_expired_shares()
        # Evict oldest if at capacity
        if len(_share_store) >= _SHARE_MAX:
            oldest_key = min(_share_store, key=lambda k: _share_store[k]["created_at"])
            _share_store.pop(oldest_key, None)
        _share_store[share_id] = payload

    return payload


def _landmark_map(pose2d: list) -> Dict[int, list]:
    points: Dict[int, list] = {}
    for pt in pose2d:
        if isinstance(pt, list) and len(pt) >= 4 and isinstance(pt[0], int):
            points[pt[0]] = pt
    return points


def _extract_mask_polygon(mask: Any, max_points: int = 48) -> Optional[list]:
    """Extract a simplified normalized contour polygon from a binary body mask."""
    if mask is None:
        return None

    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return None

    h, w = mask_np.shape[:2]
    if h <= 1 or w <= 1:
        return None

    if mask_np.dtype != np.uint8:
        mask_bin = (mask_np > 0.5).astype(np.uint8) * 255
    else:
        mask_bin = (mask_np > 0).astype(np.uint8) * 255

    if int(mask_bin.sum()) == 0:
        return None

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None

    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2)

    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int32)
        points = points[idx]

    poly = []
    for p in points:
        x = float(p[0]) / float(max(1, w - 1))
        y = float(p[1]) / float(max(1, h - 1))
        poly.append([round(x, 6), round(y, 6)])

    return poly if len(poly) >= 3 else None


def _compute_pose_signals(pose2d: list) -> Dict[str, Any]:
    """Estimate torso anchor and normalized scale; smooth via EMA for stable overlays."""
    pts = _landmark_map(pose2d)
    ls = pts.get(11)
    rs = pts.get(12)
    lh = pts.get(23)
    rh = pts.get(24)
    if not (ls and rs and lh and rh):
        return {
            "anchor": None,
            "scale": None,
            "pose_confidence": 0.0,
            "shoulder_width_norm": None,
            "torso_height_norm": None,
        }

    lsv, rsv, lhv, rhv = float(ls[3]), float(rs[3]), float(lh[3]), float(rh[3])
    confidence = max(0.0, min(1.0, (lsv + rsv + lhv + rhv) / 4.0))

    s_cx = (float(ls[1]) + float(rs[1])) * 0.5
    s_cy = (float(ls[2]) + float(rs[2])) * 0.5
    h_cx = (float(lh[1]) + float(rh[1])) * 0.5
    h_cy = (float(lh[2]) + float(rh[2])) * 0.5
    anchor_x = (s_cx + h_cx) * 0.5
    anchor_y = (s_cy + h_cy) * 0.5

    shoulder_w = float(np.hypot(float(ls[1]) - float(rs[1]), float(ls[2]) - float(rs[2])))
    torso_h = float(np.hypot(s_cx - h_cx, s_cy - h_cy))
    raw_scale = max(0.05, shoulder_w * 2.2)

    alpha = 0.22
    with _signal_lock:
        if np.isnan(_pose_signal_ema["anchor_x"]) or np.isnan(_pose_signal_ema["anchor_y"]) or np.isnan(_pose_signal_ema["scale"]):
            _pose_signal_ema["anchor_x"] = anchor_x
            _pose_signal_ema["anchor_y"] = anchor_y
            _pose_signal_ema["scale"] = raw_scale
        else:
            anchor_x_prev = _pose_signal_ema["anchor_x"]
            anchor_y_prev = _pose_signal_ema["anchor_y"]
            scale_prev = _pose_signal_ema["scale"]
            _pose_signal_ema["anchor_x"] = anchor_x_prev + (anchor_x - anchor_x_prev) * alpha
            _pose_signal_ema["anchor_y"] = anchor_y_prev + (anchor_y - anchor_y_prev) * alpha
            _pose_signal_ema["scale"] = scale_prev + (raw_scale - scale_prev) * alpha

        smoothed_anchor_x = _pose_signal_ema["anchor_x"]
        smoothed_anchor_y = _pose_signal_ema["anchor_y"]
        smoothed_scale = _pose_signal_ema["scale"]

    return {
        "anchor": {"x": round(smoothed_anchor_x, 6), "y": round(smoothed_anchor_y, 6)},
        "scale": round(smoothed_scale, 6),
        "pose_confidence": round(confidence, 6),
        "shoulder_width_norm": round(shoulder_w, 6),
        "torso_height_norm": round(torso_h, 6),
    }


def _compute_segmentation_payload(pose2d: list, meas: Optional[dict] = None) -> Dict[str, Any]:
    """Emit real segmentation contour when body mask is available, else fallback polygon."""
    start = time.perf_counter()
    pts = _landmark_map(pose2d)
    ls = pts.get(11)
    rs = pts.get(12)
    lh = pts.get(23)
    rh = pts.get(24)

    polygon = None
    if ls and rs and lh and rh:
        polygon = [
            [round(float(ls[1]), 6), round(float(ls[2]), 6)],
            [round(float(rs[1]), 6), round(float(rs[2]), 6)],
            [round(float(rh[1]), 6), round(float(rh[2]), 6)],
            [round(float(lh[1]), 6), round(float(lh[2]), 6)],
        ]

    body_mask_poly = None
    if meas:
        body_mask_poly = _extract_mask_polygon(meas.get("body_mask"))

    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "maskAvailable": body_mask_poly is not None,
        "mode": "body_mask_contour" if body_mask_poly is not None else "placeholder",
        "latencyMs": round(float(latency_ms), 3),
        "maskPolygon": body_mask_poly if body_mask_poly is not None else polygon,
    }


def _normalize_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 1.0:
        v *= 100.0
    return max(0.0, min(100.0, v))


def _percentiles(values: list[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p90": None, "p99": None, "min": None, "max": None, "avg": None}

    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "avg": round(float(np.mean(arr)), 2),
    }


def _build_accuracy_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    meas = state.get("measurements") or {}
    pose_conf = _normalize_score(state.get("pose_confidence")) or 0.0
    mconf = meas.get("measurement_confidence") or {}

    metric_scores = [
        _normalize_score(mconf.get("shoulder")),
        _normalize_score(mconf.get("chest")),
        _normalize_score(mconf.get("waist")),
        _normalize_score(mconf.get("torso")),
        _normalize_score(mconf.get("hip")),
    ]
    metric_scores = [v for v in metric_scores if v is not None]
    inference_quality = round(sum(metric_scores) / len(metric_scores), 1) if metric_scores else 0.0

    yaw_deg = _as_float(meas.get("yaw_deg"))
    yaw_penalty = min(35.0, abs(yaw_deg) * 1.2) if yaw_deg is not None else 0.0
    alignment_ok = bool(meas.get("pose_alignment_ok", True))
    alignment_factor = 100.0 - yaw_penalty
    if not alignment_ok:
        alignment_factor = min(alignment_factor, 70.0)

    pose_quality = round(
        max(0.0, min(100.0, (pose_conf * 0.55) + (inference_quality * 0.35) + (alignment_factor * 0.10))),
        1,
    )

    with _fitengine_lock:
        session = dict(_fitengine_session)

    result = session.get("result") or {}
    breakdown = result.get("confidence_breakdown") or {}
    final_conf = _normalize_score(result.get("confidence"))
    base_conf = _normalize_score(breakdown.get("base"))
    stability_score = _normalize_score(breakdown.get("stability"))
    coverage_score = _normalize_score(breakdown.get("coverage"))

    side_distinctness = None
    side_distinct_hint = ""
    if session.get("front_captured") and session.get("side_captured"):
        ok, hint, score = _side_snapshot_distinct(session)
        side_distinctness = {
            "score": None if score is None else round(float(score), 1),
            "ok": bool(ok),
            "hint": hint,
        }
    elif session.get("step") == "side":
        ok, hint = _side_pose_ready(state, session)
        side_distinct_hint = hint
        side_distinctness = {
            "score": 100.0 if ok else 0.0,
            "ok": bool(ok),
            "hint": hint,
        }

    with _metrics_lock:
        fps_hist = list(_perf_history["fps"])
        lat_hist = list(_perf_history["latency_ms"])

    return {
        "timestamp": time.time(),
        "pose_quality_score": pose_quality,
        "pose_confidence": round(pose_conf, 1),
        "inference_quality": inference_quality,
        "alignment": {
            "pose_alignment_ok": alignment_ok,
            "yaw_deg": yaw_deg,
            "alignment_factor": round(alignment_factor, 1),
        },
        "measurement": {
            "stability_score": stability_score,
            "coverage_score": coverage_score,
            "base_score": base_conf,
            "size_confidence": _normalize_score(meas.get("size_confidence")),
        },
        "capture_distinctness": side_distinctness,
        "confidence": {
            "final": final_conf,
            "level": result.get("confidence_level"),
            "summary": result.get("summary"),
            "pending_hint": side_distinct_hint,
        },
        "rolling_performance": {
            "window_frames": len(fps_hist),
            "fps": _percentiles(fps_hist),
            "latency_ms": _percentiles(lat_hist),
        },
    }


def _is_measurement_ready(state: Dict[str, Any]) -> bool:
    meas = state.get("measurements") or {}
    return all(meas.get(k) is not None for k in ("shoulder_cm", "chest_cm", "torso_cm"))


def _measurement_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    meas = state.get("measurements") or {}
    base_conf_raw = meas.get("size_confidence")
    base_conf: Optional[float]
    if isinstance(base_conf_raw, (int, float, str)):
        try:
            base_conf = float(base_conf_raw)
        except (TypeError, ValueError):
            base_conf = None
    else:
        base_conf = None

    quality_score = 40.0
    if state.get("has_landmarks"):
        quality_score += 20.0
    if base_conf is not None:
        quality_score += max(0.0, min(40.0, base_conf * 0.4))

    return {
        "ts": time.time(),
        "quality_score": round(min(100.0, quality_score), 1),
        "measurements": {
            "shoulder_cm": meas.get("precise_shoulder_cm") or meas.get("shoulder_cm"),
            "chest_cm": meas.get("precise_chest_cm") or meas.get("chest_cm"),
            "torso_cm": meas.get("precise_torso_cm") or meas.get("torso_cm"),
            "waist_cm": meas.get("waist_cm"),
            "size_recommendation": meas.get("size_recommendation"),
            "size_confidence": meas.get("size_confidence"),
        },
    }


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _side_pose_ready(state: Dict[str, Any], session: Dict[str, Any]) -> tuple[bool, str]:
    """Heuristic: side capture should show meaningfully narrower upper-body span than front capture."""
    front_snapshot = session.get("front_snapshot") or {}
    front_meas = (front_snapshot.get("measurements") or {})
    current_meas = state.get("measurements") or {}

    front_shoulder = _as_float(front_meas.get("shoulder_cm"))
    front_chest = _as_float(front_meas.get("chest_cm"))
    curr_shoulder = _as_float(current_meas.get("precise_shoulder_cm") or current_meas.get("shoulder_cm"))
    curr_chest = _as_float(current_meas.get("precise_chest_cm") or current_meas.get("chest_cm"))

    # If we cannot compare, don't hard-block side capture based on this check alone.
    if front_shoulder is None or curr_shoulder is None:
        return True, ""

    shoulder_ratio = curr_shoulder / max(front_shoulder, 1e-6)
    chest_ratio = None
    if front_chest is not None and curr_chest is not None:
        chest_ratio = curr_chest / max(front_chest, 1e-6)

    # Require a clearly narrower upper-body projection for a valid side pose.
    # If chest is present, both checks should pass; otherwise use stricter shoulder-only gate.
    shoulder_ok = shoulder_ratio <= 0.83
    if chest_ratio is not None:
        chest_ok = chest_ratio <= 0.86
        pose_ok = shoulder_ok and chest_ok
    else:
        chest_ok = None
        pose_ok = shoulder_ok

    if pose_ok:
        return True, ""
    ratio_text = f"(shoulder ratio={shoulder_ratio:.2f}"
    if chest_ok is not None and chest_ratio is not None:
        ratio_text += f", chest ratio={chest_ratio:.2f}"
    ratio_text += ")"
    return False, f"Turn further to your side (about 90 degrees) before capturing side view {ratio_text}."


def _side_snapshot_distinct(session: Dict[str, Any]) -> tuple[bool, str, Optional[float]]:
    """Validate that side snapshot is measurably different from front snapshot."""
    front_snapshot = session.get("front_snapshot") or {}
    side_snapshot = session.get("side_snapshot") or {}
    front_meas = (front_snapshot.get("measurements") or {})
    side_meas = (side_snapshot.get("measurements") or {})

    front_shoulder = _as_float(front_meas.get("shoulder_cm"))
    side_shoulder = _as_float(side_meas.get("shoulder_cm"))
    front_chest = _as_float(front_meas.get("chest_cm"))
    side_chest = _as_float(side_meas.get("chest_cm"))

    if front_shoulder is None or side_shoulder is None:
        return True, "", None

    shoulder_ratio = side_shoulder / max(front_shoulder, 1e-6)
    chest_ratio = None
    if front_chest is not None and side_chest is not None:
        chest_ratio = side_chest / max(front_chest, 1e-6)
        # Ignore clearly implausible chest-ratio outliers from single-frame jitter.
        if chest_ratio < 0.45 or chest_ratio > 1.45:
            chest_ratio = None

    # Distinct if side is narrower than front. Keep this lenient to avoid blocking finalize
    # when one of the capture frames is noisy but still usable for fusion.
    if chest_ratio is not None:
        distinct = shoulder_ratio <= 0.92 and chest_ratio <= 0.95
    else:
        distinct = shoulder_ratio <= 0.92

    if distinct:
        basis = shoulder_ratio if chest_ratio is None else (shoulder_ratio + chest_ratio) / 2.0
        score = max(0.0, min(100.0, (1.0 - basis) * 260.0 + 20.0))
        return True, "", round(score, 1)

    details = f"(shoulder ratio={shoulder_ratio:.2f}"
    if chest_ratio is not None:
        details += f", chest ratio={chest_ratio:.2f}"
    details += ")"
    return False, f"Side capture looks too similar to front view. Turn 90° and recapture side {details}.", 0.0


def _build_fitengine_view(state: Dict[str, Any]) -> Dict[str, Any]:
    with _fitengine_lock:
        session = dict(_fitengine_session)

    ready = _is_measurement_ready(state)
    side_pose_ok, side_pose_hint = _side_pose_ready(state, session)
    warnings = list(session.get("warnings") or [])
    if session.get("active") and not ready:
        warnings.append("Hold a stable full upper-body pose in frame.")
    if session.get("active") and session.get("step") == "side" and ready and not side_pose_ok:
        warnings.append(side_pose_hint)

    return {
        "session": session,
        "readiness": {
            "measurement_ready": ready,
            "can_capture_front": bool(session.get("active") and session.get("step") == "front"),
            "can_capture_side": bool(session.get("active") and session.get("step") == "side"),
            "can_finalize": bool(session.get("active") and session.get("front_captured") and session.get("side_captured")),
        },
        "warnings": warnings,
    }


def get_param(key: str) -> Any:
    with _params_lock:
        return _params.get(key)


def set_params_from_web(updates: Dict[str, Any]) -> None:
    """Called by Flask thread; notifies render thread via callbacks."""
    with _params_lock:
        for k, v in updates.items():
            if k in _params:
                # coerce type
                old = _params[k]
                try:
                    if isinstance(old, bool):
                        v = bool(v)
                    elif isinstance(old, float):
                        v = float(v)
                    elif isinstance(old, int):
                        v = int(v)
                    _params[k] = v
                except (ValueError, TypeError):
                    pass
    for cb in _param_callbacks:
        try:
            cb(updates)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────
# Push API  (called from render loop)
# ─────────────────────────────────────────────────────────────────

def push_frame(frame: np.ndarray, quality: int = 70) -> None:
    """Encode frame to JPEG and store for streaming."""
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        with _frame_lock:
            _latest_jpeg = buf.tobytes()


def push_state(fps: float, garment: str, meas: Optional[dict]) -> None:
    """Update the JSON state snapshot."""
    state: Dict[str, Any] = {
        "fps":     round(fps, 1),
        "garment": garment or "",
        "ts":      time.time(),
        "ts_ms":   int(time.time() * 1000),
    }
    if meas:
        def _s(v):
            if isinstance(v, (int, float)):  return round(float(v), 2)
            return None

        pose2d = []
        pose3d = []

        for idx, lm in enumerate(meas.get("landmarks") or []):
            try:
                x = float(getattr(lm, "x", 0.0))
                y = float(getattr(lm, "y", 0.0))
                z = float(getattr(lm, "z", 0.0))
                vis = float(getattr(lm, "visibility", 0.0))
                pose2d.append([idx, round(x, 6), round(y, 6), round(vis, 6)])
                pose3d.append([idx, round(x, 6), round(y, 6), round(z, 6), round(vis, 6)])
            except (TypeError, ValueError):
                continue

        shoulder_px = meas.get("shoulder_width", 0) or 0
        torso_px = meas.get("torso_height", 0) or 0

        state["measurements"] = {
            "shoulder_cm": _s(meas.get("shoulder_width_cm")),
            "chest_cm":    _s(meas.get("chest_circumference_cm") or meas.get("chest_cm")),
            "waist_cm":    _s(meas.get("waist_circumference_cm") or meas.get("waist_cm")),
            "torso_cm":    _s(meas.get("torso_length_cm")),
            "size":        meas.get("size"),
            "size_recommendation": meas.get("size_recommendation"),
            "size_confidence": _s((meas.get("size_confidence") or 0.0) * 100.0) if meas.get("size_confidence") is not None else None,
            "size_description": meas.get("size_description", ""),
            "size_alternatives": meas.get("size_alternatives", {}),
            "fit_classification": meas.get("fit_classification"),
            "precise_shoulder_cm": _s(meas.get("shoulder_width_cm")),
            "precise_chest_cm": _s(meas.get("chest_circumference_cm") or meas.get("chest_cm")),
            "precise_torso_cm": _s(meas.get("torso_length_cm")),
            "pose_alignment_ok": bool(meas.get("pose_alignment_ok", True)),
            "alignment_warning": meas.get("alignment_warning"),
            "cm_scale_source": meas.get("cm_scale_source"),
            "measurement_confidence": meas.get("measurement_confidence", {}),
            "looseness_ratio": _s(meas.get("looseness_ratio")),
            "yaw_deg": _s(meas.get("yaw_deg")),
            "shoulder_px": _s(shoulder_px),
            "torso_px": _s(torso_px),
        }
        tb = meas.get("torso_box")
        if tb:
            state["torso_box"] = list(tb)
        # Include landmarks for skeleton drawing
        if meas.get("landmarks"):
            state["has_landmarks"] = True
        state["pose2D"] = pose2d
        state["pose3D"] = pose3d

        signals = _compute_pose_signals(pose2d)
        state.update(signals)
        state["segmentation"] = _compute_segmentation_payload(pose2d, meas)
    else:
        state["pose2D"] = []
        state["pose3D"] = []
        state["anchor"] = None
        state["scale"] = None
        state["pose_confidence"] = 0.0
        state["shoulder_width_norm"] = None
        state["torso_height_norm"] = None
        state["segmentation"] = {
            "maskAvailable": False,
            "mode": "placeholder",
            "latencyMs": 0.0,
            "maskPolygon": None,
        }
    with _state_lock:
        _state.clear()
        _state.update(state)

    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        fps_value = 0.0

    if fps_value > 0.0:
        latency_ms = 1000.0 / fps_value
        with _metrics_lock:
            _perf_history["fps"].append(fps_value)
            _perf_history["latency_ms"].append(latency_ms)
            _perf_history["ts_ms"].append(state.get("ts_ms") or int(time.time() * 1000))


def _pose_packet() -> Dict[str, Any]:
    with _state_lock:
        snap = dict(_state)
    return {
        "pose2D": snap.get("pose2D") or [],
        "pose3D": snap.get("pose3D") or [],
        "timestamp": snap.get("ts_ms") or int(time.time() * 1000),
        "anchor": snap.get("anchor"),
        "scale": snap.get("scale"),
        "poseConfidence": snap.get("pose_confidence"),
        "shoulderWidthNorm": snap.get("shoulder_width_norm"),
        "torsoHeightNorm": snap.get("torso_height_norm"),
        "segmentation": snap.get("segmentation") or {"maskAvailable": False, "mode": "placeholder", "latencyMs": 0.0, "maskPolygon": None},
        "garment": snap.get("garment") or "",
    }


def _start_pose_ws_server(host: str = "0.0.0.0", port: int = 8765, rate_hz: int = 15) -> Optional[threading.Thread]:
    """Start async websocket server that streams the latest pose packet."""

    def _run() -> None:
        try:
            import asyncio
            import websockets
        except Exception as exc:
            log.warning(f"[PoseWS] websockets unavailable: {exc}")
            return

        async def _handler(websocket):
            interval = max(0.001, 1.0 / float(max(1, rate_hz)))
            while True:
                await websocket.send(json.dumps(_pose_packet()))
                await asyncio.sleep(interval)

        async def _main() -> None:
            async with websockets.serve(_handler, host, port, max_queue=2, ping_interval=20, ping_timeout=20):
                log.info(f"[PoseWS] started -> ws://localhost:{port}")
                await asyncio.Future()

        try:
            asyncio.run(_main())
        except Exception as exc:
            log.warning(f"[PoseWS] stopped: {exc}")

    thread = threading.Thread(target=_run, daemon=True, name="PoseWebSocketServer")
    thread.start()
    return thread


def patch_state(updates: Dict[str, Any]) -> None:
    """Merge partial updates into the state snapshot (thread-safe)."""
    with _state_lock:
        _state.update(updates)


def _start_fitengine_session(height_cm: float) -> Dict[str, Any]:
    session_id = f"fit-{int(time.time() * 1000)}"
    with _fitengine_lock:
        _fitengine_session.update({
            "active": True,
            "session_id": session_id,
            "step": "front",
            "height_cm": float(height_cm),
            "front_captured": False,
            "side_captured": False,
            "front_snapshot": None,
            "side_snapshot": None,
            "truth_metrics": None,
            "result": None,
            "warnings": [],
            "updated_at": time.time(),
        })
        return dict(_fitengine_session)


def _reset_fitengine_session() -> Dict[str, Any]:
    with _fitengine_lock:
        _fitengine_session.update({
            "active": False,
            "session_id": None,
            "step": "idle",
            "height_cm": None,
            "front_captured": False,
            "side_captured": False,
            "front_snapshot": None,
            "side_snapshot": None,
            "truth_metrics": None,
            "result": None,
            "warnings": [],
            "updated_at": time.time(),
        })
        return dict(_fitengine_session)


def _capture_fitengine_step(step: str, state: Dict[str, Any]) -> tuple[bool, Dict[str, Any], Optional[str]]:
    with _fitengine_lock:
        if not _fitengine_session.get("active"):
            return False, dict(_fitengine_session), "No active FitEngine session"
        if _fitengine_session.get("step") != step:
            return False, dict(_fitengine_session), f"Current step is '{_fitengine_session.get('step')}'"

        if step == "side":
            side_ok, side_hint = _side_pose_ready(state, _fitengine_session)
            if not side_ok:
                _fitengine_session.setdefault("warnings", []).append(side_hint)

        snap = _measurement_snapshot(state)
        if not _is_measurement_ready(state):
            _fitengine_session.setdefault("warnings", []).append("Capture saved before measurements were fully ready; hold still and recapture if needed.")
        if step == "front":
            _fitengine_session["front_captured"] = True
            _fitengine_session["front_snapshot"] = snap
            _fitengine_session["step"] = "side"
        else:
            _fitengine_session["side_captured"] = True
            _fitengine_session["side_snapshot"] = snap
            _fitengine_session["step"] = "review"

        _fitengine_session["updated_at"] = time.time()
        return True, dict(_fitengine_session), None


def _update_fitengine_truth(metrics: Dict[str, Any]) -> Dict[str, Any]:
    with _fitengine_lock:
        if _fitengine_session.get("active"):
            _fitengine_session["truth_metrics"] = dict(metrics)
            _fitengine_session["updated_at"] = time.time()
        return dict(_fitengine_session)


def _finalize_fitengine_session(state: Dict[str, Any]) -> tuple[bool, Dict[str, Any], Optional[str]]:
    with _fitengine_lock:
        if not _fitengine_session.get("active"):
            return False, dict(_fitengine_session), "No active FitEngine session"
        if not (_fitengine_session.get("front_captured") and _fitengine_session.get("side_captured")):
            return False, dict(_fitengine_session), "Front and side captures are required"

        side_distinct_ok, side_distinct_msg, side_distinct_score = _side_snapshot_distinct(_fitengine_session)
        if not side_distinct_ok:
            # Do not hard-fail finalize on this heuristic; keep it as a warning and
            # confidence penalty so users still get a size recommendation.
            _fitengine_session.setdefault("warnings", []).append(side_distinct_msg)
            side_distinct_score = 0.0

        meas = state.get("measurements") or {}
        truth = _fitengine_session.get("truth_metrics") or {}
        front_snapshot = _fitengine_session.get("front_snapshot") or {}
        side_snapshot = _fitengine_session.get("side_snapshot") or {}
        front_meas = (front_snapshot.get("measurements") or {})
        side_meas = (side_snapshot.get("measurements") or {})

        chest_cm = truth.get("chest_circumference_cm") or meas.get("precise_chest_cm") or meas.get("chest_cm")
        shoulder_cm = truth.get("shoulder_cm") or meas.get("precise_shoulder_cm") or meas.get("shoulder_cm")
        torso_cm = truth.get("torso_cm") or meas.get("precise_torso_cm") or meas.get("torso_cm")
        waist_cm = truth.get("waist_circumference_cm") or meas.get("waist_cm")

        def _clamp(v: Optional[float], lo: float, hi: float) -> Optional[float]:
            if v is None:
                return None
            return max(lo, min(hi, v))

        def _fused_metric(metric: str, live_value: Any, lo: float, hi: float) -> tuple[Optional[float], bool]:
            candidates: list[float] = []
            weights: list[float] = []

            fv = _as_float(front_meas.get(metric))
            if fv is not None:
                candidates.append(fv)
                weights.append(max(0.4, _as_float(front_snapshot.get("quality_score")) or 60.0))

            sv = _as_float(side_meas.get(metric))
            if sv is not None:
                candidates.append(sv)
                weights.append(max(0.4, _as_float(side_snapshot.get("quality_score")) or 60.0))

            lv = _as_float(live_value)
            if lv is not None:
                candidates.append(lv)
                weights.append(75.0)

            if not candidates:
                return None, False

            fused = float(np.average(np.array(candidates), weights=np.array(weights)))
            clamped = fused < lo or fused > hi
            return _clamp(fused, lo, hi), clamped

        def _metric_stability(front_value: Any, side_value: Any, live_value: Any, final_value: Optional[float]) -> Optional[float]:
            points = []
            for source in (front_value, side_value, live_value):
                v = _as_float(source)
                if v is not None:
                    points.append(v)
            if final_value is not None:
                points.append(final_value)

            if len(points) < 2:
                return None

            pmax, pmin = max(points), min(points)
            base = max(1.0, abs(final_value) if final_value is not None else abs(sum(points) / len(points)))
            rel_spread = (pmax - pmin) / base
            # Softer penalty than before; avoid collapsing to zero for moderate pose variance.
            return max(10.0, min(100.0, 100.0 - rel_spread * 120.0))

        chest_cm, chest_clamped = _fused_metric("chest_cm", chest_cm, 70.0, 145.0)
        shoulder_cm, shoulder_clamped = _fused_metric("shoulder_cm", shoulder_cm, 34.0, 66.0)
        torso_cm, torso_clamped = _fused_metric("torso_cm", torso_cm, 50.0, 90.0)
        waist_cm, waist_clamped = _fused_metric("waist_cm", waist_cm, 60.0, 130.0)

        # If neck is unavailable, estimate collar from chest using a conservative ratio.
        collar_cm = _clamp(_as_float(meas.get("neck_cm")) or (chest_cm * 0.39 if chest_cm else None), 30.0, 50.0)

        collar_in = round((collar_cm / 2.54) * 2) / 2 if collar_cm is not None else None
        trouser_waist_in = round((waist_cm / 2.54) / 2) * 2 if waist_cm is not None else None

        base_conf = _as_float(truth.get("truth_confidence")) or _as_float(meas.get("size_confidence"))
        if base_conf is None or base_conf <= 0:
            front_q = _as_float(front_snapshot.get("quality_score")) or 60.0
            side_q = _as_float(side_snapshot.get("quality_score")) or 60.0
            # Conservative fallback when model confidence is unavailable.
            base_conf = ((front_q + side_q) / 2.0) * 0.75
        # Support both 0..1 and 0..100 representations.
        if base_conf <= 1.0:
            base_conf *= 100.0

        stability_components = [
            _metric_stability(front_meas.get("shoulder_cm"), side_meas.get("shoulder_cm"), meas.get("precise_shoulder_cm") or meas.get("shoulder_cm"), shoulder_cm),
            _metric_stability(front_meas.get("chest_cm"), side_meas.get("chest_cm"), meas.get("precise_chest_cm") or meas.get("chest_cm"), chest_cm),
            _metric_stability(front_meas.get("waist_cm"), side_meas.get("waist_cm"), meas.get("waist_cm"), waist_cm),
            _metric_stability(front_meas.get("torso_cm"), side_meas.get("torso_cm"), meas.get("precise_torso_cm") or meas.get("torso_cm"), torso_cm),
        ]
        stability_values = [v for v in stability_components if v is not None]
        stability_score = round(sum(stability_values) / len(stability_values), 1) if stability_values else 65.0
        stability_score = max(20.0, stability_score)

        present_count = sum(v is not None for v in (shoulder_cm, chest_cm, waist_cm, torso_cm))
        clamped_count = sum(bool(v) for v in (shoulder_clamped, chest_clamped, waist_clamped, torso_clamped))
        coverage_score = round((present_count / 4.0) * 100.0, 1)
        coverage_score = max(50.0, coverage_score - (clamped_count * 15.0)) if present_count > 0 else 0.0

        confidence = round(
            max(0.0, min(100.0, (base_conf * 0.55) + (stability_score * 0.30) + (coverage_score * 0.15))),
            1,
        )

        if confidence >= 85:
            confidence_level = "High"
        elif confidence >= 65:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        reasons = [
            f"Stability across captures: {stability_score:.0f}/100",
            f"Measurement coverage: {coverage_score:.0f}/100 ({present_count}/4 key metrics)",
        ]
        if side_distinct_score is not None:
            reasons.append(f"Side-view distinctness: {side_distinct_score:.0f}/100")
        if clamped_count > 0:
            reasons.append("Some measurements hit safety bounds; recapturing in better framing can improve precision.")
        if confidence_level == "High":
            reasons.append("Front and side captures are consistent, so size confidence is strong.")
        elif confidence_level == "Medium":
            reasons.append("Most metrics are reliable; one more recapture can further improve precision.")
        else:
            reasons.append("Try recapturing with steadier posture and better framing to improve reliability.")

        summary = f"Size guidance is based on fused front/side measurements with {confidence_level.lower()} confidence."

        _fitengine_session["result"] = {
            "recommended_size": meas.get("size_recommendation") or "M",
            "confidence": confidence,
            "confidence_level": confidence_level,
            "chest_cm": chest_cm,
            "shoulder_cm": shoulder_cm,
            "torso_cm": torso_cm,
            "waist_cm": waist_cm,
            "collar_cm": collar_cm,
            "collar_in": collar_in,
            "trouser_waist_in": trouser_waist_in,
            "confidence_breakdown": {
                "base": round(base_conf, 1),
                "stability": stability_score,
                "coverage": coverage_score,
            },
            "reasons": reasons,
            "summary": summary,
            "generated_at": time.time(),
            "truth_source": bool(truth),
            "truth_diagnostics": truth.get("diagnostics") if isinstance(truth.get("diagnostics"), dict) else None,
        }
        _fitengine_session["step"] = "complete"
        _fitengine_session["updated_at"] = time.time()
        return True, dict(_fitengine_session), None


# ─────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────

def _make_flask_app(port: int = 5051):
    app_port = port
    try:
        from flask import Flask, Response, jsonify, request
        from flask_cors import CORS
    except ImportError:
        log.error("Flask/flask-cors not installed — run: pip install flask flask-cors")
        return None

    app = Flask(__name__)

    # Configure CORS for both development and production
    allowed_origins = [
        "http://localhost:3000",            # React dev (CRA default)
        "http://localhost:3001",            # ar-mirror-ui Vite dev server
        "http://localhost:3005",            # web-ui Vite dev server (alt)
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3005",
        "https://*.vercel.app",             # Vercel deployments
        "https://*.netlify.app",            # Netlify deployments
        "https://ar-mirror-ui.vercel.app",
        "https://ar-mirror-ui.netlify.app",
    ]

    # Allow all origins in development, specific origins in production
    if os.environ.get('FLASK_ENV') == 'development' or os.environ.get('AR_DEV', '1') == '1':
        CORS(app)  # Allow all origins in development
    else:
        CORS(app, origins=allowed_origins)  # Restrict origins in production

    @app.route("/")
    def index():
        return jsonify({
            "ok": True,
            "service": "AR Mirror Web Server",
            "message": "Use /stream for MJPEG and /api/* for JSON endpoints.",
            "endpoints": {
                "stream": "/stream",
                "state": "/api/state",
                "params_get": "/api/params",
                "garments": "/api/garments",
                "snapshot": "/api/snapshot",
                "health": "/healthz",
            },
        })

    @app.route("/healthz")
    def healthz():
        return jsonify({"ok": True})

    # ── MJPEG stream ──────────────────────────────────────────────
    def _gen_frames():
        while True:
            with _frame_lock:
                frame = _latest_jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
            time.sleep(0.033)   # cap stream at ~30 fps to browser

    @app.route("/stream")
    def stream():
        return Response(
            _gen_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # ── State snapshot ────────────────────────────────────────────
    @app.route("/api/state")
    def api_state():
        with _state_lock:
            current_state = dict(_state)
        current_state["fit_engine"] = _build_fitengine_view(current_state)
        return jsonify(current_state)

    @app.route("/api/metrics/accuracy")
    def api_metrics_accuracy():
        with _state_lock:
            current_state = dict(_state)
        metrics = _build_accuracy_metrics(current_state)
        return jsonify(metrics)

    # ── Params ────────────────────────────────────────────────────
    @app.route("/api/params", methods=["GET"])
    def api_params_get():
        with _params_lock:
            vals = dict(_params)
        return jsonify({"values": vals, "meta": PARAM_META})

    @app.route("/api/params", methods=["POST"])
    def api_params_post():
        data = request.get_json(force=True, silent=True) or {}
        set_params_from_web(data)
        with _params_lock:
            return jsonify({"ok": True, "values": dict(_params)})

    # ── Garment list ──────────────────────────────────────────────
    @app.route("/api/garments")
    def api_garments():
        garments = _garment_list_fn() if _garment_list_fn else []
        return jsonify({"garments": garments})

    # ── Garment thumbnail image ───────────────────────────────────
    @app.route("/api/garment_image/<path:name>")
    def api_garment_image(name: str):
        import os, pathlib
        # Search in common cloth directories
        search_dirs = [
            pathlib.Path("dataset/train/cloth"),
            pathlib.Path("data/garments"),
            pathlib.Path("assets/garments"),
            pathlib.Path("garment_assets"),
            pathlib.Path("garment_samples"),
        ]
        for d in search_dirs:
            candidate = d / name
            if candidate.exists():
                from flask import send_file
                # Serve as thumbnail (resize on the fly if large)
                img = cv2.imread(str(candidate))
                if img is not None:
                    h, w = img.shape[:2]
                    if w > 300:
                        scale = 300 / w
                        img = cv2.resize(img, (300, int(h * scale)), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        from flask import Response as _Resp
                        return _Resp(buf.tobytes(), mimetype="image/jpeg",
                                     headers={"Cache-Control": "public, max-age=3600"})
        from flask import abort
        abort(404)

    @app.route("/api/garment", methods=["POST"])
    def api_garment_post():
        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name", "")
        if _garment_callback and name:
            _garment_callback(name)
            return jsonify({"ok": True, "garment": name})
        return jsonify({"ok": False, "error": "no callback or empty name"}), 400

    # ── Snapshot: frame + state + params in one call ──────────────
    @app.route("/api/snapshot")
    def api_snapshot():
        import base64
        with _frame_lock:
            frame = _latest_jpeg
        with _state_lock:
            state = dict(_state)
        with _params_lock:
            params = dict(_params)
        b64 = base64.b64encode(frame).decode() if frame else ""
        state["fit_engine"] = _build_fitengine_view(state)
        return jsonify({"frame_jpeg_b64": b64, "state": state, "params": params})

    @app.route("/api/fitengine/session/status", methods=["GET"])
    def fitengine_status():
        with _state_lock:
            current_state = dict(_state)
        return jsonify(_build_fitengine_view(current_state))

    @app.route("/api/fitengine/session/start", methods=["POST"])
    def fitengine_start():
        data = request.get_json(force=True, silent=True) or {}
        height_cm = data.get("height_cm", 170)
        try:
            height_cm = float(height_cm)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "height_cm must be numeric"}), 400
        if height_cm < 130 or height_cm > 240:
            return jsonify({"ok": False, "error": "height_cm out of range"}), 400
        session = _start_fitengine_session(height_cm)
        return jsonify({"ok": True, "session": session})

    @app.route("/api/fitengine/session/capture-front", methods=["POST"])
    def fitengine_capture_front():
        with _state_lock:
            current_state = dict(_state)
        ok, session, err = _capture_fitengine_step("front", current_state)
        code = 200 if ok else 400
        return jsonify({"ok": ok, "session": session, "error": err}), code

    @app.route("/api/fitengine/session/capture-side", methods=["POST"])
    def fitengine_capture_side():
        with _state_lock:
            current_state = dict(_state)
        ok, session, err = _capture_fitengine_step("side", current_state)
        code = 200 if ok else 400
        return jsonify({"ok": ok, "session": session, "error": err}), code

    @app.route("/api/fitengine/session/truth", methods=["POST"])
    def fitengine_truth():
        data = request.get_json(force=True, silent=True) or {}
        metrics = data.get("metrics") if isinstance(data, dict) else None
        if not isinstance(metrics, dict):
            return jsonify({"ok": False, "error": "metrics payload must be an object"}), 400
        session = _update_fitengine_truth(metrics)
        return jsonify({"ok": True, "session": session})

    @app.route("/api/fitengine/session/finalize", methods=["POST"])
    def fitengine_finalize():
        with _state_lock:
            current_state = dict(_state)
        ok, session, err = _finalize_fitengine_session(current_state)
        code = 200 if ok else 400
        return jsonify({"ok": ok, "session": session, "error": err}), code

    @app.route("/api/fitengine/session/reset", methods=["POST"])
    def fitengine_reset():
        session = _reset_fitengine_session()
        return jsonify({"ok": True, "session": session})

    # ── Social Share: QR + NFC ────────────────────────────────────────────
    @app.route("/api/share/generate", methods=["POST"])
    def api_share_generate():
        """Capture current AR frame, generate QR code and NFC payload."""
        try:
            payload = _create_share_entry(flask_port=app_port)
            # Don't return the full preview in the JSON (it's large); client
            # already has it via the MJPEG stream. We return qr_b64 and metadata.
            return jsonify({
                "ok":          True,
                "share_id":    payload["share_id"],
                "share_url":   payload["share_url"],
                "nfc_payload": payload["nfc_payload"],
                "qr_b64":      payload["qr_b64"],
                "preview_b64": payload["preview_b64"],
                "expires_at":  payload["expires_at"],
            })
        except Exception as exc:
            log.exception("[Share] generate failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/share/view/<share_id>")
    def api_share_view(share_id: str):
        """Return a mobile-friendly HTML page showing the captured try-on image."""
        with _share_lock:
            entry = _share_store.get(share_id)

        if not entry or time.time() - entry["created_at"] > _SHARE_TTL_S:
            return "<h1 style='font-family:sans-serif;text-align:center;margin-top:20vh'>Link expired or not found.</h1>", 404

        preview_b64 = entry.get("preview_b64", "")
        share_url   = entry.get("share_url", "")
        img_src     = f"data:image/jpeg;base64,{preview_b64}" if preview_b64 else ""
        created     = time.strftime("%H:%M", time.localtime(entry["created_at"]))

        html = f"""\
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta property="og:title" content="My Virtual Try-On — AR Mirror" />
  <meta property="og:description" content="Check out my virtual try-on from AR Mirror!" />
  <meta property="og:image" content="{img_src}" />
  <title>My Try-On · AR Mirror</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #05050A; color: #F8F8FF;
      font-family: system-ui, -apple-system, sans-serif;
      display: flex; flex-direction: column; align-items: center;
      min-height: 100vh; padding: 24px 16px;
    }}
    .pill {{
      background: rgba(124,58,237,0.2); border: 1px solid rgba(124,58,237,0.5);
      color: #A78BFA; border-radius: 999px; padding: 4px 12px;
      font-size: 12px; letter-spacing: 0.08em; margin-bottom: 18px;
    }}
    h1 {{ font-size: clamp(18px, 5vw, 26px); font-weight: 700; margin-bottom: 6px;
          background: linear-gradient(90deg, #7C3AED, #06B6D4); -webkit-background-clip: text;
          -webkit-text-fill-color: transparent; }}
    p.sub {{ font-size: 13px; color: #8B8BA7; margin-bottom: 22px; }}
    .frame {{
      width: min(420px, 95vw);
      border-radius: 16px; overflow: hidden;
      border: 1px solid rgba(124,58,237,0.35);
      box-shadow: 0 0 40px rgba(124,58,237,0.2);
      margin-bottom: 22px;
    }}
    .frame img {{ width: 100%; display: block; }}
    .share-row {{
      display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;
      margin-bottom: 20px;
    }}
    .btn {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 10px 20px; border-radius: 999px; border: none; cursor: pointer;
      font-size: 14px; font-weight: 600; text-decoration: none;
    }}
    .btn-wa  {{ background: #25D366; color: #fff; }}
    .btn-x   {{ background: #000; color: #fff; border: 1px solid #333; }}
    .btn-cp  {{ background: rgba(255,255,255,0.08); color: #F8F8FF; border: 1px solid rgba(255,255,255,0.15); }}
    .meta {{ font-size: 11px; color: #4A4A65; }}
  </style>
</head>
<body>
  <span class="pill">AR MIRROR · VIRTUAL TRY-ON</span>
  <h1>My Try-On Look</h1>
  <p class="sub">Captured at {created}</p>
  {f'<div class="frame"><img src="{img_src}" alt="Try-on snapshot" /></div>' if img_src else ''}
  <div class="share-row">
    <a class="btn btn-wa" href="whatsapp://send?text=Check out my try-on! {share_url}" target="_blank">
      💬 WhatsApp
    </a>
    <a class="btn btn-x" href="https://twitter.com/intent/tweet?text=My+virtual+try-on+result!&url={share_url}" target="_blank">
      𝕏 Share
    </a>
    <button class="btn btn-cp" onclick="navigator.clipboard.writeText('{share_url}').then(()=>this.textContent='✓ Copied!')">
      🔗 Copy Link
    </button>
  </div>
  <p class="meta">Link expires 15 minutes after capture.</p>
</body>
</html>"""
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    return app


# ─────────────────────────────────────────────────────────────────
# WebServer  (public class used by tryon_selector)
# ─────────────────────────────────────────────────────────────────

class WebServer:
    """
    Thin wrapper that starts Flask in a daemon thread.

    Usage::

        ws = WebServer(port=5050)
        ws.register_param_callback(lambda updates: ...)
        ws.register_garment_callback(lambda name: ...)
        ws.register_garment_list(lambda: ["00000_00.jpg", ...])
        ws.start()
        # per frame:
        ws.push_frame(frame)
        ws.push_state(fps, sel, meas)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5050):
        self.host = host
        self.port = port
        self._app = None
        self._thread: Optional[threading.Thread] = None

    # ── Registration helpers ──────────────────────────────────────
    def register_param_callback(self, cb):
        _param_callbacks.append(cb)

    def register_garment_callback(self, cb):
        global _garment_callback
        _garment_callback = cb

    def register_garment_list(self, fn):
        global _garment_list_fn
        _garment_list_fn = fn

    # ── Push helpers (delegates to module-level fns) ──────────────
    @staticmethod
    def push_frame(frame: np.ndarray, quality: int = 70):
        push_frame(frame, quality)

    @staticmethod
    def push_state(fps: float, garment: str, meas: Optional[dict]):
        push_state(fps, garment, meas)

    @staticmethod
    def get_param(key: str) -> Any:
        return get_param(key)

    @staticmethod
    def patch_state(updates: Dict[str, Any]):
        patch_state(updates)

    # ── Lifecycle ─────────────────────────────────────────────────
    def start(self):
        global _ws_thread
        app = _make_flask_app(port=self.port)
        if app is None:
            return False
        self._app = app

        def _run():
            import logging as _l
            _l.getLogger("werkzeug").setLevel(_l.WARNING)
            app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

        self._thread = threading.Thread(target=_run, daemon=True, name="WebServer")
        self._thread.start()

        if _ws_thread is None or not _ws_thread.is_alive():
            ws_port = int(os.environ.get("AR_MIRROR_POSE_WS_PORT", "8765"))
            _ws_thread = _start_pose_ws_server(host=self.host, port=ws_port, rate_hz=15)

        log.info(f"[WebServer] started → http://localhost:{self.port}  |  stream: /stream")
        return True
