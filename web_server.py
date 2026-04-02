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

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Shared state  (thread-safe via locks)
# ─────────────────────────────────────────────────────────────────

_frame_lock  = threading.Lock()
_latest_jpeg: Optional[bytes] = None      # MJPEG frame bytes

_state_lock  = threading.Lock()
_state: Dict[str, Any] = {}

_params_lock = threading.Lock()
_params: Dict[str, Any] = {
    # ── Visual overlay ───────────────────────────────────────────
    "show_skeleton":       True,
    "show_garment_box":    True,
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
}

# Param type registry so web UI shows sliders / toggles correctly
PARAM_META: Dict[str, Dict] = {
    "show_skeleton":         {"type": "bool",  "label": "Show skeleton"},
    "show_garment_box":      {"type": "bool",  "label": "Show garment box"},
    "torso_x_pad_pct":       {"type": "float", "min": 0.0, "max": 0.5,  "step": 0.01, "label": "Torso x-pad (%)"},
    "target_w_min_pct":      {"type": "float", "min": 0.2, "max": 0.9,  "step": 0.01, "label": "Garment min width (%)"},
    "target_w_scale":        {"type": "float", "min": 0.8, "max": 2.0,  "step": 0.05, "label": "Width scale"},
    "target_h_scale":        {"type": "float", "min": 0.8, "max": 2.0,  "step": 0.05, "label": "Height scale"},
    "shoulder_y_offset_px":  {"type": "int",   "min": -15, "max": 30,   "step": 1,    "label": "Shoulder Y offset (px)"},
    "mask_blend_body":       {"type": "float", "min": 0.0, "max": 1.0,  "step": 0.05, "label": "Body mask blend"},
    "mask_dilation_px":      {"type": "int",   "min": 1,   "max": 101,  "step": 2,    "label": "Mask dilation (px)"},
    "depth_skip_n":          {"type": "int",   "min": 1,   "max": 60,   "step": 1,    "label": "Depth skip N frames"},
    "meas_ttl":              {"type": "int",   "min": 1,   "max": 30,   "step": 1,    "label": "Measure TTL frames"},
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
    "result": None,
    "warnings": [],
    "updated_at": time.time(),
}


def _is_measurement_ready(state: Dict[str, Any]) -> bool:
    meas = state.get("measurements") or {}
    if not state.get("has_landmarks"):
        return False
    return all(meas.get(k) is not None for k in ("shoulder_cm", "chest_cm", "torso_cm"))


def _measurement_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    meas = state.get("measurements") or {}
    return {
        "ts": time.time(),
        "measurements": {
            "shoulder_cm": meas.get("precise_shoulder_cm") or meas.get("shoulder_cm"),
            "chest_cm": meas.get("precise_chest_cm") or meas.get("chest_cm"),
            "torso_cm": meas.get("precise_torso_cm") or meas.get("torso_cm"),
            "waist_cm": meas.get("waist_cm"),
            "size_recommendation": meas.get("size_recommendation"),
            "size_confidence": meas.get("size_confidence"),
        },
    }


def _build_fitengine_view(state: Dict[str, Any]) -> Dict[str, Any]:
    with _fitengine_lock:
        session = dict(_fitengine_session)

    ready = _is_measurement_ready(state)
    warnings = list(session.get("warnings") or [])
    if session.get("active") and not ready:
        warnings.append("Hold a stable full upper-body pose in frame.")

    return {
        "session": session,
        "readiness": {
            "measurement_ready": ready,
            "can_capture_front": bool(session.get("active") and session.get("step") == "front" and ready),
            "can_capture_side": bool(session.get("active") and session.get("step") == "side" and ready),
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
    print(f"[DEBUG] push_state called with:")
    print(f"  fps: {fps}")
    print(f"  garment: {garment}")
    print(f"  meas: {meas}")
    if meas:
        print(f"  meas keys: {list(meas.keys())}")

    state: Dict[str, Any] = {
        "fps":     round(fps, 1),
        "garment": garment or "",
        "ts":      time.time(),
    }
    if meas:
        def _s(v):
            if isinstance(v, (int, float)):  return round(float(v), 2)
            return None

        # Convert pixel measurements to approximate cm
        # At typical webcam distance (~1.5m), shoulder width of ~45cm appears as ~300px
        # So conversion: cm = pixels * 0.15
        px_to_cm = 0.15  # Approximate conversion factor

        shoulder_px = meas.get("shoulder_width", 0) or 0
        torso_px = meas.get("torso_height", 0) or 0

        # Debug: log raw values periodically
        if shoulder_px > 10:
            log.info(f"[BODY] shoulder_px={shoulder_px:.1f}, torso_px={torso_px:.1f}")

        # Calculate size recommendation using WORKING body measurements
        size_rec_data = {}
        if shoulder_px > 0 and torso_px > 0:
            print(f"[SIZE DEBUG] Calculating with WORKING measurements: shoulder_px={shoulder_px:.1f}, torso_px={torso_px:.1f}")
            try:
                from src.core.size_recommendation import get_size_recommendation, format_size_recommendation
                test_measurements = {
                    'shoulder_width': shoulder_px,
                    'torso_height': torso_px,
                    'confidence': 0.9
                }
                print(f"[SIZE DEBUG] Input: {test_measurements}")
                size_rec = get_size_recommendation(test_measurements)
                print(f"[SIZE DEBUG] SUCCESS! Size: {size_rec['recommended_size']} (confidence: {size_rec['confidence']:.2f})")

                size_rec_data = {
                    'size_recommendation': size_rec['recommended_size'],
                    'size_confidence': size_rec['confidence'],
                    'size_description': format_size_recommendation(size_rec),
                    'size_alternatives': size_rec['all_sizes'],
                    'precise_shoulder_cm': size_rec['measurements_cm']['shoulder_width'],
                    'precise_chest_cm': size_rec['measurements_cm']['chest_width'],
                    'precise_torso_cm': size_rec['measurements_cm']['torso_length']
                }
                print(f"[SIZE DEBUG] Created size data successfully!")
            except Exception as e:
                print(f"[SIZE DEBUG] Size calculation FAILED: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[SIZE DEBUG] NO measurements: shoulder_px={shoulder_px}, torso_px={torso_px}")

        state["measurements"] = {
            "shoulder_cm": _s(shoulder_px * px_to_cm) if shoulder_px > 0 else None,
            "chest_cm":    _s(shoulder_px * 0.9 * px_to_cm) if shoulder_px > 0 else None,
            "waist_cm":    _s(shoulder_px * 0.75 * px_to_cm) if shoulder_px > 0 else None,
            "torso_cm":    _s(torso_px * px_to_cm) if torso_px > 0 else None,
            "size":        meas.get("size"),
            # WORKING SIZE RECOMMENDATIONS - Added directly here!
            "size_recommendation": None,
            "size_confidence": None,
            "size_description": "",
            "size_alternatives": {},
            "precise_shoulder_cm": None,
            "precise_chest_cm": None,
            "precise_torso_cm": None,
        }

        # Calculate size recommendation using the WORKING measurements (add after measurements dict)
        if shoulder_px > 0 and torso_px > 0:
            try:
                from src.core.size_recommendation import get_size_recommendation, format_size_recommendation
                test_measurements = {
                    'shoulder_width': shoulder_px,
                    'torso_height': torso_px,
                    'confidence': 0.9
                }
                size_rec = get_size_recommendation(test_measurements)

                # Update the measurements dict with size data
                state["measurements"]["size_recommendation"] = size_rec['recommended_size']
                state["measurements"]["size_confidence"] = round(size_rec['confidence'] * 100, 1)
                state["measurements"]["size_description"] = format_size_recommendation(size_rec)
                state["measurements"]["size_alternatives"] = size_rec['all_sizes']
                state["measurements"]["precise_shoulder_cm"] = round(size_rec['measurements_cm']['shoulder_width'], 1)
                state["measurements"]["precise_chest_cm"] = round(size_rec['measurements_cm']['chest_width'], 1)
                state["measurements"]["precise_torso_cm"] = round(size_rec['measurements_cm']['torso_length'], 1)

                print(f"[SIZE SUCCESS] Recommended size: {size_rec['recommended_size']} (confidence: {size_rec['confidence']:.2f})")
            except Exception as e:
                print(f"[SIZE ERROR] Size calculation failed: {e}")
                state["measurements"]["size_recommendation"] = "M"  # Fallback
                state["measurements"]["size_confidence"] = 50
        tb = meas.get("torso_box")
        if tb:
            state["torso_box"] = list(tb)
        # Include landmarks for skeleton drawing
        if meas.get("landmarks"):
            state["has_landmarks"] = True
    with _state_lock:
        _state.clear()
        _state.update(state)


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
            "result": None,
            "warnings": [],
            "updated_at": time.time(),
        })
        return dict(_fitengine_session)


def _capture_fitengine_step(step: str, state: Dict[str, Any]) -> tuple[bool, Dict[str, Any], Optional[str]]:
    ready = _is_measurement_ready(state)
    with _fitengine_lock:
        if not _fitengine_session.get("active"):
            return False, dict(_fitengine_session), "No active FitEngine session"
        if _fitengine_session.get("step") != step:
            return False, dict(_fitengine_session), f"Current step is '{_fitengine_session.get('step')}'"
        if not ready:
            return False, dict(_fitengine_session), "Measurement signal is not stable yet"

        snap = _measurement_snapshot(state)
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


def _finalize_fitengine_session(state: Dict[str, Any]) -> tuple[bool, Dict[str, Any], Optional[str]]:
    with _fitengine_lock:
        if not _fitengine_session.get("active"):
            return False, dict(_fitengine_session), "No active FitEngine session"
        if not (_fitengine_session.get("front_captured") and _fitengine_session.get("side_captured")):
            return False, dict(_fitengine_session), "Front and side captures are required"

        meas = state.get("measurements") or {}
        chest_cm = meas.get("precise_chest_cm") or meas.get("chest_cm")
        shoulder_cm = meas.get("precise_shoulder_cm") or meas.get("shoulder_cm")
        torso_cm = meas.get("precise_torso_cm") or meas.get("torso_cm")
        waist_cm = meas.get("waist_cm")

        def _as_float(v: Any) -> Optional[float]:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _clamp(v: Optional[float], lo: float, hi: float) -> Optional[float]:
            if v is None:
                return None
            return max(lo, min(hi, v))

        chest_cm = _clamp(_as_float(chest_cm), 70.0, 145.0)
        shoulder_cm = _clamp(_as_float(shoulder_cm), 34.0, 66.0)
        torso_cm = _clamp(_as_float(torso_cm), 50.0, 90.0)
        waist_cm = _clamp(_as_float(waist_cm), 60.0, 130.0)

        # If neck is unavailable, estimate collar from chest using a conservative ratio.
        collar_cm = _clamp(_as_float(meas.get("neck_cm")) or (chest_cm * 0.39 if chest_cm else None), 30.0, 50.0)

        collar_in = round((collar_cm / 2.54) * 2) / 2 if collar_cm is not None else None
        trouser_waist_in = round((waist_cm / 2.54) / 2) * 2 if waist_cm is not None else None

        _fitengine_session["result"] = {
            "recommended_size": meas.get("size_recommendation") or "M",
            "confidence": meas.get("size_confidence") or 0,
            "chest_cm": chest_cm,
            "shoulder_cm": shoulder_cm,
            "torso_cm": torso_cm,
            "waist_cm": waist_cm,
            "collar_cm": collar_cm,
            "collar_in": collar_in,
            "trouser_waist_in": trouser_waist_in,
            "summary": meas.get("size_description") or "Recommendation generated from captured measurements.",
            "generated_at": time.time(),
        }
        _fitengine_session["step"] = "complete"
        _fitengine_session["updated_at"] = time.time()
        return True, dict(_fitengine_session), None


# ─────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────

def _make_flask_app():
    try:
        from flask import Flask, Response, jsonify, request
        from flask_cors import CORS
    except ImportError:
        log.error("Flask/flask-cors not installed — run: pip install flask flask-cors")
        return None

    app = Flask(__name__)

    # Configure CORS for both development and production
    allowed_origins = [
        "http://localhost:3000",           # React development server
        "http://localhost:3001",           # Alternative React port
        "https://*.vercel.app",           # Vercel deployments
        "https://*.netlify.app",          # Netlify deployments
        "https://ar-mirror-ui.vercel.app", # Your specific Vercel app
        "https://ar-mirror-ui.netlify.app", # Your specific Netlify app
    ]

    # Allow all origins in development, specific origins in production
    if os.environ.get('FLASK_ENV') == 'development':
        CORS(app)  # Allow all origins in development
    else:
        CORS(app, origins=allowed_origins)  # Restrict origins in production

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
        app = _make_flask_app()
        if app is None:
            return False
        self._app = app

        def _run():
            import logging as _l
            _l.getLogger("werkzeug").setLevel(_l.WARNING)
            app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

        self._thread = threading.Thread(target=_run, daemon=True, name="WebServer")
        self._thread.start()
        log.info(f"[WebServer] started → http://localhost:{self.port}  |  stream: /stream")
        return True
