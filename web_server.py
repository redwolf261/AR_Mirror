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
    }
    if meas:
        def _s(v):
            if isinstance(v, (int, float)):  return round(float(v), 2)
            return None
        state["measurements"] = {
            "shoulder_cm": _s(meas.get("shoulder_cm")),
            "chest_cm":    _s(meas.get("chest_cm")),
            "waist_cm":    _s(meas.get("waist_cm")),
            "torso_cm":    _s(meas.get("torso_cm")),
            "size":        meas.get("size"),
        }
        tb = meas.get("torso_box")
        if tb:
            state["torso_box"] = list(tb)
    with _state_lock:
        _state.clear()
        _state.update(state)


def patch_state(updates: Dict[str, Any]) -> None:
    """Merge partial updates into the state snapshot (thread-safe)."""
    with _state_lock:
        _state.update(updates)


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
    CORS(app)

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
            return jsonify(dict(_state))

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
        return jsonify({"frame_jpeg_b64": b64, "state": state, "params": params})

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
