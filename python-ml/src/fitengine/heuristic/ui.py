"""
FitEngine Pilot — Gradio 2-photo capture app.

Two-step flow:
  Step 1: Front-view capture with alignment guidance
  Step 2: Side-view capture with alignment guidance
  Result: Collar / Jacket / Trouser recommendation + confidence badge

Supports both live webcam capture (default) and photo upload mode.
Photo upload is immediately available — switch if completion < 35%.

Run:
    python -m fitengine.heuristic.ui
    # or
    python python-ml/src/fitengine/heuristic/ui.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure the src directory is on the path when running directly
_SRC = Path(__file__).resolve().parents[3]  # .../python-ml/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fitengine.alignment import AlignmentGuide
from fitengine.heuristic.estimator import HeuristicEstimator
from fitengine.heuristic.pilot_logger import PilotLogger
from fitengine.pipeline import FitEnginePipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEIGHT_MIN_CM = 155
_HEIGHT_MAX_CM = 210
_HOLD_SECONDS  = 1.0       # stable frames needed before auto-capture
_TIMEOUT_SECONDS = 10.0    # seconds before "Take photo anyway" appears
_LOG_PATH = Path("data/pilot_log.jsonl")

_DISCLAIMER = (
    "⚠️  Heuristic estimate — results may vary. "
    "Please cross-reference with the brand size guide before purchasing."
)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> "gr.Blocks":  # type: ignore
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError("gradio is required. pip install gradio>=4.0.0") from e

    pipeline = FitEnginePipeline(chart="generic")
    logger_inst = PilotLogger(_LOG_PATH)
    guide = AlignmentGuide()

    # ── Helper: process a captured/uploaded image ─────────────────────────

    def _run_pipeline(
        front_img_np: Optional[np.ndarray],
        side_img_np:  Optional[np.ndarray],
        height_cm:    float,
        session_id:   str,
    ) -> dict:
        if front_img_np is None or side_img_np is None:
            return {}

        # BGR conversion (Gradio sends RGB)
        front_bgr = cv2.cvtColor(front_img_np, cv2.COLOR_RGB2BGR)
        side_bgr  = cv2.cvtColor(side_img_np,  cv2.COLOR_RGB2BGR)

        result = pipeline.predict(
            front_bgr, side_bgr, height_cm,
            session_id=session_id,
        )
        logger_inst.log_event(session_id, "recommendation_shown")
        return result

    def _size_card_html(result: dict) -> str:
        if not result:
            return "<p style='color:#888'>No result yet.</p>"

        conf = result.get("confidence_level", "heuristic")
        conf_color = {
            "High":      "#22c55e",
            "Medium":    "#f59e0b",
            "Low":       "#ef4444",
            "heuristic": "#94a3b8",
        }.get(conf, "#94a3b8")

        return f"""
<div style="font-family:sans-serif; background:#1e293b; color:#f8fafc;
            padding:24px; border-radius:12px; max-width:360px;">
  <h2 style="margin:0 0 16px; font-size:20px;">Your Size Recommendation</h2>
  <table style="width:100%; border-collapse:collapse; font-size:16px;">
    <tr><td style="padding:8px 0; color:#94a3b8;">Collar</td>
        <td style="font-weight:bold;">{result.get('collar','?')}"</td></tr>
    <tr><td style="padding:8px 0; color:#94a3b8;">Jacket</td>
        <td style="font-weight:bold;">{result.get('jacket','?')}</td></tr>
    <tr><td style="padding:8px 0; color:#94a3b8;">Trouser Waist</td>
        <td style="font-weight:bold;">{result.get('trouser_waist','?')}"</td></tr>
  </table>
  <div style="margin-top:16px; display:flex; align-items:center; gap:8px;">
    <div style="width:10px; height:10px; border-radius:50%;
                background:{conf_color};"></div>
    <span style="color:{conf_color}; font-size:13px;">
      Fit Confidence: {conf}
    </span>
  </div>
  <p style="margin-top:16px; font-size:12px; color:#64748b;">{_DISCLAIMER}</p>
</div>
"""

    # ── Height input validation ────────────────────────────────────────────

    def _validate_height(height_str: str) -> tuple[float, str]:
        """Parse height input, return (cm_float, error_or_empty)."""
        h = height_str.strip().replace(",", ".")
        # Handle ft'in" format e.g. "5'11"
        if "'" in h:
            parts = h.replace('"', '').split("'")
            try:
                feet   = float(parts[0])
                inches = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
                cm = feet * 30.48 + inches * 2.54
            except ValueError:
                return 0.0, "Cannot parse height. Use cm (e.g. 178) or ft'in (e.g. 5'11)."
        else:
            try:
                cm = float(h)
            except ValueError:
                return 0.0, "Please enter height as a number (e.g. 178)."

        if cm < _HEIGHT_MIN_CM:
            return 0.0, f"Height seems too short ({cm:.0f} cm). Minimum: {_HEIGHT_MIN_CM} cm."
        if cm > _HEIGHT_MAX_CM:
            return 0.0, f"Height seems too tall ({cm:.0f} cm). Maximum: {_HEIGHT_MAX_CM} cm."
        return cm, ""

    # ── Alignment check on uploaded image ─────────────────────────────────

    def check_front_upload(img_np: Optional[np.ndarray]):
        if img_np is None:
            return None, "Upload a front-view photo above."
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        kp33 = pipeline._detector.detect(bgr)
        ok, msg = guide.check_front(kp33)
        annotated = AlignmentGuide.draw_overlay(bgr, msg, ok)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        status = "✅ Front photo looks good!" if ok else f"⚠️  {msg}"
        return annotated_rgb, status

    def check_side_upload(img_np: Optional[np.ndarray]):
        if img_np is None:
            return None, "Upload a side-view photo above."
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        kp33 = pipeline._detector.detect(bgr)
        ok, msg = guide.check_side(kp33)
        annotated = AlignmentGuide.draw_overlay(bgr, msg, ok)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        status = "✅ Side photo looks good!" if ok else f"⚠️  {msg}"
        return annotated_rgb, status

    # ── Main submit handler ────────────────────────────────────────────────

    def submit(
        front_img,
        side_img,
        height_input: str,
        session_id: str,
    ):
        height_cm, err = _validate_height(height_input)
        if err:
            return gr.update(), f"<p style='color:#ef4444'>{err}</p>", {}

        result = _run_pipeline(
            np.array(front_img) if front_img is not None else None,
            np.array(side_img)  if side_img  is not None else None,
            height_cm,
            session_id,
        )
        return gr.update(visible=True), _size_card_html(result), result

    def _mfb(measurement: str, accepted: bool):
        """Return a Gradio callback that logs per-measurement feedback."""
        event   = "measurement_accepted" if accepted else "measurement_rejected"
        key_map = {"collar": "collar", "jacket": "jacket", "waist": "trouser_waist"}
        icon    = "✅" if accepted else "❌"
        label   = "correct" if accepted else "wrong"

        def _fn(session_id: str, result: dict) -> str:
            value = result.get(key_map.get(measurement, measurement), "?")
            logger_inst.log_event(
                session_id, event,
                extra={"measurement": measurement, "value": str(value)},
            )
            return f"{icon} **{measurement.capitalize()}** marked as {label}. Thanks!"
        return _fn

    def new_session():
        sid = PilotLogger.new_session_id()
        logger_inst.log_event(sid, "widget_shown")
        return sid

    # ── Build Gradio UI ────────────────────────────────────────────────────

    with gr.Blocks(title="FitEngine — Find Your Size") as demo:

        session_state = gr.State(value="")

        gr.Markdown(
            "# FitEngine — Find Your Size\n"
            "Takes about **20 seconds**. Two photos — one front, one side.\n\n"
            "*Used by FitEngine for research purposes. "
            "No biometrics are stored on our servers. All processing is local.*"
        )

        # Auto-start session on load
        demo.load(new_session, outputs=session_state)
        session_state.change(
            lambda sid: logger_inst.log_event(sid, "flow_started") or sid,
            inputs=session_state, outputs=session_state,
        )

        # ── Height input ──────────────────────────────────────────────────
        with gr.Row():
            height_input = gr.Textbox(
                label="Your Height  (cm or ft'in — e.g. 178  or  5'11)",
                placeholder="Enter height before taking photos",
                scale=2,
            )
            height_hint = gr.Markdown(
                "_Reference: 5'7\" = 170 cm | 5'10\" = 178 cm | 6'0\" = 183 cm | 6'2\" = 188 cm_"
            )

        # ── Photo captures ────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "### Step 1 — Front View\n"
                    "Stand straight, arms slightly away from body, "
                    "full body visible including feet."
                )
                front_img = gr.Image(
                    label="Front Photo",
                    sources=["webcam", "upload"],
                    type="numpy",
                )
                front_check_img    = gr.Image(label="Alignment Check", visible=True)
                front_check_status = gr.Markdown("")
                front_img.change(check_front_upload,
                                 inputs=front_img,
                                 outputs=[front_check_img, front_check_status])

            with gr.Column():
                gr.Markdown(
                    "### Step 2 — Side View  (right side)\n"
                    "Turn 90° to your right. Stand straight, "
                    "full body visible including feet."
                )
                side_img = gr.Image(
                    label="Side Photo",
                    sources=["webcam", "upload"],
                    type="numpy",
                )
                side_check_img    = gr.Image(label="Alignment Check", visible=True)
                side_check_status = gr.Markdown("")
                side_img.change(check_side_upload,
                                inputs=side_img,
                                outputs=[side_check_img, side_check_status])

        # ── Get size button ───────────────────────────────────────────────
        get_size_btn = gr.Button("Get My Size →", variant="primary")

        result_state = gr.State(value={})

        result_group = gr.Column(visible=False)
        with result_group:
            result_html = gr.HTML("")
            gr.Markdown("---\n**Was each measurement correct for you?**")
            with gr.Row():
                gr.Markdown("**Collar**")
                collar_ok_btn  = gr.Button("✓ Correct", variant="secondary", scale=1)
                collar_bad_btn = gr.Button("✗ Wrong",   variant="stop",      scale=1)
            with gr.Row():
                gr.Markdown("**Jacket**")
                jacket_ok_btn  = gr.Button("✓ Correct", variant="secondary", scale=1)
                jacket_bad_btn = gr.Button("✗ Wrong",   variant="stop",      scale=1)
            with gr.Row():
                gr.Markdown("**Trouser Waist**")
                waist_ok_btn   = gr.Button("✓ Correct", variant="secondary", scale=1)
                waist_bad_btn  = gr.Button("✗ Wrong",   variant="stop",      scale=1)
            feedback_status = gr.Markdown("")

        # Step 1: instant feedback — disable button, show spinner message
        # Step 2: run inference
        # Step 3: re-enable button
        (
            get_size_btn
            .click(
                fn=lambda: (
                    gr.update(value="⏳  Analysing…", interactive=False),
                    "<p style='color:#94a3b8; font-family:sans-serif;'>"
                    "📸  Analysing your photos — takes about 5 seconds…</p>",
                ),
                inputs=None,
                outputs=[get_size_btn, result_html],
                queue=False,
            )
            .then(
                submit,
                inputs=[front_img, side_img, height_input, session_state],
                outputs=[result_group, result_html, result_state],
            )
            .then(
                fn=lambda: gr.update(value="Get My Size →", interactive=True),
                inputs=None,
                outputs=get_size_btn,
                queue=False,
            )
        )
        collar_ok_btn.click( _mfb("collar", True),  inputs=[session_state, result_state], outputs=feedback_status)
        collar_bad_btn.click(_mfb("collar", False), inputs=[session_state, result_state], outputs=feedback_status)
        jacket_ok_btn.click( _mfb("jacket", True),  inputs=[session_state, result_state], outputs=feedback_status)
        jacket_bad_btn.click(_mfb("jacket", False), inputs=[session_state, result_state], outputs=feedback_status)
        waist_ok_btn.click(  _mfb("waist",  True),  inputs=[session_state, result_state], outputs=feedback_status)
        waist_bad_btn.click( _mfb("waist",  False), inputs=[session_state, result_state], outputs=feedback_status)

        gr.Markdown(
            "---\n"
            "_FitEngine Phase 1 Pilot — heuristic estimate. "
            "Powered by RTMPose + geometry sizing._"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import gradio as gr  # noqa: F401 — needed for theme
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Default(),
    )
