"""
AR Mirror — Overlay / HUD Drawing

Extracted from app.py to reduce monolith size.
Contains all performance and info overlay rendering.
"""

import os
import cv2
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class OverlayRenderer:
    """
    Mixin providing overlay/HUD methods for ARMirrorApp.
    
    Expects the host class to provide:
      - self.frame_times, self.frame_count, self.start_time
      - self.dataset_pairs, self.current_garment_idx, self.garments
      - self.phase, self.body_fitter
    """

    def _draw_overlay(self, frame, telemetry):
        """Draw performance and info overlay."""
        h, w = frame.shape[:2]
        
        # Calculate FPS
        avg_latency = 0
        if self.frame_times:
            avg_latency = np.mean(list(self.frame_times))
            current_fps = 1.0 / avg_latency if avg_latency > 0 else 0
        else:
            current_fps = 0
        
        # Background panel (top-left)
        cv2.rectangle(frame, (10, 10), (420, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (420, 180), (0, 255, 0), 2)
        
        cv2.putText(frame, "AR MIRROR - LIVE DEMO", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames: {self.frame_count}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        garment_display_name = ""
        garment_color = (0, 255, 0)
        if self.dataset_pairs and self.current_garment_idx < len(self.dataset_pairs):
            pair = self.dataset_pairs[self.current_garment_idx]
            garment_display_name = os.path.basename(pair['garment'])
        else:
            garment = self.garments[self.current_garment_idx]
            garment_display_name = garment['name']
            garment_color = garment['color']

        cv2.putText(frame, f"Garment: {garment_display_name}", (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, garment_color, 2)
        
        # Display current phase
        phase_names = {
            2: "Phase 2: Neural",
            1: "Phase 1: GMM",
            0: "Phase 0: Blend"
        }
        mode_status = phase_names.get(self.phase, "Unknown")
        mode_color = {2: (0, 255, 255), 1: (255, 255, 0), 0: (0, 255, 0)}.get(self.phase, (128, 128, 128))
        cv2.putText(frame, f"Mode: {mode_status}", (20, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Right panel - performance details
        cv2.rectangle(frame, (w-300, 10), (w-10, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (w-300, 10), (w-10, 200), (255, 0, 0), 2)
        
        cv2.putText(frame, "PERFORMANCE", (w-280, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Latency: {avg_latency*1000:.1f}ms", (w-280, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        baseline_fps = 14.0
        if current_fps >= baseline_fps:
            status = "EXCELLENT"
            status_color = (0, 255, 0)
        elif current_fps >= 10:
            status = "GOOD"
            status_color = (0, 165, 255)
        else:
            status = "LOW"
            status_color = (0, 0, 255)
            
        cv2.putText(frame, f"Status: {status}", (w-280, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        elapsed = time.time() - self.start_time
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (w-280, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        cv2.putText(frame, "Target: 10+ FPS", (w-280, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Detection status panel (bottom-right) — always visible
        if self.body_fitter:
            diag = self.body_fitter.get_diagnostics()
            status = diag['status']
            conf = diag['confidence']
            rate = diag['success_rate']
            fails = diag['consecutive_failures']
            
            status_colors = {
                'detected': (0, 255, 0),
                'low_confidence': (0, 200, 255),
                'no_person': (0, 0, 255),
                'not_started': (128, 128, 128),
            }
            sc = status_colors.get(status, (128, 128, 128))
            
            panel_x = w - 300
            panel_y = h - 110
            cv2.rectangle(frame, (panel_x, panel_y), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (panel_x, panel_y), (w - 10, h - 10), sc, 2)
            
            cv2.circle(frame, (panel_x + 15, panel_y + 20), 6, sc, -1)
            cv2.putText(frame, f"POSE: {status.replace('_', ' ').upper()}", 
                       (panel_x + 30, panel_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)
            cv2.putText(frame, f"Confidence: {conf:.0%} | Rate: {rate:.0f}%",
                       (panel_x + 15, panel_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            if fails > 3:
                cv2.putText(frame, f"No body for {fails} frames - step back/improve lighting",
                           (panel_x + 15, panel_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            elif status == 'low_confidence':
                cv2.putText(frame, "Partial detection - face the camera",
                           (panel_x + 15, panel_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
            elif status == 'detected':
                cv2.putText(frame, "Body tracked [d=debug]",
                           (panel_x + 15, panel_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        # ── Stability metrics panel (bottom-left, debug mode) ──
        if getattr(self, 'show_debug', False):
            self._draw_stability_panel(frame)
        
        return frame

    def _draw_stability_panel(self, frame):
        """Draw live stability instrumentation panel (bottom-left).
        
        Shows GPD, landmark jitter, θ drift, and STABLE/UNSTABLE verdict.
        Requires self.body_fitter (for landmark stats) and self.phase2_pipeline
        (for transform/GPD stats).
        """
        h, w = frame.shape[:2]
        # Panel dimensions
        px, py = 10, h - 200
        pw, ph = 340, 190
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 0), 2)
        
        y = py + 22
        cv2.putText(frame, "STABILITY METRICS", (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        y += 28
        
        # Landmark jitter
        lm_jitter = -1.0
        if getattr(self, 'body_fitter', None) and hasattr(self.body_fitter, 'landmark_logger'):
            lm_stats = self.body_fitter.landmark_logger.get_stats()
            lm_jitter = lm_stats.get('mean_displacement_static_only', -1.0)
            lm_txt = f"{lm_jitter:.2f} px" if lm_jitter >= 0 else "N/A"
            cv2.putText(frame, f"  Landmark jitter: {lm_txt}", (px + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            y += 22
        
        # GMM θ drift & GPD — from Phase 2 pipeline
        theta_drift = -1.0
        gpd_static = -1.0
        if getattr(self, 'phase2_pipeline', None) and hasattr(self.phase2_pipeline, 'transform_logger'):
            t_stats = self.phase2_pipeline.transform_logger.get_stats()
            theta_drift = t_stats.get('theta_frame_drift', -1.0)
            det_var = t_stats.get('determinant_variance', -1.0)
            t_txt = f"{theta_drift:.4f}" if theta_drift >= 0 else "N/A"
            cv2.putText(frame, f"  Theta drift: {t_txt}", (px + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            y += 22
            
            d_txt = f"{det_var:.6f}" if det_var >= 0 else "N/A"
            cv2.putText(frame, f"  Det variance: {d_txt}", (px + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            y += 22
            
            g_stats = self.phase2_pipeline.gpd_metric.get_stats()
            gpd_static = g_stats.get('gpd_rgb_static', -1.0)
            gpd_lum = g_stats.get('gpd_luminance_static', -1.0)
            g_txt = f"{gpd_static:.2f}" if gpd_static >= 0 else "N/A"
            l_txt = f"{gpd_lum:.2f}" if gpd_lum >= 0 else "N/A"
            cv2.putText(frame, f"  GPD (RGB): {g_txt}  (Lum): {l_txt}", (px + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            y += 28
        
        # One-glance verdict
        stable = True
        if lm_jitter > 3.0 or lm_jitter < 0:
            stable = False
        if theta_drift > 0.01 or theta_drift < 0:
            stable = False
        if gpd_static > 1.0 or gpd_static < 0:
            stable = False
        
        verdict = "STABLE" if stable else "UNSTABLE"
        verdict_color = (0, 255, 0) if stable else (0, 0, 255)
        cv2.putText(frame, f"  STATUS: {verdict}", (px + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, verdict_color, 2)
