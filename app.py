#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AR MIRROR - PRODUCTION APP WITH NEURAL WARPING (PHASE 2)"""

import cv2
import numpy as np
import time
import sys
import os
import argparse
import threading
import tracemalloc
from pathlib import Path
from collections import deque
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Auto-register NVIDIA CUDA DLLs from pip-installed packages
# Search in .venv first, then any sibling venv dirs for backwards compatibility
_nvidia_search_dirs = [
    Path(__file__).parent / ".venv" / "Lib" / "site-packages" / "nvidia",
    Path(__file__).parent / "ar"   / "Lib" / "site-packages" / "nvidia",
]
for _nvidia_dir in _nvidia_search_dirs:
    if _nvidia_dir.exists():
        for _bin in _nvidia_dir.glob("*/bin"):
            if _bin.is_dir() and str(_bin) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")
        logger.debug("NVIDIA CUDA DLLs added to PATH from %s", _nvidia_dir)
        break

# Import Phase 2 Neural Pipeline
PHASE2_AVAILABLE = False
try:
    from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
    from src.core.live_pose_converter import LivePoseConverter, LiveBodySegmenter
    PHASE2_AVAILABLE = True
except Exception as e:
    logger.warning(f"Phase 2 neural pipeline not available: {e}")

# Legacy GMM warper removed — Phase 2 neural pipeline is the sole renderer

# Import rendering and overlay mixins
from src.app.rendering import GarmentRenderer, load_viton_cloth
from src.app.overlay import OverlayRenderer

# Import skeleton drawing for web stream
try:
    from tryon_selector import draw_skeleton_overlay
    SKELETON_DRAW_AVAILABLE = True
except Exception:
    SKELETON_DRAW_AVAILABLE = False

# Web UI server
try:
    from web_server import WebServer as _WebServer
    WEB_SERVER_AVAILABLE = True
except Exception as _ws_err:
    logger.warning(f"WebServer not available: {_ws_err}")
    WEB_SERVER_AVAILABLE = False

# ── Data Flywheel ────────────────────────────────────────────────────────────
FLYWHEEL_AVAILABLE = False
try:
    from src.core.session_logger import get_session_logger, BodyMeasurements, FitDecision, FrameMetadata
    from src.core.sku_bias_corrector import get_sku_corrector
    FLYWHEEL_AVAILABLE = True
except Exception as _fw_err:
    logger.warning(f"Data flywheel not available: {_fw_err}")


class ARMirrorApp(GarmentRenderer, OverlayRenderer):
    """AR Mirror application — orchestrates camera, pipeline init, and main loop.
    
    Rendering logic is in src.app.rendering.GarmentRenderer.
    Overlay/HUD logic is in src.app.overlay.OverlayRenderer.
    """

    def __init__(self, target_fps=30, demo_duration=120, phase=2):
        self.target_fps = target_fps
        self.demo_duration = demo_duration
        self.frame_times = deque(maxlen=30)
        self.frame_count = 0
        self.start_time: float = 0.0
        self.cap: Optional[cv2.VideoCapture] = None
        self.headless = False
        self.current_garment_idx = 0
        self.show_overlay = True
        self.render_tryon_overlay = True
        self.garments = []
        self.garment_images = []
        self.dataset_pairs = []
        self.pose = None
        self.mp_drawing = None
        
        # Phase selection
        if phase == 1:
            logger.warning("Phase 1 is deprecated; using Phase 0 fallback mode")
            self.phase = 0
        else:
            self.phase = phase
        self.phase2_pipeline = None
        self.gmm_warper = None
        self.body_fitter = None
        self.segmentation_fitter = None
        self.semantic_parser = None
        
        # Garment image cache
        self._garment_cache: dict = {}
        
        # Skip-frame optimization caches
        self._cached_body_parts: Optional[dict] = None
        self._semantic_frame_counter: int = 0
        self._semantic_skip_interval: int = 5
        self._cached_body_measurements: Optional[dict] = None
        self._pose_frame_counter: int = 0
        self._pose_skip_interval: int = 2

        # Lock-free double buffers (pointer swap) for hot-path state sharing
        self._capture_buffers: list[Optional[np.ndarray]] = [None, None]
        self._capture_active_idx = 0
        self._capture_seq = 0
        self._capture_ts = 0.0

        self._pose_buffers: list[Optional[dict]] = [None, None]
        self._pose_active_idx = 0
        self._pose_seq = 0
        self._pose_ts = 0.0
        self._pose_velocity_px = np.zeros(2, dtype=np.float32)
        self._last_pose_center: Optional[np.ndarray] = None

        # Temporal insulation thresholds (fresh/stale/decay)
        self._pose_fresh_s = 0.030
        self._pose_stale_s = 0.100
        self._pose_drop_s = 0.350
        self._pose_decay_tau_s = 0.120
        self._max_translation_px = 12.0
        self._max_scale_delta = 0.06
        self._max_angle_delta_deg = 5.0
        self._prev_warp_transform = None
        self._warp_budget_ms = 20.0
        self._warp_guard_hits = 0
        self._warp_guard_blend_alpha = 0.30

        self._seg_mask_buffers: list[Optional[np.ndarray]] = [None, None]
        self._seg_mask_active_idx = 0
        self._seg_seq = 0
        self._seg_ts = 0.0

        # Worker lifecycle
        self._stop_workers = threading.Event()
        self._capture_thread = None
        self._pose_thread = None
        self._seg_thread = None

        # Diagnostics and watchdog
        self._diag_capture_loop_ms = 0.0
        self._diag_pose_loop_ms = 0.0
        self._diag_pose_extract_ms = 0.0
        self._diag_seg_loop_ms = 0.0
        self._diag_render_ms = 0.0
        self._diag_pose_age_ms = 0.0
        self._diag_pose_mode = "fresh"
        self._diag_pose_clamped = False
        self._diag_warp_guard_reused = False
        self._last_frame_time_ms = 0.0
        self._diag_mem_baseline = None
        self._enable_memory_trace = os.environ.get("AR_TRACE_ALLOC", "0") == "1"
        self._frame_spike_count = 0
        self._frame_spikes = deque(maxlen=30)
        
        # Debug/feedback mode
        self.show_debug = False
        self._last_body_measurements: Optional[dict] = None

        # Web UI server
        self._web_server = None

        # Data flywheel
        self._session_logger = get_session_logger() if FLYWHEEL_AVAILABLE else None
        self._sku_corrector = get_sku_corrector() if FLYWHEEL_AVAILABLE else None
        self._current_session_id: Optional[str] = None
        
        # Per-stage timing accumulators for profiling
        self._stage_times: dict = {
            'pose_detect': [], 'neural_warp': [], 'semantic_parse': [],
            'composite': [], 'total_render': []
        }
        self._warp_sub_times: dict = {}
        
    def initialize(self):
        """Initialize the AR Mirror system"""
        print("\n" + "="*80)
        print("AR MIRROR - INITIALIZATION")
        print("="*80)
        
        try:
            print("\n[1/4] Initializing computer vision...")
            try:
                from mediapipe.tasks import vision  # type: ignore
                print("     [OK] MediaPipe loaded (optional for pose detection)")
            except:
                print("     [WARN] MediaPipe pose detection not available (mask-only mode)")
            
            print("[2/4] Initializing garment database...")
            self.garments = self._get_available_garments()
            self.dataset_pairs = self._load_dataset_pairs()
            print(f"     [OK] Found {len(self.garments)} garments")
            print(f"     [OK] Loaded {len(self.dataset_pairs)} dataset image pairs")
            
            # Initialize warping based on phase
            if self.phase == 2 and PHASE2_AVAILABLE:
                print("[3/4] Loading Phase 2 Neural Pipeline (GMM + TOM)...")
                try:
                    self.phase2_pipeline = Phase2NeuralPipeline(
                        device='auto',
                        enable_tom=False,
                        batch_size=1,
                        enable_optimizations=True
                    )
                    stats = self.phase2_pipeline.get_statistics()
                    print(f"     [OK] Phase 2 loaded on {stats['device']}")
                except Exception as e:
                    print(f"     [WARN] Phase 2 loading failed: {e}")
                    print(f"     [INFO] Falling back to Phase 0")
                    self.phase = 0
            else:
                print("[3/4] Phase 2 unavailable — using Phase 0 (simple blending)")
                self.phase = 0
            
            # Initialize body-aware fitter
            print("[3.5/4] Loading Body-Aware Fitter...")
            try:
                from src.core.body_aware_fitter import BodyAwareGarmentFitter
                self.body_fitter = BodyAwareGarmentFitter(output_segmentation_masks=False)
                self.segmentation_fitter = BodyAwareGarmentFitter(output_segmentation_masks=True)
                print("     [OK] Decoupled body fitters enabled (pose + segmentation)")
            except Exception as e:
                print(f"     [WARN] Body-aware fitter not available: {e}")
                self.body_fitter = None
                self.segmentation_fitter = None
            
            # Start data flywheel session
            if FLYWHEEL_AVAILABLE and self._session_logger:
                try:
                    sku = self.garments[self.current_garment_idx]["sku"] if self.garments else "UNKNOWN"
                    self._current_session_id = self._session_logger.start_session(
                        sku=sku, size_label="M"
                    )
                    logger.info(f"     [OK] Data flywheel session started: {self._current_session_id}")
                except Exception as _e:
                    logger.warning(f"     [WARN] Flywheel session failed to start: {_e}")

            # Start Web UI server
            if WEB_SERVER_AVAILABLE:
                self._web_server = _WebServer(port=5051)
                self._web_server.register_garment_list(
                    lambda: [g.get("file", g.get("name", "")) for g in self.garments]
                )
                self._web_server.register_garment_callback(self._on_web_garment_select)
                self._web_server.register_param_callback(self._on_web_params_update)
                self._web_server.start()
                if self.body_fitter or self.segmentation_fitter:
                    try:
                        init_height = self._web_server.get_param("user_height_cm")
                        init_height = float(init_height)
                        if self.body_fitter:
                            self.body_fitter.set_user_height_cm(init_height)
                        if self.segmentation_fitter:
                            self.segmentation_fitter.set_user_height_cm(init_height)
                    except (TypeError, ValueError):
                        pass
                    try:
                        init_square = self._web_server.get_param("calibration_square_cm")
                        init_square = float(init_square)
                        if self.body_fitter:
                            self.body_fitter.set_calibration_square_cm(init_square)
                        if self.segmentation_fitter:
                            self.segmentation_fitter.set_calibration_square_cm(init_square)
                    except (TypeError, ValueError):
                        pass
                print("     [OK] Web UI available at http://localhost:5051")
                print("          Open the React UI at http://localhost:3001")

            print("[4/4] Opening webcam...")
            # Try different camera backends for better Windows compatibility
            backends = [
                ("DirectShow", cv2.CAP_DSHOW),
                ("Media Foundation", cv2.CAP_MSMF),
                ("Default", cv2.CAP_ANY)
            ]

            self.cap = None
            for backend_name, backend_flag in backends:
                print(f"     Trying {backend_name} backend...")
                self.cap = cv2.VideoCapture(0, backend_flag)
                if self.cap.isOpened():
                    # Test if we can actually read frames
                    ret, test_frame = self.cap.read()
                    if ret:
                        print(f"     [OK] {backend_name} backend working")
                        break
                    else:
                        print(f"     [WARN] {backend_name} opened but no frames")
                        self.cap.release()
                        self.cap = None
                else:
                    print(f"     [WARN] {backend_name} backend failed to open")
                    if self.cap:
                        self.cap.release()
                        self.cap = None

            if not self.cap or not self.cap.isOpened():
                print("     [ERROR] Cannot open camera with any backend")
                print("     [TIP] Close other apps using camera (Edge, Teams, Zoom, etc.)")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # type: ignore
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # type: ignore
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # type: ignore

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore
            print(f"     [OK] Camera open at {width}x{height}")

            print("[5/5] Starting demo...")
            self.start_time = time.time()
            mode_map = {
                2: "PHASE 2: NEURAL WARPING (GMM+TOM)",
                0: "PHASE 0: ALPHA BLENDING"
            }
            mode_str = mode_map.get(self.phase, "UNKNOWN")
            print(f"     [OK] Ready to start [{mode_str}]")

            return True
            
        except Exception as e:
            print(f"     [ERROR] {e}")
            return False
    
    def _get_available_garments(self):
        """Get list of available garments from the real catalog first, then fallback assets."""
        def _has_real_photo_asset(image_path: str) -> bool:
            root = Path(image_path)
            if not root.is_absolute():
                root = Path(image_path)
            if not root.exists():
                return False
            if root.is_file():
                return root.suffix.lower() in {".png", ".jpg", ".jpeg", ".glb", ".gltf"}
            if not root.is_dir():
                return False
            for candidate in ["image.png", "image.jpg", "image.jpeg", "14274_00.jpg", "model.glb", "model.gltf"]:
                if (root / candidate).exists():
                    return True
            if list(root.glob("*.glb")) or list(root.glob("*.gltf")):
                return True
            return False

        inventory_files = [
            Path("assets/garments/garment_inventory.json"),
            Path("config/garment_inventory.json"),
        ]
        for inventory_file in inventory_files:
            if inventory_file.exists():
                try:
                    import json
                    with open(inventory_file, 'r', encoding='utf-8') as handle:
                        inventory = json.load(handle)
                    garments = []
                    for item in inventory:
                        if not isinstance(item, dict):
                            continue
                        sku = item.get("sku")
                        image_path = item.get("image_path")
                        if sku and image_path and _has_real_photo_asset(image_path):
                            garments.append({
                                **item,
                                "name": item.get("name") or sku,
                                "sku": sku,
                                "image_path": image_path,
                            })
                    if garments:
                        print(f"     [OK] Loaded {len(garments)} real garments from {inventory_file}")
                        return garments
                except Exception as e:
                    print(f"     [WARN] Could not load garment inventory {inventory_file}: {e}")

        cloth_dir = Path("dataset/train/cloth")
        if cloth_dir.exists():
            files = sorted(cloth_dir.glob("*.jpg"))[:200]  # cap at 200 for fast start
            garments = [
                {"name": f.stem, "sku": f.stem, "file": f.name}
                for f in files
            ]
            if garments:
                return garments
        # Fallback to colored placeholders
        return [
            {"name": "T-Shirt (Red)",   "sku": "TSH-001", "color": (0, 0, 255)},
            {"name": "T-Shirt (Blue)",  "sku": "TSH-002", "color": (255, 0, 0)},
            {"name": "Shirt (White)",   "sku": "SHT-001", "color": (255, 255, 255)},
            {"name": "Sweater",         "sku": "SWT-001", "color": (0, 165, 255)},
            {"name": "Jacket",          "sku": "JKT-001", "color": (100, 100, 100)},
            {"name": "Hoodie",          "sku": "HOD-001", "color": (75, 0, 130)},
        ]
    
    def _load_dataset_pairs(self):
        """Load VITON dataset garment pairs directly from cloth directory"""
        pairs = []
        cloth_dir = Path("dataset/train/cloth")
        mask_dir  = Path("dataset/train/cloth-mask")

        # Prefer train_pairs.txt if it exists
        dataset_file = Path("dataset/train_pairs.txt")
        if dataset_file.exists():
            try:
                with open(dataset_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            garment_path = cloth_dir / parts[1]
                            if garment_path.exists():
                                pairs.append({'person': parts[0], 'garment': parts[1]})
                                if len(pairs) >= 200:
                                    break
                if pairs:
                    return pairs
            except Exception as e:
                print(f"     [WARN] Could not read pairs file: {e}")

        # Fall back to scanning cloth directory directly
        if cloth_dir.exists():
            for f in sorted(cloth_dir.glob("*.jpg"))[:200]:
                # Only include garments that have a matching mask
                if mask_dir.exists() and not (mask_dir / f.name).exists():
                    continue
                pairs.append({'person': f.name, 'garment': f.name})
            if pairs:
                print(f"     [INFO] Loaded {len(pairs)} garments from dataset/train/cloth")
        else:
            print(f"     [WARN] Cloth directory not found: {cloth_dir}")

        return pairs

    def _get_capture_snapshot(self):
        idx = self._capture_active_idx
        frame = self._capture_buffers[idx]
        return self._capture_seq, frame, self._capture_ts

    def _torso_center(self, measurements: Optional[dict]) -> Optional[np.ndarray]:
        if not measurements:
            return None
        torso = measurements.get('torso_box')
        if not torso or len(torso) != 4:
            return None
        x1, y1, x2, y2 = torso
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    def _shift_measurements(self, measurements: dict, shift_xy: np.ndarray, frame_shape) -> dict:
        h, w = frame_shape[:2]
        dx = int(round(float(shift_xy[0])))
        dy = int(round(float(shift_xy[1])))
        shifted = dict(measurements)

        torso = shifted.get('torso_box')
        if torso and len(torso) == 4:
            x1, y1, x2, y2 = torso
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            nx1 = int(np.clip(x1 + dx, 0, max(0, w - bw)))
            ny1 = int(np.clip(y1 + dy, 0, max(0, h - bh)))
            shifted['torso_box'] = (nx1, ny1, nx1 + bw, ny1 + bh)

        return shifted

    def _get_pose_snapshot(self):
        pose = self._pose_buffers[self._pose_active_idx]
        if not pose:
            return None, 0.0
        seg_mask = self._seg_mask_buffers[self._seg_mask_active_idx]
        if seg_mask is None:
            return pose, self._pose_ts
        merged = dict(pose)
        merged['body_mask'] = seg_mask
        return merged, self._pose_ts

    def _get_temporally_stable_measurements(self, frame_shape):
        pose, pose_ts = self._get_pose_snapshot()
        if not pose or pose_ts <= 0.0:
            return None, 0.0, "none"

        now = time.perf_counter()
        dt = max(0.0, now - pose_ts)
        velocity = self._pose_velocity_px

        if dt <= self._pose_fresh_s:
            return pose, dt, "fresh"

        if dt <= self._pose_stale_s:
            shift = velocity * dt
            predicted = self._shift_measurements(pose, shift, frame_shape)
            return predicted, dt, "extrapolate"

        # Very stale: decay motion and gradually freeze
        if dt <= self._pose_drop_s:
            decay = float(np.exp(-(dt - self._pose_stale_s) / max(self._pose_decay_tau_s, 1e-3)))
            effective_dt = self._pose_stale_s + (dt - self._pose_stale_s) * decay
            shift = velocity * effective_dt
            predicted = self._shift_measurements(pose, shift, frame_shape)
            return predicted, dt, "decay"

        # Too stale: keep last stable pose without additional motion
        return pose, dt, "freeze"

    def _clamp_pose_deltas(self, measurements: Optional[dict], frame_shape):
        """Bound translation/scale/angle change per frame before warp stage."""
        if not measurements:
            self._diag_pose_clamped = False
            return measurements

        center = self._torso_center(measurements)
        torso = measurements.get('torso_box')
        if center is None or not torso or len(torso) != 4:
            self._diag_pose_clamped = False
            return measurements

        x1, y1, x2, y2 = torso
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        current_scale = float(measurements.get('shoulder_width', bw))
        current_angle = float(measurements.get('yaw_deg', 0.0))

        if self._prev_warp_transform is None:
            self._prev_warp_transform = {
                'center': center,
                'scale': current_scale,
                'angle': current_angle,
                'bw': bw,
                'bh': bh,
            }
            self._diag_pose_clamped = False
            return measurements

        prev = self._prev_warp_transform

        # Clamp translation magnitude
        dxy = center - prev['center']
        dxy_norm = float(np.linalg.norm(dxy))
        if dxy_norm > self._max_translation_px:
            dxy = (dxy / max(dxy_norm, 1e-6)) * self._max_translation_px
        clamped_center = prev['center'] + dxy

        # Clamp multiplicative scale change
        if prev['scale'] > 1e-6 and current_scale > 1e-6:
            ratio = current_scale / prev['scale']
            ratio = float(np.clip(ratio, 1.0 - self._max_scale_delta, 1.0 + self._max_scale_delta))
            clamped_scale = prev['scale'] * ratio
        else:
            clamped_scale = current_scale

        # Clamp angle delta (degrees)
        angle_delta = current_angle - prev['angle']
        angle_delta = float(np.clip(angle_delta, -self._max_angle_delta_deg, self._max_angle_delta_deg))
        clamped_angle = prev['angle'] + angle_delta

        scale_factor = float(clamped_scale / max(current_scale, 1e-6))
        out = dict(measurements)
        for key in ('shoulder_width', 'torso_height', 'chest_width', 'waist_width', 'hip_width'):
            if key in out and out[key] is not None:
                out[key] = float(max(1.0, float(out[key]) * scale_factor))

        # Keep torso box coherent with clamped center + scale
        new_bw = max(1, int(round(bw * scale_factor)))
        new_bh = max(1, int(round(bh * scale_factor)))
        h, w = frame_shape[:2]
        nx1 = int(np.clip(round(float(clamped_center[0]) - new_bw * 0.5), 0, max(0, w - new_bw)))
        ny1 = int(np.clip(round(float(clamped_center[1]) - new_bh * 0.5), 0, max(0, h - new_bh)))
        out['torso_box'] = (nx1, ny1, nx1 + new_bw, ny1 + new_bh)
        out['yaw_deg'] = clamped_angle

        self._prev_warp_transform = {
            'center': clamped_center,
            'scale': clamped_scale,
            'angle': clamped_angle,
            'bw': float(new_bw),
            'bh': float(new_bh),
        }
        self._diag_pose_clamped = bool(
            dxy_norm > self._max_translation_px
            or abs(current_scale - clamped_scale) > 1e-3
            or abs(current_angle - clamped_angle) > 1e-3
        )
        return out

    def _get_latest_body_measurements(self):
        pose = self._pose_buffers[self._pose_active_idx]
        if not pose:
            return None
        seg_mask = self._seg_mask_buffers[self._seg_mask_active_idx]
        if seg_mask is None:
            return pose
        merged = dict(pose)
        merged['body_mask'] = seg_mask
        return merged

    def _capture_worker_loop(self):
        while not self._stop_workers.is_set():
            t0 = time.perf_counter()
            if not self.cap:
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            next_idx = 1 - self._capture_active_idx
            self._capture_buffers[next_idx] = frame
            self._capture_seq += 1
            self._capture_ts = time.perf_counter()
            self._capture_active_idx = next_idx
            self._diag_capture_loop_ms = (time.perf_counter() - t0) * 1000.0

    def _pose_worker_loop(self):
        last_seq = -1
        while not self._stop_workers.is_set():
            t0 = time.perf_counter()
            if not self.body_fitter:
                time.sleep(0.02)
                continue

            seq, frame, _ = self._get_capture_snapshot()
            if frame is None or seq == last_seq or (seq - last_seq) < self._pose_skip_interval:
                time.sleep(0.003)
                continue

            t_extract = time.perf_counter()
            measurements = self.body_fitter.extract_body_measurements(frame)
            self._diag_pose_extract_ms = (time.perf_counter() - t_extract) * 1000.0
            if measurements is not None:
                now = time.perf_counter()
                center = self._torso_center(measurements)
                if center is not None and self._last_pose_center is not None and self._pose_ts > 0.0:
                    dt = max(1e-3, now - self._pose_ts)
                    vel = (center - self._last_pose_center) / dt
                    self._pose_velocity_px = (0.7 * self._pose_velocity_px + 0.3 * vel).astype(np.float32)
                if center is not None:
                    self._last_pose_center = center

                next_idx = 1 - self._pose_active_idx
                self._pose_buffers[next_idx] = measurements
                self._pose_seq = seq
                self._pose_ts = now
                self._pose_active_idx = next_idx

            last_seq = seq
            self._diag_pose_loop_ms = (time.perf_counter() - t0) * 1000.0

    def _seg_worker_loop(self):
        last_seq = -1
        skip_frames_remaining = 0
        while not self._stop_workers.is_set():
            t0 = time.perf_counter()
            if not self.segmentation_fitter:
                time.sleep(0.05)
                continue

            if skip_frames_remaining > 0:
                skip_frames_remaining -= 1
                time.sleep(0.005)
                continue

            seq, frame, _ = self._get_capture_snapshot()
            if frame is None or seq == last_seq or (seq - last_seq) < self._semantic_skip_interval:
                time.sleep(0.02)
                continue

            measurements = self.segmentation_fitter.extract_body_measurements(frame)
            seg_mask = measurements.get('body_mask') if measurements else None
            if seg_mask is not None:
                next_idx = 1 - self._seg_mask_active_idx
                self._seg_mask_buffers[next_idx] = seg_mask
                self._seg_seq = seq
                self._seg_ts = time.perf_counter()
                self._seg_mask_active_idx = next_idx

            last_seq = seq
            self._diag_seg_loop_ms = (time.perf_counter() - t0) * 1000.0
            if self._diag_seg_loop_ms > 100.0:
                skip_frames_remaining = max(
                    skip_frames_remaining,
                    int(np.ceil(self._diag_seg_loop_ms / 33.0)) - 1,
                )
            else:
                skip_frames_remaining = max(skip_frames_remaining, self._semantic_skip_interval - 1)
            time.sleep(0.005)

    def _start_pipeline_workers(self):
        self._stop_workers.clear()
        self._capture_thread = threading.Thread(target=self._capture_worker_loop, daemon=True)
        self._pose_thread = threading.Thread(target=self._pose_worker_loop, daemon=True)
        self._seg_thread = threading.Thread(target=self._seg_worker_loop, daemon=True)
        self._capture_thread.start()
        self._pose_thread.start()
        self._seg_thread.start()

    def _stop_pipeline_workers(self):
        self._stop_workers.set()
        for t in (self._capture_thread, self._pose_thread, self._seg_thread):
            if t and t.is_alive():
                t.join(timeout=0.8)

    def _log_frame_spike(self, frame_time_s: float):
        self._frame_spike_count += 1
        render_diag = getattr(self, '_render_stage_diag', {}) or {}
        spike = {
            'frame_ms': round(frame_time_s * 1000.0, 1),
            'pose_loop_ms': round(self._diag_pose_loop_ms, 1),
            'pose_extract_ms': round(self._diag_pose_extract_ms, 1),
            'seg_loop_ms': round(self._diag_seg_loop_ms, 1),
            'capture_loop_ms': round(self._diag_capture_loop_ms, 1),
            'render_ms': round(self._diag_render_ms, 1),
            'render_pre_ms': round(float(render_diag.get('pre_ms', 0.0)), 1),
            'render_warp_ms': round(float(render_diag.get('warp_ms', 0.0)), 1),
            'render_comp_ms': round(float(render_diag.get('comp_ms', 0.0)), 1),
            'render_post_ms': round(float(render_diag.get('post_ms', 0.0)), 1),
            'render_fit_post_ms': round(float(render_diag.get('fit_post_ms', 0.0)), 1),
            'render_fit_roi_area': round(float(render_diag.get('fit_roi_area', 0.0)), 1),
            'render_fit_roi_ratio': round(float(render_diag.get('fit_roi_ratio', 0.0)), 4),
            'pose_age_ms': round(self._diag_pose_age_ms, 1),
            'pose_mode': self._diag_pose_mode,
            'pose_clamped': self._diag_pose_clamped,
            'warp_guard': bool(render_diag.get('warp_guard_reused', False)),
        }
        if self._enable_memory_trace and self._diag_mem_baseline is not None:
            try:
                mem_after = tracemalloc.take_snapshot()
                diffs = mem_after.compare_to(self._diag_mem_baseline, 'filename')
                alloc_bytes = sum(s.size_diff for s in diffs if s.size_diff > 0)
                spike['alloc_kb_since_start'] = round(float(alloc_bytes) / 1024.0, 1)
            except Exception:
                pass
        self._frame_spikes.append(spike)
        if self._frame_spike_count <= 5:
            logger.warning(f"[FRAME SPIKE] {spike}")

    def _log_memory_audit(self, frame_bytes: int, seq: int):
        if not self._enable_memory_trace or not tracemalloc.is_tracing():
            return
        try:
            current, peak = tracemalloc.get_traced_memory()
            logger.info(
                "[MEM AUDIT] frame_bytes=%d current_kb=%.1f peak_kb=%.1f seq=%d",
                frame_bytes,
                current / 1024.0,
                peak / 1024.0,
                seq,
            )
        except Exception:
            pass
    
    def run(self):
        """Run the main demo loop"""
        print("\n" + "="*80)
        print("STARTING LIVE AR MIRROR DEMO")
        print("="*80)
        _infinite = self.demo_duration <= 0
        _dur_str = "until stopped" if _infinite else f"{self.demo_duration} seconds"
        print(f"\nRunning for {_dur_str}...")
        print("Press 'q' to quit, arrows to change garments")
        print("Press 'd' to toggle debug overlay (skeleton + detection status)")
        print("Press 'o' to toggle info overlay\n")
        
        demo_end_time = time.time() + (self.demo_duration if not _infinite else 1e18)
        self._warp_guard_hits = 0

        if self._enable_memory_trace:
            if not tracemalloc.is_tracing():
                tracemalloc.start(10)
            try:
                self._diag_mem_baseline = tracemalloc.take_snapshot()
            except Exception:
                self._diag_mem_baseline = None
        else:
            self._diag_mem_baseline = None

        self._start_pipeline_workers()
        # Warm up render preprocess paths outside the measured hot loop.
        if self.garments:
            try:
                _warm = self.garments[self.current_garment_idx]
                self._load_garment_image(_warm)
            except Exception:
                pass
        try:
            self._get_holistic_tracker()
        except Exception:
            pass
        try:
            consecutive_failures = 0
            while time.time() < demo_end_time:
                frame_start = time.time()

                _seq, source_frame, _ts = self._get_capture_snapshot()
                if source_frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 60:
                        print(f"[ERROR] No capture frames for {consecutive_failures} loops, exiting")
                        break
                    time.sleep(0.01)
                    continue
                consecutive_failures = 0
                frame = source_frame

                try:
                    display_frame = cv2.flip(frame, 1)
                    garment = self.garments[self.current_garment_idx]

                    body_measurements, pose_age_s, pose_mode = self._get_temporally_stable_measurements(frame.shape)
                    self._diag_pose_age_ms = pose_age_s * 1000.0
                    self._diag_pose_mode = pose_mode
                    body_measurements = self._clamp_pose_deltas(body_measurements, frame.shape)
                    self._cached_body_measurements = body_measurements

                    # Apply SKU bias correction to cached body measurements
                    if self._sku_corrector and self._cached_body_measurements:
                        try:
                            raw = self._cached_body_measurements
                            corrected = self._sku_corrector.apply(
                                sku=garment["sku"],
                                shoulder_cm=raw.get("shoulder_width_cm", 0),
                                chest_cm=raw.get("chest_cm", 0),
                                hip_cm=raw.get("hip_cm", 0),
                                length_cm=raw.get("torso_length_cm", 0),
                            )
                            self._cached_body_measurements = {**raw, **corrected}
                            body_measurements = self._cached_body_measurements
                        except Exception:
                            pass

                    if self.render_tryon_overlay:
                        if self.body_fitter and hasattr(self.body_fitter, 'set_runtime_frame_time_ms'):
                            self.body_fitter.set_runtime_frame_time_ms(self._last_frame_time_ms)
                        if self.segmentation_fitter and hasattr(self.segmentation_fitter, 'set_runtime_frame_time_ms'):
                            self.segmentation_fitter.set_runtime_frame_time_ms(self._last_frame_time_ms)
                        t_render = time.perf_counter()
                        display_frame = self._render_garment(display_frame, garment, body_measurements=body_measurements)
                        self._diag_render_ms = (time.perf_counter() - t_render) * 1000.0
                        render_diag = getattr(self, '_render_stage_diag', {}) or {}
                        self._diag_warp_guard_reused = bool(render_diag.get('warp_guard_reused', False))
                        if self._diag_warp_guard_reused:
                            self._warp_guard_hits += 1
                    output_frame = display_frame

                    # Log measurements to flywheel every 30 frames
                    if (FLYWHEEL_AVAILABLE and self._session_logger
                            and self._current_session_id
                            and self._cached_body_measurements
                            and self.frame_count % 30 == 0):
                        try:
                            bm = BodyMeasurements(
                                shoulder_width_cm=self._cached_body_measurements.get("shoulder_width_cm", 0.0),
                                chest_width_cm=self._cached_body_measurements.get("chest_cm", 0.0),
                                torso_length_cm=self._cached_body_measurements.get("torso_length_cm", 0.0),
                                hip_width_cm=self._cached_body_measurements.get("hip_cm", 0.0),
                                height_cm=self._cached_body_measurements.get("height_cm", 0.0),
                                depth_backend=self._cached_body_measurements.get("depth_backend", "geometric"),
                            )
                            fd = FitDecision(
                                overall="GOOD",
                                shoulder="UNKNOWN",
                            )
                            _depth_proxy = getattr(self, '_last_depth_proxy', 0.0)
                            fm = FrameMetadata(
                                distance_proxy=_depth_proxy,
                                pose_confidence=self._cached_body_measurements.get("confidence", 0.0),
                                frame_width=frame.shape[1],
                                frame_height=frame.shape[0],
                            )
                            self._session_logger.log_measurements(
                                self._current_session_id, bm, fd, fm
                            )
                        except Exception:
                            pass

                except Exception as e:
                    logger.warning(f"Render error: {e}")
                    output_frame = frame

                # Push frame and state to WebUI
                if self._web_server:
                    try:
                        _ws_fps = float(1.0 / np.mean(list(self.frame_times))) if self.frame_times else 0.0
                        _ws_gname = self.garments[self.current_garment_idx].get(
                            "file", self.garments[self.current_garment_idx].get("name", "")
                        )
                        _ws_meas = self._cached_body_measurements

                        # Draw skeleton overlay on a single copied web frame.
                        web_frame = output_frame.copy()
                        show_skeleton_overlay = True
                        try:
                            show_skeleton_overlay = bool(self._web_server.get_param("show_skeleton"))
                        except Exception:
                            show_skeleton_overlay = True

                        if SKELETON_DRAW_AVAILABLE and _ws_meas and show_skeleton_overlay:
                            try:
                                draw_skeleton_overlay(web_frame, _ws_meas)
                            except Exception:
                                pass

                        self._web_server.push_frame(web_frame)
                        self._web_server.push_state(_ws_fps, _ws_gname, _ws_meas)
                    except Exception:
                        pass

                # Draw overlay
                if self.show_overlay:
                    display_frame = self._draw_overlay(output_frame, None)
                else:
                    display_frame = output_frame

                # Show frame (skip in headless mode)
                if not getattr(self, 'headless', False):
                    cv2.imshow("AR MIRROR", display_frame)

                # Record timing and watchdog diagnostics
                frame_time = time.time() - frame_start
                self._last_frame_time_ms = frame_time * 1000.0
                if frame_time > 0.100:
                    self._log_frame_spike(frame_time)
                self.frame_times.append(frame_time)
                self.frame_count += 1

                if self._enable_memory_trace and self.frame_count % 180 == 0:
                    self._log_memory_audit(source_frame.nbytes if hasattr(source_frame, 'nbytes') else 0, self._capture_seq)

                # Handle keyboard (skip in headless mode)
                if getattr(self, 'headless', False):
                    # In headless mode, just sleep briefly to control frame rate
                    time.sleep(0.033)  # ~30 FPS cap
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('o'):
                        self.show_overlay = not self.show_overlay
                    elif key == ord('d'):
                        self.show_debug = not self.show_debug
                        print(f"\nDebug overlay: {'ON' if self.show_debug else 'OFF'}")
                    elif key == 261 or key == ord('\t'):  # Right arrow or Tab
                        if self.dataset_pairs:
                            self.current_garment_idx = (self.current_garment_idx + 1) % len(self.dataset_pairs)
                        else:
                            self.current_garment_idx = (self.current_garment_idx + 1) % len(self.garments)
                        print(f"\nSwitched to garment #{self.current_garment_idx + 1}")
                        self._on_garment_change()
                    elif key == 260:  # Left arrow
                        if self.dataset_pairs:
                            self.current_garment_idx = (self.current_garment_idx - 1) % len(self.dataset_pairs)
                        else:
                            self.current_garment_idx = (self.current_garment_idx - 1) % len(self.garments)
                        print(f"\nSwitched to garment #{self.current_garment_idx + 1}")
                        self._on_garment_change()
                
                # Print progress
                if self.frame_count % 10 == 0:
                    avg_fps = 1.0 / np.mean(list(self.frame_times)) if self.frame_times else 0
                    elapsed = time.time() - self.start_time
                    if self.demo_duration > 0:
                        progress = int((elapsed / self.demo_duration) * 50)
                        bar = "[" + "=" * progress + " " * (50 - progress) + "]"
                        print(f"\r {bar} {elapsed:.0f}s/{self.demo_duration}s | FPS: {avg_fps:.1f}", end="", flush=True)
                    else:
                        print(f"\r  [{elapsed:.0f}s] elapsed | FPS: {avg_fps:.1f}", end="", flush=True)
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self._stop_pipeline_workers()
            self._cleanup()
            self._print_results()
    
    def _on_web_garment_select(self, name: str):
        """Called from the WebServer thread when user selects a garment in the browser UI."""
        for i, g in enumerate(self.garments):
            if g.get("sku") == name or g.get("file") == name or g.get("name") == name or g.get("image_path") == name:
                self.current_garment_idx = i
                self._on_garment_change()
                logger.info(f"[Web] Garment switched → {name} (idx={i})")
                return
        logger.warning(f"[Web] Garment not found: {name}")

    def _on_web_params_update(self, updates: dict):
        """Apply web-exposed runtime params to rendering controls."""
        if "render_tryon_overlay" in updates:
            self.render_tryon_overlay = bool(updates.get("render_tryon_overlay"))
        if "user_height_cm" in updates:
            try:
                user_h_raw = updates.get("user_height_cm")
                if user_h_raw is None:
                    raise ValueError("missing user_height_cm")
                user_h = float(user_h_raw)
                if self.body_fitter:
                    self.body_fitter.set_user_height_cm(user_h)
                if self.segmentation_fitter:
                    self.segmentation_fitter.set_user_height_cm(user_h)
            except (TypeError, ValueError):
                pass
        if "calibration_square_cm" in updates:
            try:
                square_raw = updates.get("calibration_square_cm")
                if square_raw is None:
                    raise ValueError("missing calibration_square_cm")
                square_cm = float(square_raw)
                if self.body_fitter:
                    self.body_fitter.set_calibration_square_cm(square_cm)
                if self.segmentation_fitter:
                    self.segmentation_fitter.set_calibration_square_cm(square_cm)
            except (TypeError, ValueError):
                pass

    def _on_garment_change(self):
        """Close current flywheel session and open a new one for the new garment."""
        if not (FLYWHEEL_AVAILABLE and self._session_logger):
            return
        try:
            if self._current_session_id:
                self._session_logger.close_session(self._current_session_id)
            sku = self.garments[self.current_garment_idx]["sku"]
            self._current_session_id = self._session_logger.start_session(
                sku=sku, size_label="M"
            )
        except Exception as _e:
            logger.debug(f"Flywheel garment-change error: {_e}")

    def _cleanup(self):
        self._stop_pipeline_workers()
        # Close flywheel session
        if FLYWHEEL_AVAILABLE and self._session_logger and self._current_session_id:
            try:
                self._session_logger.close_session(self._current_session_id)
            except Exception:
                pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def _print_results(self):
        print("\n\n" + "="*80)
        print("DEMO COMPLETE")
        print("="*80)
        
        elapsed = time.time() - self.start_time
        
        if self.frame_times:
            avg_latency = np.mean(list(self.frame_times))
            min_latency = np.min(list(self.frame_times))
            max_latency = np.max(list(self.frame_times))
            
            avg_fps = 1.0 / avg_latency if avg_latency > 0 else 0
            max_fps = 1.0 / min_latency if min_latency > 0 else 0
            min_fps = 1.0 / max_latency if max_latency > 0 else 0
        else:
            avg_fps = max_fps = min_fps = 0
            avg_latency = 0
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"   Total Frames: {self.frame_count}")
        print(f"   Duration: {elapsed:.2f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Max FPS: {max_fps:.1f}")
        print(f"   Min FPS: {min_fps:.1f}")
        print(f"   Avg Latency: {avg_latency*1000:.1f}ms")
        
        print(f"\nSTATUS:")
        if avg_fps >= 14.0:
            print(f"   EXCELLENT - {avg_fps:.1f} FPS baseline met")
        elif avg_fps >= 10.0:
            print(f"   GOOD - {avg_fps:.1f} FPS acceptable")
        else:
            print(f"   NEEDS OPTIMIZATION - {avg_fps:.1f} FPS")

        print(f"\nFRAME WATCHDOG:")
        print(f"   Spikes >100ms: {self._frame_spike_count}")
        print(f"   Warp guard reuses: {self._warp_guard_hits}")
        reuse_rate = (100.0 * self._warp_guard_hits / max(1, self.frame_count))
        print(f"   Warp guard reuse rate: {reuse_rate:.2f}%")
        if self._frame_spikes:
            worst = max(self._frame_spikes, key=lambda s: s['frame_ms'])
            culprit = max(
                ('pose_loop_ms', 'pose_extract_ms', 'seg_loop_ms', 'capture_loop_ms', 'render_ms'),
                key=lambda k: worst.get(k, 0.0),
            )
            print(f"   Worst spike: {worst['frame_ms']:.1f}ms")
            print(f"   Worst frame time: {worst['frame_ms']:.1f}ms")
            print(f"   Likely culprit: {culprit}={worst.get(culprit, 0.0):.1f}ms")
            print("   Render breakdown:")
            print(f"      preprocess: {worst.get('render_pre_ms', 0.0):.1f}ms")
            print(f"      warp      : {worst.get('render_warp_ms', 0.0):.1f}ms")
            print(f"      compose   : {worst.get('render_comp_ms', 0.0):.1f}ms")
            print(f"      post      : {worst.get('render_post_ms', 0.0):.1f}ms")
            print(f"      fit_post  : {worst.get('render_fit_post_ms', 0.0):.1f}ms")
            print(f"      roi area  : {worst.get('render_fit_roi_area', 0.0):.0f}")
            print(f"      roi ratio : {worst.get('render_fit_roi_ratio', 0.0):.3f}")
            print(f"      guard used: {bool(worst.get('warp_guard', False))}")
            if 'alloc_kb_since_start' in worst:
                print(f"   Alloc delta since start: {worst['alloc_kb_since_start']:.1f} KB")
        
        # Per-stage timing breakdown
        if any(len(v) > 0 for v in self._stage_times.values()):
            print(f"\nPER-STAGE TIMING (averages):")
            for stage, times in self._stage_times.items():
                if times:
                    avg_ms = np.mean(times) * 1000
                    p95_ms = np.percentile(times, 95) * 1000 if len(times) > 5 else avg_ms
                    print(f"   {stage:20s}: {avg_ms:6.1f}ms avg | {p95_ms:6.1f}ms p95 | {len(times)} calls")
        
        # Warp sub-stage timing
        if self._warp_sub_times:
            print(f"\n   GMM WARP SUB-STAGES:")
            for stage, times in self._warp_sub_times.items():
                if times:
                    avg_ms = np.mean(times) * 1000
                    print(f"     {stage:18s}: {avg_ms:6.1f}ms avg | {len(times)} calls")
        
        print(f"\nSKIP-FRAME CONFIG:")
        print(f"   Pose detection: every {self._pose_skip_interval} frames")
        print(f"   Semantic parsing: every {self._semantic_skip_interval} frames")
        
        print(f"\nROADMAP:")
        print(f"   Phase 0 (Fallback): 30+ FPS")
        print(f"   Phase 2 (Production): 20-27 FPS")
        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='AR Mirror - Virtual Try-On Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  Phase 0: Simple alpha blending (fastest, 30+ FPS)
  Phase 2: Neural warping with GMM+TOM (production, 21+ FPS)

Examples:
  python app.py                    # Use Phase 2 (default)
  python app.py --phase 2          # Neural warping
  python app.py --phase 0          # Simple blending
  python app.py --duration 60      # Run for 60 seconds
        """
    )
    parser.add_argument('--phase', type=int, default=2, choices=[0, 2],
                        help='Phase to use (0=blend, 2=neural)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS (default: 30)')
    parser.add_argument('--duration', type=int, default=0,
                        help='Demo duration in seconds; 0 keeps the app running until stopped (default: 0)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without display window (web server only)')

    args = parser.parse_args()

    app = ARMirrorApp(
        target_fps=args.fps,
        demo_duration=args.duration,
        phase=args.phase
    )
    app.headless = getattr(args, 'headless', False)
    
    if app.initialize():
        app.run()
    else:
        print("\nFailed to initialize")
        sys.exit(1)


if __name__ == "__main__":
    main()
