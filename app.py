#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AR MIRROR - PRODUCTION APP WITH NEURAL WARPING (PHASE 2)"""

import cv2
import numpy as np
import time
import sys
import os
import argparse
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
                self.body_fitter = BodyAwareGarmentFitter()
                print("     [OK] Body-aware fitting enabled!")
            except Exception as e:
                print(f"     [WARN] Body-aware fitter not available: {e}")
                self.body_fitter = None
            
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
                if self.body_fitter:
                    try:
                        init_height = self._web_server.get_param("user_height_cm")
                        self.body_fitter.set_user_height_cm(float(init_height))
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
        
        try:
            consecutive_failures = 0
            while time.time() < demo_end_time:
                frame_start = time.time()

                ret, frame = self.cap.read()  # type: ignore
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures == 10:  # First warning
                        print(f"[WARN] Camera read failures: {consecutive_failures}/30")
                        print("       This often means another app is using the camera")
                        print("       Try closing: Edge browser tabs, Teams, Zoom, Skype, Discord")
                    elif consecutive_failures == 20:  # Second warning
                        print(f"[WARN] Camera read failures: {consecutive_failures}/30")
                        print("       Check Windows Privacy Settings > Camera > Allow desktop apps")
                    if consecutive_failures > 30:  # ~1 second of failures
                        print(f"[ERROR] Camera read failed {consecutive_failures} times, exiting")
                        print("        Camera is likely being used by another application")
                        print("        or blocked by Windows privacy settings")
                        break
                    time.sleep(0.033)  # Wait and retry
                    continue
                consecutive_failures = 0  # Reset on success
                
                try:
                    display_frame = cv2.flip(frame, 1)
                    garment = self.garments[self.current_garment_idx]

                    # Extract body measurements (MISSING CODE - this is why size recommendations don't work!)
                    print(f"[DEBUG] Frame {self.frame_count}: body_fitter available: {self.body_fitter is not None}")
                    if self.body_fitter:
                        try:
                            print(f"[DEBUG] Calling body_fitter.extract_body_measurements...")
                            body_measurements = self.body_fitter.extract_body_measurements(frame)
                            print(f"[DEBUG] Body measurements result: {body_measurements is not None}")
                            if body_measurements:
                                self._cached_body_measurements = body_measurements
                                print(f"[BODY] Extracted measurements: {list(body_measurements.keys())}")
                                if body_measurements.get('size_recommendation'):
                                    print(f"[SIZE] Recommended size: {body_measurements['size_recommendation']} "
                                          f"(confidence: {body_measurements.get('size_confidence', 0):.2f})")
                            else:
                                print(f"[DEBUG] No body measurements extracted")
                        except Exception as e:
                            print(f"[BODY] Measurement extraction failed: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"[DEBUG] No body_fitter available")

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
                        except Exception:
                            pass

                    if self.render_tryon_overlay:
                        display_frame = self._render_garment(display_frame, garment)
                    output_frame = display_frame.copy()

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
                        _ws_fps = 1.0 / np.mean(list(self.frame_times)) if self.frame_times else 0.0
                        _ws_gname = self.garments[self.current_garment_idx].get(
                            "file", self.garments[self.current_garment_idx].get("name", "")
                        )
                        _ws_meas = self._cached_body_measurements

                        # Draw skeleton overlay on web stream frame
                        web_frame = output_frame.copy()
                        show_skeleton_overlay = True
                        if self._web_server:
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

                # Record timing
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                self.frame_count += 1

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
        if "user_height_cm" in updates and self.body_fitter:
            try:
                self.body_fitter.set_user_height_cm(float(updates.get("user_height_cm")))
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
    parser.add_argument('--duration', type=int, default=120,
                        help='Demo duration in seconds (default: 120)')
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
