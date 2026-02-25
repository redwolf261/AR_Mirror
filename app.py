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

# Import Semantic Parser for Occlusion Handling
SEMANTIC_PARSING_AVAILABLE = False
try:
    from src.core.semantic_parser import SemanticParser, create_occlusion_aware_composite
    SEMANTIC_PARSING_AVAILABLE = True
except Exception as e:
    logger.warning(f"Semantic parsing not available: {e}")

# Import legacy GMM warper (fallback)
GMM_AVAILABLE = False
try:
    sys.path.insert(0, str(Path(__file__).parent / "scripts" / "utilities"))
    sys.path.insert(0, str(Path(__file__).parent / "cp-vton"))  # For networks module
    from gmm_warper import GMMWarper, build_agnostic_representation
    from convert_pose_map import load_openpose_json, get_pose_map
    GMM_AVAILABLE = True
except Exception as e:
    logger.debug(f"Legacy GMM warper not available: {e}")

# Import rendering and overlay mixins
from src.app.rendering import GarmentRenderer, load_viton_cloth
from src.app.overlay import OverlayRenderer

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
        self.garments = []
        self.garment_images = []
        self.dataset_pairs = []
        self.pose = None
        self.mp_drawing = None
        
        # Phase selection
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
                    print(f"     [INFO] Falling back to Phase 1")
                    self.phase = 1
            elif self.phase == 1 and GMM_AVAILABLE:
                print("[3/4] Loading Phase 1 GMM warper...")
                try:
                    self.gmm_warper = GMMWarper()
                    print("     [OK] Phase 1 GMM loaded")
                except Exception as e:
                    print(f"     [WARN] GMM loading failed: {e}")
                    print("     [INFO] Using simple blending")
                    self.phase = 0
            else:
                print("[3/4] Using Phase 0 (simple blending)")
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
            
            # Initialize semantic parser for proper occlusion
            if SEMANTIC_PARSING_AVAILABLE:
                print("[3.6/4] Initializing Semantic Parser...")
                try:
                    self.semantic_parser = SemanticParser(
                        backend='auto',
                        temporal_smoothing=True,
                        onnx_model_path='models/schp_lip.onnx'
                    )
                    backend_name = self.semantic_parser.backend.__class__.__name__
                    print(f"     [OK] Semantic parser loaded ({backend_name})")
                except Exception as e:
                    print(f"     [WARN] Semantic parser failed: {e}")
                    self.semantic_parser = None
            
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

            print("[4/4] Opening webcam...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():  # type: ignore
                print("     [ERROR] Cannot open camera")
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
                1: "PHASE 1: GMM TPS WARPING",
                0: "PHASE 0: ALPHA BLENDING"
            }
            mode_str = mode_map.get(self.phase, "UNKNOWN")
            print(f"     [OK] Ready to start [{mode_str}]")
            
            return True
            
        except Exception as e:
            print(f"     [ERROR] {e}")
            return False
    
    def _get_available_garments(self):
        """Get list of available garments"""
        garments = [
            {"name": "T-Shirt (Red)", "sku": "TSH-001", "color": (0, 0, 255)},
            {"name": "T-Shirt (Blue)", "sku": "TSH-002", "color": (255, 0, 0)},
            {"name": "Shirt (White)", "sku": "SHT-001", "color": (255, 255, 255)},
            {"name": "Sweater", "sku": "SWT-001", "color": (0, 165, 255)},
            {"name": "Jacket", "sku": "JKT-001", "color": (100, 100, 100)},
            {"name": "Hoodie", "sku": "HOD-001", "color": (75, 0, 130)},
        ]
        return garments
    
    def _load_dataset_pairs(self):
        """Load VITON dataset image pairs"""
        pairs = []
        dataset_file = Path("dataset/train_pairs.txt")
        
        if dataset_file.exists():
            try:
                with open(dataset_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            person_path = Path(f"dataset/train/image/{parts[0]}")
                            garment_path = Path(f"dataset/train/cloth/{parts[1]}")
                            
                            if person_path.exists() and garment_path.exists():
                                pairs.append({
                                    'person': parts[0],
                                    'garment': parts[1]
                                })
                                if len(pairs) >= 50:
                                    break
            except Exception as e:
                print(f"     [WARN] Could not load dataset: {e}")
        else:
            print(f"     [WARN] Dataset file not found: {dataset_file}")
        
        return pairs
    
    def run(self):
        """Run the main demo loop"""
        print("\n" + "="*80)
        print("STARTING LIVE AR MIRROR DEMO")
        print("="*80)
        print(f"\nRunning for {self.demo_duration} seconds...")
        print("Press 'q' to quit, arrows to change garments")
        print("Press 'd' to toggle debug overlay (skeleton + detection status)")
        print("Press 'o' to toggle info overlay\n")
        
        demo_end_time = time.time() + self.demo_duration
        
        try:
            while time.time() < demo_end_time:
                frame_start = time.time()
                
                ret, frame = self.cap.read()  # type: ignore
                if not ret:
                    break
                
                try:
                    display_frame = cv2.flip(frame, 1)
                    garment = self.garments[self.current_garment_idx]

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
                
                # Draw overlay
                if self.show_overlay:
                    display_frame = self._draw_overlay(output_frame, None)
                else:
                    display_frame = output_frame
                
                # Show frame
                cv2.imshow("AR MIRROR", display_frame)
                
                # Record timing
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                self.frame_count += 1
                
                # Handle keyboard
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
                    progress = int((elapsed / self.demo_duration) * 50)
                    bar = "[" + "=" * progress + " " * (50 - progress) + "]"
                    print(f"\r {bar} {elapsed:.0f}s/{self.demo_duration}s | FPS: {avg_fps:.1f}", end="", flush=True)
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self._cleanup()
            self._print_results()
    
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
        print(f"   Phase 1 (Current): 14.0 FPS")
        print(f"   Phase 2A (GPU): 18-22 FPS")
        print(f"   Phase 2B (Neural): 20-27 FPS")
        print(f"   Phase 3 (Temporal): 15-18 FPS + stability")
        print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='AR Mirror - Virtual Try-On Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  Phase 0: Simple alpha blending (fastest, 30+ FPS)
  Phase 1: GMM TPS warping (legacy, 15-20 FPS)
  Phase 2: Neural warping with GMM+TOM (production, 21+ FPS)

Examples:
  python app.py                    # Use Phase 2 (default)
  python app.py --phase 2          # Neural warping
  python app.py --phase 1          # Legacy GMM
  python app.py --phase 0          # Simple blending
  python app.py --duration 60      # Run for 60 seconds
        """
    )
    parser.add_argument('--phase', type=int, default=2, choices=[0, 1, 2],
                        help='Phase to use (0=blend, 1=gmm, 2=neural)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS (default: 30)')
    parser.add_argument('--duration', type=int, default=120,
                        help='Demo duration in seconds (default: 120)')
    
    args = parser.parse_args()
    
    app = ARMirrorApp(
        target_fps=args.fps,
        demo_duration=args.duration,
        phase=args.phase
    )
    
    if app.initialize():
        app.run()
    else:
        print("\nFailed to initialize")
        sys.exit(1)


if __name__ == "__main__":
    main()
