#!/usr/bin/env python3
"""
Test Garment Quality - Compare distortion across dataset files

Tests the hypothesis that early-numbered files (00000_00.jpg) have different
format/preprocessing than main training range (05310_00.jpg onwards).

Usage:
    python scripts/test_garment_quality.py

Runs 10-second test with each of 3 garments:
- 00000_00.jpg (early file - suspected bad format)
- 05310_00.jpg (first file in main training range)
- 10000_00.jpg (mid-range file)

Saves screenshots to data/test_results/ for manual comparison.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import time
import logging

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
from src.core.body_aware_fitter import BodyAwareGarmentFitter

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Test garments
TEST_GARMENTS = [
    ("00000_00.jpg", "Early file (suspected bad format)"),
    ("05310_00.jpg", "First file in main training range"),
    ("10000_00.jpg", "Mid-range file"),
]

DATASET_ROOT = ROOT / "dataset" / "train" / "cloth"
OUTPUT_DIR = ROOT / "data" / "test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_garment(filename: str):
    """Load garment RGB and mask from dataset"""
    cloth_path = DATASET_ROOT / filename
    if not cloth_path.exists():
        log.error(f"Garment not found: {cloth_path}")
        return None, None
    
    # Load cloth image
    cloth_bgr = cv2.imread(str(cloth_path))
    if cloth_bgr is None:
        return None, None
    
    cloth_rgb = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Generate simple mask (assume white background)
    gray = cv2.cvtColor(cloth_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    mask = mask.astype(np.float32) / 255.0
    
    return cloth_rgb, mask


def test_garment(pipeline, fitter, cap, filename: str, description: str):
    """Test a single garment file and save screenshots"""
    log.info(f"\n{'='*60}")
    log.info(f"Testing: {filename}")
    log.info(f"Description: {description}")
    log.info(f"{'='*60}")
    
    # Load garment
    cloth_rgb, cloth_mask = load_garment(filename)
    if cloth_rgb is None:
        log.error(f"Failed to load {filename}")
        return
    
    log.info(f"Garment loaded: {cloth_rgb.shape}, mask: {cloth_mask.shape}")
    
    # Capture frames for 5 seconds
    frame_count = 0
    saved_count = 0
    start_time = time.time()
    
    # Collections for analysis
    tom_active_count = 0
    gmm_fallback_count = 0
    
    while time.time() - start_time < 5.0:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize to standard resolution
        frame = cv2.resize(frame, (960, 720))
        
        # Get body measurements
        try:
            measurements = fitter.extract_body_measurements(frame)
            if measurements is None:
                continue
            
            # Warp garment
            result = pipeline.warp_garment(
                person_image=frame,
                cloth_rgb=cloth_rgb,
                cloth_mask=cloth_mask,
                mp_landmarks=measurements.get('landmarks'),
                body_mask=measurements.get('body_mask'),
                use_densepose=False,
                use_smpl=False,
                garment_type='tshirt'
            )
            
            # Check which rendering path is used
            if result.synthesized is not None:
                tom_active_count += 1
                path_status = "TOM"
            else:
                gmm_fallback_count += 1
                path_status = "GMM"
            
            # Save screenshots at specific frames
            if frame_count in [1, 5, 10, 30, 50]:
                output_path = OUTPUT_DIR / f"{filename.replace('.jpg', '')}_{path_status}_frame{frame_count:03d}.jpg"
                
                # Composite result onto frame (simplified rendering)
                if result.synthesized is not None:
                    # Use TOM output
                    synth_bgr = cv2.cvtColor(result.synthesized, cv2.COLOR_RGB2BGR)
                    synth_u8 = np.clip(synth_bgr * 255, 0, 255).astype(np.uint8)
                    display = cv2.resize(synth_u8, (960, 720))
                else:
                    # Use GMM warped cloth (fallback)
                    warped_bgr = cv2.cvtColor(result.warped_cloth, cv2.COLOR_RGB2BGR)
                    warped_u8 = np.clip(warped_bgr * 255, 0, 255).astype(np.uint8)
                    display = cv2.resize(warped_u8, (960, 720))
                
                # Add labels
                cv2.putText(display, f"{filename} - Frame {frame_count} - {path_status}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, description, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                cv2.imwrite(str(output_path), display)
                saved_count += 1
                log.info(f"  Saved frame {frame_count}: {path_status} mode → {output_path.name}")
        
        except Exception as e:
            log.debug(f"Frame {frame_count} error: {e}")
            continue
    
    # Summary
    log.info(f"\nSummary for {filename}:")
    log.info(f"  Total frames: {frame_count}")
    log.info(f"  TOM active: {tom_active_count} frames ({100*tom_active_count/max(frame_count,1):.1f}%)")
    log.info(f"  GMM fallback: {gmm_fallback_count} frames ({100*gmm_fallback_count/max(frame_count,1):.1f}%)")
    log.info(f"  Screenshots saved: {saved_count}")


def main():
    log.info("="*60)
    log.info("AR Mirror - Garment Quality Test")
    log.info("="*60)
    log.info(f"Dataset: {DATASET_ROOT}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("")
    
    # Check dataset exists
    if not DATASET_ROOT.exists():
        log.error(f"Dataset not found: {DATASET_ROOT}")
        log.error("Please ensure dataset/train/cloth/ exists")
        return 1
    
    # Initialize pipeline
    log.info("Initializing Phase2NeuralPipeline...")
    try:
        pipeline = Phase2NeuralPipeline(
            device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
            enable_tom=True,
            enable_optimizations=True
        )
        log.info("✓ Pipeline initialized")
    except Exception as e:
        log.error(f"Pipeline initialization failed: {e}")
        return 1
    
    # Initialize fitter
    log.info("Initializing BodyAwareGarmentFitter...")
    try:
        fitter = BodyAwareGarmentFitter()
        log.info("✓ Fitter initialized")
    except Exception as e:
        log.error(f"Fitter initialization failed: {e}")
        return 1
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Failed to open camera")
        return 1
    
    log.info("✓ Camera opened")
    log.info("")
    
    try:
        # Test each garment
        for filename, description in TEST_GARMENTS:
            test_garment(pipeline, fitter, cap, filename, description)
        
        log.info("\n" + "="*60)
        log.info("Test complete!")
        log.info(f"Screenshots saved to: {OUTPUT_DIR}")
        log.info("="*60)
        log.info("\nNext steps:")
        log.info("1. Review screenshots in data/test_results/")
        log.info("2. Compare distortion quality:")
        log.info("   - 00000_00.jpg vs 05310_00.jpg vs 10000_00.jpg")
        log.info("3. Check if distortion is file-specific or systemic")
        log.info("4. Note TOM vs GMM frame counts for each garment")
    
    finally:
        cap.release()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
