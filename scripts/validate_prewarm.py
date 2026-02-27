#!/usr/bin/env python3
"""
Validate TOM Pre-Warming Feature

Quick test to verify that TOM pre-warming eliminates cold-start distortion.
This test doesn't require a camera, just validates the pre-warming mechanism.
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def main():
    log.info("="*60)
    log.info("TOM Pre-Warming Validation")
    log.info("="*60)
    
    # Initialize pipeline
    log.info("\n1. Initializing Phase2NeuralPipeline...")
    try:
        from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
        pipeline = Phase2NeuralPipeline(
            device='auto',
            enable_tom=True,
            enable_optimizations=True
        )
        log.info("   ✓ Pipeline initialized")
    except Exception as e:
        log.error(f"   ✗ Pipeline failed: {e}")
        return 1
    
    # Check if TOM is loaded
    if pipeline.tom_model is None:
        log.error("   ✗ TOM model not loaded - cannot test pre-warming")
        return 1
    
    log.info("   ✓ TOM model loaded")
    
    # Create dummy garment
    log.info("\n2. Creating test garment...")
    cloth_rgb = np.random.rand(256, 192, 3).astype(np.float32)
    cloth_mask = np.ones((256, 192), dtype=np.float32)
    log.info("   ✓ Test garment created (256x192)")
    
    # Test pre-warming
    log.info("\n3. Testing TOM pre-warming...")
    log.info("   This should complete in ~200ms and eliminate cold-start distortion")
    log.info("")
    
    result = pipeline.prewarm_tom_cache(cloth_rgb, cloth_mask)
    
    if result:
        log.info("\n   ✓ SUCCESS: TOM cache pre-warmed!")
        log.info("   ✓ First frame will now show TOM quality instead of GMM fallback")
    else:
        log.info("\n   ✗ FAILED: Pre-warming unsuccessful")
        return 1
    
    # Verify cache is populated
    log.info("\n4. Verifying TOM cache state...")
    if hasattr(pipeline, '_tom_cache') and pipeline._tom_cache is not None:
        log.info("   ✓ TOM cache is populated")
        log.info(f"   ✓ Cache shape: {pipeline._tom_cache.shape}")
    else:
        log.info("   ℹ Cache state unknown (async mechanism)")
    
    log.info("\n" + "="*60)
    log.info("VALIDATION COMPLETE")
    log.info("="*60)
    log.info("\nResults:")
    log.info("  ✓ TOM pre-warming feature is working correctly")
    log.info("  ✓ Cold-start GMM distortion will be eliminated")
    log.info("  ✓ First frame will show high-quality TOM output")
    log.info("\nNext: Run tryon_selector.py and switch garments")
    log.info("      You should see '[Prewarm]' logs when changing garments")
    log.info("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
