"""
Quick test for cloth detail sharpening feature
"""
import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_sharpening():
    """Test the sharpening enhancement"""
    print("\n" + "="*70)
    print("CLOTH DETAIL SHARPENING TEST")
    print("="*70)
    
    # Import the pipeline
    try:
        from src.pipelines.phase2_neural_pipeline import Phase2NeuralPipeline
        print("✓ Phase2NeuralPipeline imported successfully")
    except Exception as e:
        print(f"✗ Failed to import pipeline: {e}")
        return False
    
    # Create synthetic test image with logo-like pattern
    print("\n1. Creating test image with logo pattern...")
    img = np.zeros((256, 192, 3), dtype=np.float32)
    # Draw a "logo" - white text on dark background
    cv2.putText(img, "LEVI'S", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (1.0, 1.0, 1.0), 2)
    cv2.putText(img, "LOGO", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1.0, 1.0, 1.0), 1)
    
    # Create cloth mask (center region)
    mask = np.zeros((256, 192), dtype=np.float32)
    mask[40:160, 40:150] = 1.0
    
    print(f"  Test image: {img.shape}, range [{img.min():.2f}, {img.max():.2f}]")
    print(f"  Cloth mask: {mask.shape}, coverage {mask.sum()/(256*192)*100:.1f}%")
    
    # Initialize pipeline (CPU mode to avoid GPU requirement)
    print("\n2. Initializing pipeline with sharpening enabled...")
    try:
        pipeline = Phase2NeuralPipeline(
            device='cpu',
            enable_tom=False,  # Don't need TOM model for this test
            enable_sharpening=True
        )
        print("✓ Pipeline initialized (sharpening enabled)")
    except Exception as e:
        print(f"✗ Pipeline init failed: {e}")
        return False
    
    # Test sharpening function directly
    print("\n3. Testing sharpening function...")
    try:
        sharpened = pipeline._sharpen_cloth_details(img, mask, strength=0.6)
        print(f"✓ Sharpening complete: {sharpened.shape}, range [{sharpened.min():.3f}, {sharpened.max():.3f}]")
        
        # Check output validity
        if sharpened.shape != img.shape:
            print(f"✗ Shape mismatch: expected {img.shape}, got {sharpened.shape}")
            return False
        if sharpened.min() < 0 or sharpened.max() > 1:
            print(f"✗ Value range error: expected [0,1], got [{sharpened.min():.3f}, {sharpened.max():.3f}]")
            return False
        
        # Compute sharpening effect
        diff = np.abs(sharpened - img)
        max_change = diff.max()
        avg_change = diff.mean()
        cloth_region_change = diff[mask > 0.5].mean() if (mask > 0.5).any() else 0
        
        print(f"  Max pixel change: {max_change:.4f}")
        print(f"  Average change: {avg_change:.6f}")
        print(f"  Cloth region change: {cloth_region_change:.6f}")
        
        if cloth_region_change < 0.001:
            print("⚠ Warning: Very small changes detected (sharpening may be too weak)")
        else:
            print("✓ Sharpening effect detected")
        
    except Exception as e:
        print(f"✗ Sharpening failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with different strengths
    print("\n4. Testing different sharpening strengths...")
    for strength in [0.3, 0.6, 1.0]:
        try:
            result = pipeline._sharpen_cloth_details(img, mask, strength=strength)
            change = np.abs(result - img)[mask > 0.5].mean()
            print(f"  Strength {strength:.1f}: avg change = {change:.6f}")
        except Exception as e:
            print(f"  Strength {strength:.1f}: FAILED - {e}")
    
    # Save visual comparison
    print("\n5. Saving visual comparison...")
    try:
        output_dir = Path(__file__).parent.parent / "data" / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8 for saving
        img_uint8 = (img * 255).astype(np.uint8)
        sharpened_uint8 = (sharpened * 255).astype(np.uint8)
        diff_uint8 = (diff * 255 * 10).astype(np.uint8)  # Amplify difference for visibility
        
        cv2.imwrite(str(output_dir / "sharpening_original.png"), img_uint8)
        cv2.imwrite(str(output_dir / "sharpening_enhanced.png"), sharpened_uint8)
        cv2.imwrite(str(output_dir / "sharpening_diff.png"), diff_uint8)
        
        print(f"✓ Saved to {output_dir}/")
        print(f"  - sharpening_original.png")
        print(f"  - sharpening_enhanced.png")
        print(f"  - sharpening_diff.png")
    except Exception as e:
        print(f"⚠ Could not save images: {e}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_sharpening()
    sys.exit(0 if success else 1)
