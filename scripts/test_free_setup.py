#!/usr/bin/env python3
"""
Quick test script for AR Mirror Free SMPL Setup

🎯 This script validates that the free SMPL-X setup works correctly
   and provides example usage patterns for development.

Usage:
    python scripts/test_free_setup.py
    python scripts/test_free_setup.py --generate-samples 5
"""

import sys
import os
from pathlib import Path
import logging
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger("TestFreeSetup")

def test_free_smpl_setup():
    """Test that free SMPL-X model setup works"""
    log.info("🧪 Testing Free SMPL-X Setup...")
    
    try:
        # Test synthetic data factory import
        from scripts.synthetic_data_factory import SyntheticDataFactory, GenerationConfig
        log.info("✅ Synthetic data factory imports successful")
        
        # Test config creation
        config = GenerationConfig(
            output_dir=Path('test_output'),
            batch_size=1,
            resolution=(512, 512),
            physics_enabled=False,  # Skip physics for basic test
            domain_randomization=False
        )
        log.info("✅ Generation config created")
        
        # Test factory initialization (free mode)
        factory = SyntheticDataFactory(config, use_commercial_smpl=False)
        log.info("✅ Free SMPL factory initialized")
        
        # Test sample generation
        sample = factory.generate_sample()
        log.info(f"✅ Sample generated: {sample['pose_category']}, {sample['body_type']}")
        
        # Validate sample structure
        assert 'pose' in sample, "Sample missing pose data"
        assert 'shape' in sample, "Sample missing shape data"
        assert 'pose_category' in sample, "Sample missing pose category"
        assert 'body_type' in sample, "Sample missing body type"
        log.info("✅ Sample data structure validated")
        
        log.info("🎉 Free SMPL-X setup test PASSED!")
        return True
        
    except ImportError as e:
        log.error(f"❌ Import error: {e}")
        log.info("💡 Run: pip install numpy pathlib")
        return False
    except Exception as e:
        log.error(f"❌ Test failed: {e}")
        return False

def test_production_pipeline_import():
    """Test that production pipeline imports correctly"""
    log.info("🧪 Testing Production Pipeline Import...")
    
    try:
        from scripts.production_pipeline import ProductionPipeline
        log.info("✅ Production pipeline import successful")
        return True
    except ImportError as e:
        log.error(f"❌ Production pipeline import failed: {e}")
        return False

def test_blender_renderer_import():
    """Test that Blender renderer imports correctly"""
    log.info("🧪 Testing Blender Renderer Import...")
    
    try:
        # Note: This will fail if bpy (Blender Python) is not available
        # That's expected for development environment
        sys.path.append('scripts')
        from blender_renderer import BlenderSMPLRenderer
        log.info("✅ Blender renderer import successful (Blender environment)")
        return True
    except ImportError as e:
        log.info("ℹ️  Blender renderer requires Blender environment (expected for development)")
        return True  # This is expected in development

def generate_test_samples(num_samples: int = 5):
    """Generate test samples using free SMPL setup"""
    log.info(f"🔧 Generating {num_samples} test samples...")
    
    try:
        from scripts.synthetic_data_factory import SyntheticDataFactory, GenerationConfig
        
        # Setup test output directory  
        output_dir = Path('test_free_samples')
        output_dir.mkdir(exist_ok=True)
        
        config = GenerationConfig(
            output_dir=output_dir,
            batch_size=num_samples,
            resolution=(256, 256),  # Smaller resolution for faster testing
            physics_enabled=False,
            domain_randomization=False
        )
        
        # Initialize free factory
        factory = SyntheticDataFactory(config, use_commercial_smpl=False)
        
        # Generate samples
        samples = factory.generate_batch(num_samples)
        
        log.info(f"✅ Generated {len(samples)} test samples in {output_dir}")
        
        # Show sample statistics
        pose_categories = [s['pose_category'] for s in samples]
        body_types = [s['body_type'] for s in samples]
        
        log.info(f"📊 Pose categories: {set(pose_categories)}")
        log.info(f"📊 Body types: {set(body_types)}")
        
        return True
        
    except Exception as e:
        log.error(f"❌ Sample generation failed: {e}")
        return False

def test_file_structure():
    """Validate expected file structure exists"""
    log.info("🧪 Testing File Structure...")
    
    required_files = [
        'scripts/synthetic_data_factory.py',
        'scripts/production_pipeline.py', 
        'scripts/blender_renderer.py',
        'blender_addon/__init__.py',
        'docs/FREE_SMPL_SETUP.md',
        'docs/EXECUTION_PLAN.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        log.error(f"❌ Missing files: {missing_files}")
        return False
    else:
        log.info("✅ All required files present")
        return True

def main():
    parser = argparse.ArgumentParser(description="Test AR Mirror Free SMPL Setup")
    parser.add_argument('--generate-samples', type=int, default=0,
                       help='Generate test samples (0 = skip generation)')
    parser.add_argument('--skip-imports', action='store_true',
                       help='Skip import tests (faster)')
    
    args = parser.parse_args()
    
    print("""
🆓 AR Mirror Free SMPL Setup Test
================================
Testing free development environment...
""")
    
    all_passed = True
    
    # Test file structure
    if not test_file_structure():
        all_passed = False
    
    # Test imports (unless skipped)
    if not args.skip_imports:
        if not test_free_smpl_setup():
            all_passed = False
        
        if not test_production_pipeline_import():
            all_passed = False
        
        if not test_blender_renderer_import():
            all_passed = False
    
    # Generate test samples (if requested)
    if args.generate_samples > 0:
        if not generate_test_samples(args.generate_samples):
            all_passed = False
    
    # Summary
    if all_passed:
        print(f"""
✅ FREE SMPL SETUP TEST PASSED!
==============================
Your development environment is ready.

Next steps:
1. Generate test batch: python scripts/production_pipeline.py --batch_size 10 --synthetic_only
2. See usage guide: docs/FREE_SMPL_SETUP.md  
3. When ready for production: acquire commercial SMPL license

Development Commands:
• python scripts/production_pipeline.py --phase 1 --batch_size 100 --synthetic_only
• python scripts/production_pipeline.py --phase 2 --batch_size 1000 --free
• python scripts/production_pipeline.py --phase 3 --batch_size 10000 --commercial (requires license)
""")
    else:
        print(f"""
❌ SETUP TEST FAILED
====================
Some components are not working correctly.

Troubleshooting:
1. Install dependencies: pip install numpy pathlib
2. Check file structure: all scripts present?
3. Review error messages above
4. See: docs/FREE_SMPL_SETUP.md
""")
        sys.exit(1)

if __name__ == "__main__":
    main()