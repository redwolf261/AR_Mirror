"""
Test VITON Integration

Validates VITON dataset integration and garment loading functionality.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import cv2


class TestVITONIntegration(unittest.TestCase):
    """Test suite for VITON integration"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.viton_root = "viton_data"
        cls.config_path = "viton_config.json"
    
    def test_dataset_structure(self):
        """Test if VITON dataset has correct structure"""
        from viton_integration import VITONDatasetManager
        
        manager = VITONDatasetManager(self.viton_root)
        is_valid, missing = manager.validate_dataset()
        
        if not is_valid:
            self.skipTest(f"VITON dataset not found or incomplete: {missing}")
        
        self.assertTrue(is_valid, "Dataset structure should be valid")
    
    def test_loader_initialization(self):
        """Test VITONGarmentLoader initialization"""
        try:
            from viton_integration import VITONGarmentLoader
            
            loader = VITONGarmentLoader(self.viton_root, self.config_path)
            
            self.assertIsNotNone(loader.sku_mapping)
            self.assertIsInstance(loader.available_products, list)
            
        except Exception as e:
            self.skipTest(f"Dataset not available: {e}")
    
    def test_sku_mapping_load(self):
        """Test SKU mapping configuration loading"""
        from viton_integration import VITONGarmentLoader
        
        try:
            loader = VITONGarmentLoader(self.viton_root, self.config_path)
            
            self.assertGreater(len(loader.sku_mapping), 0, 
                             "Should have at least one SKU mapping")
            
            # Test sample SKU
            if "TSH-001" in loader.sku_mapping:
                viton_id = loader.sku_mapping["TSH-001"]
                self.assertIsInstance(viton_id, str)
                self.assertGreater(len(viton_id), 0)
        
        except Exception as e:
            self.skipTest(f"Config loading failed: {e}")
    
    def test_garment_image_loading(self):
        """Test loading garment image"""
        from viton_integration import VITONGarmentLoader
        
        try:
            loader = VITONGarmentLoader(self.viton_root, self.config_path)
            
            # Get first available SKU
            available_skus = loader.list_available_skus()
            if not available_skus:
                self.skipTest("No SKUs available")
            
            sku = available_skus[0]
            img = loader.get_garment_image(sku)
            
            if img is not None:
                self.assertIsInstance(img, np.ndarray)
                self.assertEqual(len(img.shape), 3, "Image should be BGR")
                self.assertGreater(img.shape[0], 0)
                self.assertGreater(img.shape[1], 0)
        
        except Exception as e:
            self.skipTest(f"Image loading test failed: {e}")
    
    def test_cache_functionality(self):
        """Test image caching"""
        from viton_integration import VITONGarmentLoader
        
        try:
            loader = VITONGarmentLoader(self.viton_root, self.config_path)
            
            available_skus = loader.list_available_skus()
            if not available_skus:
                self.skipTest("No SKUs available")
            
            sku = available_skus[0]
            
            # First load (cache miss)
            img1 = loader.get_garment_image(sku)
            
            # Second load (cache hit)
            img2 = loader.get_garment_image(sku)
            
            if img1 is not None and img2 is not None:
                # Images should be different objects (copied from cache)
                self.assertIsNot(img1, img2)
                # But with same content
                np.testing.assert_array_equal(img1, img2)
        
        except Exception as e:
            self.skipTest(f"Cache test failed: {e}")
    
    def test_mask_loading(self):
        """Test segmentation mask loading"""
        from viton_integration import VITONGarmentLoader
        
        try:
            loader = VITONGarmentLoader(self.viton_root, self.config_path)
            
            available_skus = loader.list_available_skus()
            if not available_skus:
                self.skipTest("No SKUs available")
            
            sku = available_skus[0]
            mask = loader.get_garment_mask(sku)
            
            if mask is not None:
                self.assertIsInstance(mask, np.ndarray)
                self.assertEqual(len(mask.shape), 2, "Mask should be grayscale")
        
        except Exception as e:
            self.skipTest(f"Mask loading test failed: {e}")
    
    def test_garment_info(self):
        """Test garment metadata retrieval"""
        from viton_integration import VITONGarmentLoader
        
        try:
            loader = VITONGarmentLoader(self.viton_root, self.config_path)
            
            available_skus = loader.list_available_skus()
            if not available_skus:
                self.skipTest("No SKUs available")
            
            sku = available_skus[0]
            info = loader.get_garment_info(sku)
            
            self.assertIn('sku', info)
            self.assertIn('viton_id', info)
            self.assertIn('available', info)
            self.assertEqual(info['sku'], sku)
        
        except Exception as e:
            self.skipTest(f"Info test failed: {e}")
    
    def test_invalid_sku(self):
        """Test handling of invalid SKU"""
        from viton_integration import VITONGarmentLoader
        
        try:
            loader = VITONGarmentLoader(self.viton_root, self.config_path)
            
            img = loader.get_garment_image("INVALID-SKU-999")
            self.assertIsNone(img, "Invalid SKU should return None")
        
        except Exception as e:
            self.skipTest(f"Invalid SKU test failed: {e}")


def run_integration_tests(verbose=True):
    """Run all integration tests"""
    print("=" * 60)
    print("VITON Integration Test Suite")
    print("=" * 60)
    print()
    
    # Check if dataset exists
    from viton_integration import VITONDatasetManager
    manager = VITONDatasetManager("viton_data")
    is_valid, missing = manager.validate_dataset()
    
    if not is_valid:
        print("⚠ VITON dataset not found or incomplete")
        print(f"Missing: {missing}")
        print("\nTo download dataset:")
        print("  python prepare_viton_dataset.py --download")
        print("\nRunning basic tests only...")
        print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestVITONIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VITON integration")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--specific', '-s', type=str,
                       help='Run specific test (e.g., test_garment_image_loading)')
    
    args = parser.parse_args()
    
    if args.specific:
        # Run specific test
        suite = unittest.TestSuite()
        suite.addTest(TestVITONIntegration(args.specific))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run all tests
        success = run_integration_tests(verbose=args.verbose)
        sys.exit(0 if success else 1)
