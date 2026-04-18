import unittest
import numpy as np
from src.sizing_pipeline import (
    BodyMeasurements,
    BodyRegion,
    GarmentSpec,
    FitDecision,
    MeasurementEstimator,
    SizeMatcher
)


class TestMeasurementEstimator(unittest.TestCase):
    
    def setUp(self):
        self.estimator = MeasurementEstimator(640, 480)
        
    def test_valid_measurements(self):
        landmarks = {
            0: {'x': 0.5, 'y': 0.2, 'z': 0, 'visibility': 0.9},
            11: {'x': 0.4, 'y': 0.35, 'z': 0, 'visibility': 0.9},
            12: {'x': 0.6, 'y': 0.35, 'z': 0, 'visibility': 0.9},
            23: {'x': 0.42, 'y': 0.65, 'z': 0, 'visibility': 0.9},
            24: {'x': 0.58, 'y': 0.65, 'z': 0, 'visibility': 0.9}
        }
        
        measurements = self.estimator.estimate(landmarks)
        self.assertIsNotNone(measurements)
        assert measurements is not None  # Type narrowing
        self.assertGreater(measurements.shoulder_width_cm, 35)
        self.assertLess(measurements.shoulder_width_cm, 55)
        
    def test_tilted_shoulders_rejected(self):
        landmarks = {
            0: {'x': 0.5, 'y': 0.2, 'z': 0, 'visibility': 0.9},
            11: {'x': 0.4, 'y': 0.3, 'z': 0, 'visibility': 0.9},
            12: {'x': 0.6, 'y': 0.4, 'z': 0, 'visibility': 0.9},
            23: {'x': 0.42, 'y': 0.65, 'z': 0, 'visibility': 0.9},
            24: {'x': 0.58, 'y': 0.65, 'z': 0, 'visibility': 0.9}
        }
        
        measurements = self.estimator.estimate(landmarks)
        self.assertIsNone(measurements)
    
    def test_invalid_distance_rejected(self):
        landmarks = {
            0: {'x': 0.5, 'y': 0.1, 'z': 0, 'visibility': 0.9},
            11: {'x': 0.4, 'y': 0.12, 'z': 0, 'visibility': 0.9},
            12: {'x': 0.6, 'y': 0.12, 'z': 0, 'visibility': 0.9},
            23: {'x': 0.42, 'y': 0.15, 'z': 0, 'visibility': 0.9},
            24: {'x': 0.58, 'y': 0.15, 'z': 0, 'visibility': 0.9}
        }
        
        measurements = self.estimator.estimate(landmarks)
        self.assertIsNone(measurements)


class TestSizeMatcher(unittest.TestCase):
    
    def setUp(self):
        garments = [
            {
                "sku": "TEST-M",
                "size_label": "M",
                "shoulder_cm": 44.0,
                "chest_cm": 50.0,
                "length_cm": 65.0
            }
        ]
        
        import tempfile
        import json
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(garments, self.temp_file)
        self.temp_file.close()
        
        self.matcher = SizeMatcher(self.temp_file.name)
        
    def test_tight_fit_detection(self):
        measurements = BodyMeasurements(
            shoulder_width_cm=43.0,
            chest_width_cm=49.0,
            torso_length_cm=63.0,
            confidence=0.9,
            timestamp=0,
            detected_regions={BodyRegion.UPPER_BODY}
        )
        
        result = self.matcher.match(measurements, "TEST-M")
        self.assertIsNotNone(result)
        assert result is not None  # Type narrowing
        self.assertEqual(result.decision, FitDecision.TIGHT)
    
    def test_good_fit_detection(self):
        measurements = BodyMeasurements(
            shoulder_width_cm=40.0,
            chest_width_cm=44.0,
            torso_length_cm=60.0,
            confidence=0.9,
            timestamp=0,
            detected_regions={BodyRegion.UPPER_BODY}
        )
        
        result = self.matcher.match(measurements, "TEST-M")
        self.assertIsNotNone(result)
        assert result is not None  # Type narrowing
        self.assertEqual(result.decision, FitDecision.GOOD)
    
    def test_loose_fit_detection(self):
        measurements = BodyMeasurements(
            shoulder_width_cm=36.0,
            chest_width_cm=40.0,
            torso_length_cm=56.0,
            confidence=0.9,
            timestamp=0,
            detected_regions={BodyRegion.UPPER_BODY}
        )
        
        result = self.matcher.match(measurements, "TEST-M")
        self.assertIsNotNone(result)
        assert result is not None  # Type narrowing
        self.assertEqual(result.decision, FitDecision.LOOSE)
    
    def tearDown(self):
        import os
        os.unlink(self.temp_file.name)


class TestScaleFactor(unittest.TestCase):
    
    def test_scale_factor_computation(self):
        estimator = MeasurementEstimator(640, 480)
        
        landmarks = {
            0: {'x': 0.5, 'y': 0.2, 'z': 0, 'visibility': 0.9},
            11: {'x': 0.4, 'y': 0.35, 'z': 0, 'visibility': 0.9},
            12: {'x': 0.6, 'y': 0.35, 'z': 0, 'visibility': 0.9},
            23: {'x': 0.42, 'y': 0.65, 'z': 0, 'visibility': 0.9},
            24: {'x': 0.58, 'y': 0.65, 'z': 0, 'visibility': 0.9}
        }
        
        scale = estimator._compute_scale_factor(landmarks)
        self.assertIsNotNone(scale)
        assert scale is not None  # Type narrowing
        self.assertGreater(scale, 0)
        self.assertLess(scale, 1.0)


if __name__ == '__main__':
    unittest.main()
