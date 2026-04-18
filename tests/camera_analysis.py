"""
Camera Image Analysis for Hybrid AR Try-On System
Analyzes visual quality, segmentation accuracy, and garment placement
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import cv2
import time
from datetime import datetime

from src.hybrid.hybrid_pipeline import HybridTryOnPipeline
from src.hybrid.body_understanding.segmentation import MediaPipeSegmenter


class CameraImageAnalyzer:
    """
    Comprehensive image analysis for quality validation
    Analyzes: Segmentation, artifacts, color accuracy, warping quality
    """
    
    def __init__(self):
        self.pipeline = None
        self.segmenter = None
        self.cap = None
        self.results = {
            'segmentation_quality': [],
            'artifact_scores': [],
            'color_accuracy': [],
            'boundary_smoothness': [],
            'temporal_stability': [],
            'garment_placement': []
        }
    
    def setup(self):
        """Initialize pipeline and camera"""
        print("[SETUP] Setting up camera analysis...\n")
        
        try:
            self.pipeline = HybridTryOnPipeline(use_gpu=False, enable_temporal_stabilization=True)
            print("[OK] Pipeline initialized")
        except Exception as e:
            print(f"[ERROR] Pipeline error: {e}")
            return False
        
        try:
            self.segmenter = MediaPipeSegmenter()
            print("[OK] Segmenter initialized")
        except Exception as e:
            print(f"[ERROR] Segmenter error: {e}")
            return False
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[ERROR] Camera not available")
                return False
            print("[OK] Camera connected")
        except Exception as e:
            print(f"[ERROR] Camera error: {e}")
            return False
        
        return True
    
    def analyze_segmentation_quality(self, rgb_frame, mask):
        """Analyze segmentation mask quality"""
        # Edge smoothness
        edges = cv2.Canny(mask.astype(np.uint8) * 255, 30, 100)
        edge_count = np.count_nonzero(edges)
        
        # Calculate perimeter smoothness (lower is smoother)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoothness_score = 100.0
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(main_contour, True)
            area = cv2.contourArea(main_contour)
            
            if area > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                smoothness_score = min(100, circularity * 100)
        
        # Mask connectivity
        num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
        connectivity_score = min(100, 100 / num_labels) if num_labels > 0 else 50
        
        quality_score = (smoothness_score * 0.5 + connectivity_score * 0.5)
        
        return quality_score
    
    def analyze_color_accuracy(self, original, composite):
        """Analyze color preservation in composite"""
        # Focus on face and torso regions
        h, w = original.shape[:2]
        roi_y_start = int(h * 0.1)
        roi_y_end = int(h * 0.6)
        roi_x_start = int(w * 0.2)
        roi_x_end = int(w * 0.8)
        
        original_roi = original[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        composite_roi = composite[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # Calculate color similarity using histogram correlation
        scores = []
        for channel in range(3):
            hist_orig = cv2.calcHist([original_roi], [channel], None, [256], [0, 256])
            hist_comp = cv2.calcHist([composite_roi], [channel], None, [256], [0, 256])
            
            # Normalize
            hist_orig = cv2.normalize(hist_orig, hist_orig).flatten()
            hist_comp = cv2.normalize(hist_comp, hist_comp).flatten()
            
            # Correlation
            correlation = cv2.compareHist(hist_orig, hist_comp, cv2.HISTCMP_CORREL)
            scores.append(correlation * 100)
        
        return np.mean(scores)
    
    def analyze_boundary_smoothness(self, mask, composite_rgb):
        """Analyze garment-body boundary smoothness"""
        # Find edges
        edges = cv2.Canny(mask.astype(np.uint8) * 255, 20, 60)
        
        # Erode and dilate to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Count edge pixels (fewer = smoother)
        edge_pixels = np.count_nonzero(cleaned)
        
        # Normalize by image size
        total_pixels = mask.shape[0] * mask.shape[1]
        edge_ratio = (edge_pixels / total_pixels) * 100
        
        # Smoothness inversely proportional to edge ratio
        smoothness = max(0, 100 - edge_ratio * 10)
        
        return smoothness
    
    def analyze_temporal_stability(self, prev_composite, curr_composite):
        """Analyze frame-to-frame temporal stability"""
        if prev_composite is None:
            return 100.0
        
        # Resize for faster computation
        h, w = curr_composite.shape[:2]
        resize_shape = (w // 2, h // 2)
        
        prev_resized = cv2.resize(prev_composite, resize_shape)
        curr_resized = cv2.resize(curr_composite, resize_shape)
        
        # Calculate optical flow to detect jitter
        prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Magnitude of motion
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = np.mean(magnitude)
        
        # Lower motion = higher stability
        stability = max(0, 100 - avg_motion * 5)
        
        return stability
    
    def analyze_garment_placement(self, mask, composite_rgb):
        """Analyze garment positioning accuracy"""
        h, w = composite_rgb.shape[:2]
        
        # Expected clothing region (roughly chest to waist)
        garment_top = int(h * 0.15)
        garment_bottom = int(h * 0.65)
        garment_left = int(w * 0.2)
        garment_right = int(w * 0.8)
        
        # Extract garment region from mask
        garment_region = mask[garment_top:garment_bottom, garment_left:garment_right]
        coverage = np.sum(garment_region) / garment_region.size * 100
        
        # Check for extreme values
        if coverage < 10:
            score = 50.0  # Insufficient coverage
        elif coverage > 90:
            score = 60.0  # Excessive coverage
        else:
            score = 100.0 - abs(coverage - 50) * 0.5  # Optimal around 50%
        
        return score
    
    def run_analysis(self, duration_seconds=20, garment_sku='TSH-001'):
        """Run continuous camera analysis"""
        if not self.setup():
            return False
        
        print(f"\n[CAMERA ANALYSIS - {duration_seconds}s Analysis]")
        print(f"[GARMENT] {garment_sku}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        frame_count = 0
        prev_composite = None
        
        try:
            while time.time() - start_time < duration_seconds:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Process with pipeline
                    t0 = time.time()
                    result = self.pipeline.process_frame(rgb_frame, garment_sku)
                    process_time = time.time() - t0
                    
                    composite = result['composite']
                    mask = result['body_mask']
                    
                    # Analyze quality metrics
                    seg_quality = self.analyze_segmentation_quality(rgb_frame, mask)
                    color_acc = self.analyze_color_accuracy(rgb_frame, composite)
                    boundary = self.analyze_boundary_smoothness(mask, composite)
                    temporal = self.analyze_temporal_stability(prev_composite, composite)
                    garment_placement = self.analyze_garment_placement(mask, composite)
                    
                    # Store results
                    self.results['segmentation_quality'].append(seg_quality)
                    self.results['color_accuracy'].append(color_acc)
                    self.results['boundary_smoothness'].append(boundary)
                    self.results['temporal_stability'].append(temporal)
                    self.results['garment_placement'].append(garment_placement)
                    
                    frame_count += 1
                    prev_composite = composite.copy()
                    
                    if frame_count % 5 == 0:
                        print(f"[FRAME {frame_count:3d}] Seg={seg_quality:.1f}%, Color={color_acc:.1f}%, "
                              f"Boundary={boundary:.1f}%, Temporal={temporal:.1f}%, FPS={1/process_time:.1f}")
                
                except Exception as e:
                    print(f"[ERROR] Analysis error: {e}")
                    continue
        
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Analysis interrupted by user")
        
        finally:
            if self.cap:
                self.cap.release()
        
        self.print_analysis_report()
        return True
    
    def print_analysis_report(self):
        """Print comprehensive analysis report"""
        print(f"\n{'='*60}")
        print(f"CAMERA IMAGE ANALYSIS REPORT")
        print(f"{'='*60}\n")
        
        metrics = {
            'Segmentation Quality': self.results['segmentation_quality'],
            'Color Accuracy': self.results['color_accuracy'],
            'Boundary Smoothness': self.results['boundary_smoothness'],
            'Temporal Stability': self.results['temporal_stability'],
            'Garment Placement': self.results['garment_placement']
        }
        
        print("[QUALITY METRICS]\n")
        
        for metric_name, values in metrics.items():
            if values:
                values_array = np.array(values)
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                min_val = np.min(values_array)
                max_val = np.max(values_array)
                
                status = "[OK]" if mean_val >= 80 else "[WARN]" if mean_val >= 70 else "[FAIL]"
                print(f"{status} {metric_name:<25} {mean_val:5.1f}% (±{std_val:4.1f}%, range: {min_val:5.1f}-{max_val:5.1f}%)")
        
        # Overall assessment
        print(f"\n{'='*60}")
        all_values = []
        for values in metrics.values():
            all_values.extend(values)
        
        overall_score = np.mean(all_values) if all_values else 0
        
        if overall_score >= 85:
            print(f"[EXCELLENT] EXCELLENT QUALITY - {overall_score:.1f}%")
        elif overall_score >= 75:
            print(f"[GOOD] GOOD QUALITY - {overall_score:.1f}% (Minor improvements recommended)")
        elif overall_score >= 65:
            print(f"[ACCEPTABLE] ACCEPTABLE QUALITY - {overall_score:.1f}% (Tuning needed)")
        else:
            print(f"[POOR] POOR QUALITY - {overall_score:.1f}% (Review configuration)")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    analyzer = CameraImageAnalyzer()
    
    # Analyze with multiple garments
    garments_to_test = ['TSH-001', 'TSH-002', 'SHT-001']
    
    for sku in garments_to_test:
        analyzer = CameraImageAnalyzer()
        analyzer.run_analysis(duration_seconds=15, garment_sku=sku)
        print("\n" + "="*60 + "\n")
