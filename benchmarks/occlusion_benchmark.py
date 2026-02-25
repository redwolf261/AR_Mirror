#!/usr/bin/env python3
"""
Occlusion Error Benchmarking
Measures what actually matters for production quality, not just FPS

Metrics:
1. Occlusion error rate: % of frames where garment overlaps face/hair
2. Temporal stability: Mask jitter between consecutive frames
3. Failure taxonomy: Performance on edge cases
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.semantic_parser import SemanticParser, create_occlusion_aware_composite


class OcclusionBenchmark:
    """Benchmark occlusion correctness, not just speed"""
    
    def __init__(self, parser: SemanticParser):
        self.parser = parser
        self.results = {
            'occlusion_errors': [],
            'temporal_jitter': [],
            'frame_times': [],
            'failure_cases': {}
        }
    
    def measure_occlusion_error(
        self,
        garment_mask: np.ndarray,
        face_mask: np.ndarray,
        hair_mask: np.ndarray
    ) -> float:
        """
        Calculate occlusion error rate
        
        Returns:
            Error rate (0.0 = perfect, 1.0 = complete overlap)
        """
        # Check if garment overlaps face
        face_overlap = cv2.bitwise_and(garment_mask, face_mask)
        face_error = face_overlap.sum() / max(face_mask.sum(), 1)
        
        # Check if garment overlaps hair
        hair_overlap = cv2.bitwise_and(garment_mask, hair_mask)
        hair_error = hair_overlap.sum() / max(hair_mask.sum(), 1)
        
        # Combined error rate
        total_error = (face_error + hair_error) / 2.0
        
        return total_error
    
    def measure_temporal_stability(
        self,
        current_mask: np.ndarray,
        previous_mask: Optional[np.ndarray]
    ) -> float:
        """
        Measure mask jitter between consecutive frames
        
        Returns:
            Jitter score (0.0 = stable, 1.0 = completely different)
        """
        if previous_mask is None:
            return 0.0
        
        # Calculate pixel-wise difference
        diff = cv2.absdiff(current_mask, previous_mask)
        jitter = diff.sum() / (current_mask.shape[0] * current_mask.shape[1] * 255.0)
        
        return jitter
    
    def benchmark_frame(
        self,
        frame: np.ndarray,
        pose_landmarks: Optional[Any] = None,
        prev_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Benchmark a single frame
        
        Returns:
            Dict with metrics for this frame
        """
        h, w = frame.shape[:2]
        
        # Time the parsing
        start_time = time.time()
        masks = self.parser.parse(
            frame,
            pose_landmarks=pose_landmarks,
            target_resolution=(473, 473)
        )
        parse_time = time.time() - start_time
        
        # Calculate garment region (upper_body + arms)
        garment_region = cv2.bitwise_or(
            masks['upper_body'],
            masks['arms']
        )
        
        # Measure occlusion error
        occlusion_error = self.measure_occlusion_error(
            garment_region,
            masks['face'],
            masks['hair']
        )
        
        # Measure temporal stability
        temporal_jitter = self.measure_temporal_stability(
            garment_region,
            prev_mask
        )
        
        return {
            'occlusion_error': occlusion_error,
            'temporal_jitter': temporal_jitter,
            'parse_time': parse_time,
            'garment_mask': garment_region,
            'masks': masks
        }
    
    def benchmark_video(
        self,
        video_path: str,
        max_frames: int = 100,
        use_geometric_constraints: bool = True
    ) -> Dict:
        """
        Run benchmark on video file
        
        Args:
            video_path: Path to test video
            max_frames: Maximum frames to process
            use_geometric_constraints: Whether to use pose landmarks
        
        Returns:
            Summary statistics
        """
        print(f"\nBenchmarking: {video_path}")
        print(f"Geometric constraints: {'ON' if use_geometric_constraints else 'OFF'}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        prev_mask = None
        
        # Optional: Initialize pose detector if using constraints
        pose_detector = None
        if use_geometric_constraints:
            try:
                from src.core.body_aware_fitter import BodyAwareGarmentFitter
                pose_detector = BodyAwareGarmentFitter()
                print("Pose detector initialized for geometric constraints")
            except Exception as e:
                print(f"Warning: Could not initialize pose detector: {e}")
                use_geometric_constraints = False
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract pose landmarks if using constraints
            pose_landmarks = None
            if use_geometric_constraints and pose_detector:
                try:
                    measurements = pose_detector.extract_body_measurements(frame)
                    if measurements:
                        pose_landmarks = measurements['landmarks']
                except Exception as e:
                    pass  # Skip pose for this frame
            
            # Benchmark this frame
            result = self.benchmark_frame(frame, pose_landmarks, prev_mask)
            
            # Store results
            self.results['occlusion_errors'].append(result['occlusion_error'])
            self.results['temporal_jitter'].append(result['temporal_jitter'])
            self.results['frame_times'].append(result['parse_time'])
            
            prev_mask = result['garment_mask'].copy()
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...", end='\r')
        
        cap.release()
        
        # Calculate summary statistics
        summary = self._calculate_summary(frame_count)
        
        return summary
    
    def benchmark_webcam(
        self,
        duration_seconds: int = 10,
        use_geometric_constraints: bool = True
    ) -> Dict:
        """
        Run benchmark on live webcam feed
        
        Args:
            duration_seconds: How long to capture
            use_geometric_constraints: Whether to use pose landmarks
        
        Returns:
            Summary statistics
        """
        print(f"\nBenchmarking webcam for {duration_seconds} seconds...")
        print(f"Geometric constraints: {'ON' if use_geometric_constraints else 'OFF'}")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")
        
        # Optional: Initialize pose detector
        pose_detector = None
        if use_geometric_constraints:
            try:
                from src.core.body_aware_fitter import BodyAwareGarmentFitter
                pose_detector = BodyAwareGarmentFitter()
                print("Pose detector initialized")
            except Exception as e:
                print(f"Warning: Could not initialize pose detector: {e}")
                use_geometric_constraints = False
        
        start_time = time.time()
        frame_count = 0
        prev_mask = None
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract pose landmarks
            pose_landmarks = None
            if use_geometric_constraints and pose_detector:
                try:
                    measurements = pose_detector.extract_body_measurements(frame)
                    if measurements:
                        pose_landmarks = measurements['landmarks']
                except Exception as e:
                    pass
            
            # Benchmark this frame
            result = self.benchmark_frame(frame, pose_landmarks, prev_mask)
            
            # Store results
            self.results['occlusion_errors'].append(result['occlusion_error'])
            self.results['temporal_jitter'].append(result['temporal_jitter'])
            self.results['frame_times'].append(result['parse_time'])
            
            prev_mask = result['garment_mask'].copy()
            frame_count += 1
            
            # Display progress
            elapsed = time.time() - start_time
            print(f"Frame {frame_count} | {elapsed:.1f}s / {duration_seconds}s", end='\r')
        
        cap.release()
        
        # Calculate summary
        summary = self._calculate_summary(frame_count)
        
        return summary
    
    def _calculate_summary(self, frame_count: int) -> Dict:
        """Calculate summary statistics from results"""
        summary = {
            'total_frames': frame_count,
            'avg_occlusion_error': np.mean(self.results['occlusion_errors']) if self.results['occlusion_errors'] else 0,
            'max_occlusion_error': np.max(self.results['occlusion_errors']) if self.results['occlusion_errors'] else 0,
            'frames_with_errors': sum(1 for e in self.results['occlusion_errors'] if e > 0.05),
            'error_rate_percent': (sum(1 for e in self.results['occlusion_errors'] if e > 0.05) / max(frame_count, 1)) * 100,
            'avg_temporal_jitter': np.mean(self.results['temporal_jitter']) if self.results['temporal_jitter'] else 0,
            'max_temporal_jitter': np.max(self.results['temporal_jitter']) if self.results['temporal_jitter'] else 0,
            'avg_parse_time_ms': np.mean(self.results['frame_times']) * 1000 if self.results['frame_times'] else 0,
            'fps': frame_count / sum(self.results['frame_times']) if self.results['frame_times'] else 0
        }
        
        return summary


def print_summary(summary: Dict, title: str):
    """Print benchmark summary in readable format"""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
    
    print(f"\nFrames Processed: {summary['total_frames']}")
    print(f"Average FPS: {summary['fps']:.1f}")
    print(f"Average Parse Time: {summary['avg_parse_time_ms']:.1f} ms")
    
    print(f"\nOcclusion Quality:")
    print(f"  Average Error: {summary['avg_occlusion_error']:.3f}")
    print(f"  Max Error: {summary['max_occlusion_error']:.3f}")
    print(f"  Frames with Errors (>5%): {summary['frames_with_errors']} ({summary['error_rate_percent']:.1f}%)")
    
    print(f"\nTemporal Stability:")
    print(f"  Average Jitter: {summary['avg_temporal_jitter']:.3f}")
    print(f"  Max Jitter: {summary['max_temporal_jitter']:.3f}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    if summary['error_rate_percent'] < 5:
        print("  [EXCELLENT] Occlusion error rate < 5%")
    elif summary['error_rate_percent'] < 15:
        print("  [GOOD] Occlusion error rate < 15%")
    else:
        print("  [NEEDS IMPROVEMENT] Occlusion error rate > 15%")
    
    if summary['avg_temporal_jitter'] < 0.1:
        print("  [EXCELLENT] Temporal stability high (jitter < 0.1)")
    elif summary['avg_temporal_jitter'] < 0.2:
        print("  [GOOD] Temporal stability acceptable")
    else:
        print("  [NEEDS IMPROVEMENT] Temporal jitter high")


def main():
    """Run benchmark suite"""
    print("=" * 70)
    print("OCCLUSION ERROR BENCHMARK")
    print("=" * 70)
    
    # Initialize parser
    print("\nInitializing semantic parser...")
    parser = SemanticParser(
        backend='auto',
        temporal_smoothing=True,
        onnx_model_path='models/schp_lip.onnx'
    )
    print(f"Backend: {parser.backend.__class__.__name__}")
    
    # Create benchmark
    benchmark = OcclusionBenchmark(parser)
    
    # Option 1: Benchmark webcam
    print("\n" + "=" * 70)
    print("OPTION 1: Webcam Benchmark (10 seconds)")
    print("=" * 70)
    response = input("Run webcam benchmark? (y/n): ")
    
    if response.lower() == 'y':
        summary = benchmark.benchmark_webcam(
            duration_seconds=10,
            use_geometric_constraints=True
        )
        print_summary(summary, "WEBCAM BENCHMARK RESULTS")
    
    # Option 2: Benchmark video file
    print("\n" + "=" * 70)
    print("OPTION 2: Video File Benchmark")
    print("=" * 70)
    video_path = input("Enter video path (or press Enter to skip): ")
    
    if video_path and Path(video_path).exists():
        summary = benchmark.benchmark_video(
            video_path,
            max_frames=100,
            use_geometric_constraints=True
        )
        print_summary(summary, f"VIDEO BENCHMARK: {Path(video_path).name}")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
