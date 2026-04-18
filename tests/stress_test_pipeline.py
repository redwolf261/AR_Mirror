"""
Stress Test Suite for Hybrid AR Try-On Pipeline
Tests system stability, performance consistency, and error recovery
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import cv2
import time
from collections import deque
import psutil
import threading
from datetime import datetime

from src.hybrid.hybrid_pipeline import HybridTryOnPipeline
from src.viton.viton_integration import VITONGarmentLoader


class StressTestRunner:
    """
    Comprehensive stress testing framework
    Tests: Performance, stability, memory, error handling
    """
    
    def __init__(self):
        self.results = {
            'fps_history': deque(maxlen=1000),
            'frame_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'errors': [],
            'warnings': [],
            'start_time': None,
            'end_time': None,
            'total_frames': 0
        }
        self.pipeline = None
        self.cap = None
    
    def setup(self):
        """Initialize pipeline and camera"""
        print("[SETUP] Setting up stress test environment...\n")
        
        try:
            self.pipeline = HybridTryOnPipeline(use_gpu=False, enable_temporal_stabilization=True)
            print("[OK] Pipeline initialized")
        except Exception as e:
            print(f"[ERROR] Pipeline initialization failed: {e}")
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
        
        self.results['start_time'] = datetime.now()
        return True
    
    def test_1_continuous_operation(self, duration_seconds=30):
        """Test 1: Continuous operation for extended period"""
        print(f"\n{'='*60}")
        print(f"TEST 1: Continuous Operation ({duration_seconds}s)")
        print(f"{'='*60}")
        
        frame_count = 0
        start = time.time()
        garments = ['TSH-001', 'TSH-002', 'SHT-001', 'JKT-001']
        garment_idx = 0
        
        print(f"Running continuous frames...\n")
        
        while time.time() - start < duration_seconds:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_sku = garments[garment_idx % len(garments)]
            
            try:
                t0 = time.time()
                result = self.pipeline.process_frame(rgb_frame, current_sku)
                frame_time = time.time() - t0
                fps = 1.0 / frame_time
                
                self.results['fps_history'].append(fps)
                self.results['frame_times'].append(frame_time)
                
                # Get memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.results['memory_usage'].append(memory_mb)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    avg_fps = np.mean(list(self.results['fps_history'][-30:]))
                    max_mem = max(list(self.results['memory_usage'][-30:]))
                    print(f"  Frame {frame_count:3d}: FPS={avg_fps:5.1f}, Mem={memory_mb:6.1f}MB, Sku={current_sku}")
                
                if frame_count % 15 == 0:
                    garment_idx += 1
            
            except Exception as e:
                self.results['errors'].append(f"Frame {frame_count}: {str(e)}")
                print(f"  ✗ Error at frame {frame_count}: {str(e)[:60]}")
        
        self.results['total_frames'] += frame_count
        return frame_count
    
    def test_2_rapid_garment_switching(self, iterations=50):
        """Test 2: Rapid garment switching (stress memory/cache)"""
        print(f"\n{'='*60}")
        print(f"TEST 2: Rapid Garment Switching ({iterations} changes)")
        print(f"{'='*60}")
        
        garments = ['TSH-001', 'TSH-002', 'SHT-001', 'SHT-002', 'JKT-001', 'JKT-002']
        frame_count = 0
        
        print(f"Switching between garments...\n")
        
        for i in range(iterations):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_sku = garments[i % len(garments)]
            
            try:
                t0 = time.time()
                result = self.pipeline.process_frame(rgb_frame, current_sku)
                frame_time = time.time() - t0
                fps = 1.0 / frame_time
                
                self.results['fps_history'].append(fps)
                self.results['frame_times'].append(frame_time)
                
                frame_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Switched {i+1}/{iterations} times, FPS={fps:.1f}")
            
            except Exception as e:
                self.results['errors'].append(f"Switch {i}: {str(e)}")
        
        self.results['total_frames'] += frame_count
        return frame_count
    
    def test_3_performance_consistency(self, num_frames=100):
        """Test 3: Frame-to-frame performance consistency"""
        print(f"\n{'='*60}")
        print(f"TEST 3: Performance Consistency ({num_frames} frames)")
        print(f"{'='*60}")
        
        frame_times = []
        sku = 'TSH-001'
        
        print(f"Measuring frame time variance...\n")
        
        for i in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                t0 = time.time()
                result = self.pipeline.process_frame(rgb_frame, sku)
                frame_time = time.time() - t0
                frame_times.append(frame_time)
                
                self.results['frame_times'].append(frame_time)
                fps = 1.0 / frame_time
                self.results['fps_history'].append(fps)
            
            except Exception as e:
                self.results['errors'].append(f"Frame {i}: {str(e)}")
        
        # Analyze consistency
        frame_times = np.array(frame_times)
        mean_time = np.mean(frame_times)
        std_time = np.std(frame_times)
        cv = (std_time / mean_time) * 100  # Coefficient of variation
        
        print(f"  Mean frame time:   {mean_time*1000:.1f}ms")
        print(f"  Std deviation:     {std_time*1000:.1f}ms")
        print(f"  Coefficient of var: {cv:.1f}%")
        print(f"  Min/Max:           {np.min(frame_times)*1000:.1f}ms / {np.max(frame_times)*1000:.1f}ms")
        
        self.results['total_frames'] += len(frame_times)
        return len(frame_times)
    
    def test_4_memory_stability(self, num_frames=200):
        """Test 4: Memory usage stability (detect leaks)"""
        print(f"\n{'='*60}")
        print(f"TEST 4: Memory Stability ({num_frames} frames)")
        print(f"{'='*60}")
        
        memory_readings = []
        sku = 'TSH-001'
        
        print(f"Monitoring memory usage...\n")
        
        for i in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                result = self.pipeline.process_frame(rgb_frame, sku)
                
                # Measure memory
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_readings.append(memory_mb)
                self.results['memory_usage'].append(memory_mb)
                
                if (i + 1) % 50 == 0:
                    avg_mem = np.mean(memory_readings[-50:])
                    print(f"  Frame {i+1}: Memory={memory_mb:.1f}MB, 50-frame avg={avg_mem:.1f}MB")
            
            except Exception as e:
                self.results['errors'].append(f"Memory test frame {i}: {str(e)}")
        
        # Analyze memory trend
        memory_readings = np.array(memory_readings)
        initial_mem = memory_readings[0]
        final_mem = memory_readings[-1]
        mem_increase = final_mem - initial_mem
        
        # Linear regression to detect leak
        x = np.arange(len(memory_readings))
        z = np.polyfit(x, memory_readings, 1)
        slope = z[0] * 1000  # MB per frame
        
        print(f"\n  Initial memory:     {initial_mem:.1f}MB")
        print(f"  Final memory:       {final_mem:.1f}MB")
        print(f"  Total increase:     {mem_increase:.1f}MB")
        print(f"  Memory leak rate:   {slope:.3f}MB/frame")
        
        if slope < 0.05:
            print(f"  ✓ No significant memory leak detected")
        else:
            self.results['warnings'].append(f"Possible memory leak: {slope:.3f}MB/frame")
        
        self.results['total_frames'] += len(memory_readings)
        return len(memory_readings)
    
    def test_5_error_recovery(self, num_invalid=20):
        """Test 5: Error handling and recovery"""
        print(f"\n{'='*60}")
        print(f"TEST 5: Error Recovery ({num_invalid} invalid inputs)")
        print(f"{'='*60}")
        
        recovery_count = 0
        sku = 'TSH-001'
        
        print(f"Testing error handling...\n")
        
        for i in range(num_invalid):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Generate invalid inputs
            if i % 4 == 0:
                # Invalid SKU
                invalid_sku = f"INVALID-{i}"
                try:
                    result = self.pipeline.process_frame(rgb_frame, invalid_sku)
                    recovery_count += 1
                except Exception as e:
                    self.results['warnings'].append(f"Handled invalid SKU gracefully: {str(e)[:50]}")
            
            elif i % 4 == 1:
                # Corrupted frame (all zeros)
                corrupted_frame = np.zeros_like(rgb_frame)
                try:
                    result = self.pipeline.process_frame(corrupted_frame, sku)
                    recovery_count += 1
                except Exception as e:
                    self.results['warnings'].append(f"Handled corrupted frame: {str(e)[:50]}")
            
            elif i % 4 == 2:
                # Very small frame
                tiny_frame = cv2.resize(rgb_frame, (50, 50))
                try:
                    result = self.pipeline.process_frame(tiny_frame, sku)
                    recovery_count += 1
                except Exception as e:
                    self.results['warnings'].append(f"Handled tiny frame: {str(e)[:50]}")
            
            else:
                # Normal frame
                try:
                    result = self.pipeline.process_frame(rgb_frame, sku)
                    recovery_count += 1
                except Exception as e:
                    self.results['errors'].append(f"Normal frame processing failed: {str(e)}")
        
        print(f"\n  Error scenarios: {num_invalid}")
        print(f"  Successful recovery: {recovery_count}/{num_invalid}")
        print(f"  Recovery rate: {(recovery_count/num_invalid)*100:.1f}%")
        
        self.results['total_frames'] += recovery_count
        return recovery_count
    
    def generate_report(self):
        """Generate comprehensive stress test report"""
        print(f"\n{'='*60}")
        print(f"STRESS TEST REPORT")
        print(f"{'='*60}\n")
        
        fps_array = np.array(list(self.results['fps_history']))
        frame_time_array = np.array(list(self.results['frame_times']))
        memory_array = np.array(list(self.results['memory_usage']))
        
        # Performance metrics
        print("📊 PERFORMANCE METRICS")
        print(f"  Total frames processed: {self.results['total_frames']}")
        print(f"  Average FPS:           {np.mean(fps_array):.1f}")
        print(f"  Min FPS:               {np.min(fps_array):.1f}")
        print(f"  Max FPS:               {np.max(fps_array):.1f}")
        print(f"  Std deviation:         {np.std(fps_array):.2f}")
        
        print(f"\n  Average frame time:    {np.mean(frame_time_array)*1000:.1f}ms")
        print(f"  Min frame time:        {np.min(frame_time_array)*1000:.1f}ms")
        print(f"  Max frame time:        {np.max(frame_time_array)*1000:.1f}ms")
        print(f"  95th percentile:       {np.percentile(frame_time_array, 95)*1000:.1f}ms")
        
        # Memory metrics
        print(f"\n💾 MEMORY METRICS")
        print(f"  Initial memory:        {memory_array[0]:.1f}MB")
        print(f"  Final memory:          {memory_array[-1]:.1f}MB")
        print(f"  Peak memory:           {np.max(memory_array):.1f}MB")
        print(f"  Average memory:        {np.mean(memory_array):.1f}MB")
        
        # Stability assessment
        print(f"\n⚡ STABILITY ASSESSMENT")
        consistency_score = 100 - (np.std(fps_array) / np.mean(fps_array) * 100)
        print(f"  FPS consistency:       {consistency_score:.1f}% ({'[EXCELLENT]' if consistency_score > 95 else '[FAIR]'})")
        
        # Error summary
        print(f"\n🔍 ERROR SUMMARY")
        print(f"  Total errors:          {len(self.results['errors'])}")
        print(f"  Total warnings:        {len(self.results['warnings'])}")
        
        if self.results['errors']:
            print(f"\n  Errors:")
            for error in self.results['errors'][:5]:
                print(f"    - {error[:70]}")
        
        # Status
        print(f"\n{'='*60}")
        if len(self.results['errors']) == 0 and consistency_score > 90:
            print(f"✅ STRESS TEST PASSED - System is production-ready")
        elif len(self.results['errors']) < 5:
            print(f"[PASSED-WARNINGS] STRESS TEST PASSED WITH WARNINGS - Monitor noted issues")
        else:
            print(f"❌ STRESS TEST FAILED - Review errors above")
        print(f"{'='*60}\n")
        
        return {
            'avg_fps': np.mean(fps_array),
            'consistency': consistency_score,
            'errors': len(self.results['errors']),
            'warnings': len(self.results['warnings'])
        }
    
    def run_all(self):
        """Run complete stress test suite"""
        if not self.setup():
            return False
        
        print("\n[START STRESS TEST SUITE]")
        print("=" * 60)
        print("HYBRID AR TRY-ON PIPELINE - STRESS TEST SUITE")
        
        try:
            self.test_1_continuous_operation(duration_seconds=30)
            self.test_2_rapid_garment_switching(iterations=50)
            self.test_3_performance_consistency(num_frames=100)
            self.test_4_memory_stability(num_frames=200)
            self.test_5_error_recovery(num_invalid=20)
        
        except KeyboardInterrupt:
            print("\n\n[INTERRUPT] Stress test interrupted by user")
        
        except Exception as e:
            print(f"\n\n[FATAL] Stress test error: {e}")
            self.results['errors'].append(f"Test suite error: {str(e)}")
        
        finally:
            self.results['end_time'] = datetime.now()
            self.generate_report()
            
            if self.cap:
                self.cap.release()
            
            print(f"Total test duration: {(self.results['end_time'] - self.results['start_time']).total_seconds():.1f}s")


if __name__ == "__main__":
    runner = StressTestRunner()
    runner.run_all()
