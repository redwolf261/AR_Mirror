
# Chic India AR Platform - High Performance Async Implementation
# "Masterpiece" Architecture: Async Pipelining + Adaptive LOD + Telemetry

import cv2
import numpy as np
import time
import argparse
import sys
import os
from typing import Dict, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Core Architecture
from src.core.async_camera import AsyncCamera
from src.core.pipeline import Stage, AsyncPipeline, ResourceManager
from src.core.telemetry import PerformanceMonitor
from .chic_india_demo import ChicIndiaAREngine

# Define the Inference Stage
class InferenceStage(Stage):
    def __init__(self, engine: ChicIndiaAREngine, monitor: PerformanceMonitor):
        super().__init__("Inference")
        self.engine = engine
        self.monitor = monitor
        self.frame_count = 0
        self.adaptive_downscale = 1.0
        self.low_perf_mode = False
        self.hysteresis_counter = 0
    
    def process(self, input_data):
        """
        Input: {'frame': np.ndarray, 'user_id': str}
        Output: Processed result dict
        """
        frame = input_data['frame']
        user_id = input_data.get('user_id', 'user_async')
        
        # Adaptive Logic: Automatic toggle based on perf
        stats = self.monitor.get_stats("Inference")
        avg_ms = stats['avg']
        
        self.hysteresis_counter += 1
        if self.hysteresis_counter > 30: # Check every 30 frames
            if not self.low_perf_mode and avg_ms > 100: # If slower than 10 FPS
                self.low_perf_mode = True
                print("[Auto-Config] Switching to Low Perf Mode (High Latency)")
                self.hysteresis_counter = 0
            elif self.low_perf_mode and avg_ms < 50: # If faster than 20 FPS
                self.low_perf_mode = False
                print("[Auto-Config] Switching to High Quality Mode")
                self.hysteresis_counter = 0
        
        h, w = frame.shape[:2]
        if self.low_perf_mode:
            # Downscale for inference speed
            proc_frame = cv2.resize(frame, (320, 240))
        else:
            proc_frame = frame
            
        # Run heavy inference
        result = self.engine.process_frame(proc_frame, user_id=user_id)
        
        return {
            'original_shape': (w, h),
            'ar_result': result,
            'timestamp': time.time(),
            'low_perf_mode': self.low_perf_mode
        }

def draw_telemetry(frame, monitor: PerformanceMonitor, low_perf_mode: bool = False):
    """Overlay telemetry data on the main UI"""
    y = 30
    stats_inf = monitor.get_stats("Inference")
    fps_inf = monitor.get_fps("Inference")
    
    # Background
    cv2.rectangle(frame, (10, 5), (420, 60), (0, 0, 0), -1)
    
    # Text
    mode_text = "LQ" if low_perf_mode else "HQ"
    mode_color = (0, 165, 255) if low_perf_mode else (0, 255, 0)
    
    cv2.putText(frame, f"UI FPS: {monitor.get_fps('UI'):.1f}", (20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"AI FPS: {fps_inf:.1f} ({stats_inf['avg']:.1f}ms) [{mode_text}]", (160, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    return frame

def main():
    print("=" * 70)
    print("CHIC INDIA AR - MASTERPIECE ARCHITECTURE (ASYNC)")
    print("Initializing...")
    
    # 1. Initialize Components
    monitor = PerformanceMonitor()
    
    # Initialize Engine (Heavy Load)
    print("Loading AI Engine...")
    engine = ChicIndiaAREngine()
    
    # Create Pipeline
    pipeline = AsyncPipeline(monitor=monitor)
    inference_stage = InferenceStage(engine, monitor)
    
    # 2. Start Camera (Async)
    camera = AsyncCamera(src=0, width=640, height=480)
    camera.start()
    
    # 3. Start Inference Pipeline
    pipeline.start(inference_stage)
    
    # 4. Main Event Loop (UI Thread)
    print("Starting Main Loop (UI)...")
    print("Press 'q' to quit, 'd' to toggle debug, 's' to toggle performance mode")
    
    show_debug = True
    last_inference_result = None
    
    try:
        while True:
            # Measure UI Timing
            monitor.start_timer("UI")
            
            # A. Get latest frame
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
                
            # B. Submit frame to AI (Non-blocking)
            pipeline.submit({'frame': frame.copy(), 'user_id': "demo_user"})
            
            # C. Check for new AI results
            new_result = pipeline.get_result()
            if new_result:
                last_inference_result = new_result
            
            # D. Render
            display_frame = frame.copy()
            
            # Overlay AR if we have a result
            if last_inference_result:
                ar_data = last_inference_result['ar_result']
                
                # Check execution mode
                # If we downscaled in inference, we must upscale mask/overlay here?
                # The current engine returns a fully rendered frame. 
                # Ideally, engine returns MASKS + ASSETS, but for this demo phase,
                # we will just take the rendered output and blend it or display it.
                # Since engine.process_frame returns 'rendered_frame', we use that.
                
                # If resolution matches, we can just use the rendered frame
                # Careful: The 'rendered_frame' currently is the full output. 
                # If we used downscaled input, the output is downscaled. 
                rendered = ar_data['rendered_frame']
                
                if rendered.shape[:2] != display_frame.shape[:2]:
                    rendered = cv2.resize(rendered, (display_frame.shape[1], display_frame.shape[0]))
                
                display_frame = rendered
            
            # E. Telemetry Overlay
            if show_debug:
                is_low_perf = last_inference_result['low_perf_mode'] if last_inference_result else False
                display_frame = draw_telemetry(display_frame, monitor, is_low_perf)
            
            cv2.imshow("Chic India AR (Masterpiece Architecture)", display_frame)
            
            monitor.stop_timer("UI")
            
            # F. Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_debug = not show_debug
            elif key == ord('a'):
                # Add garment demo
                print("Adding demo garment...")
                engine.add_garment("shirt_001", "shirt", "upper")
            
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        camera.stop()
        pipeline.stop()
        cv2.destroyAllWindows()
        monitor.print_report()

if __name__ == "__main__":
    main()
