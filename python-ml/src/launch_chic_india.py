"""
Chic India AR Platform - Live Camera Demo
Launch the complete AR fashion platform with live camera feed
"""
import cv2
import numpy as np
from typing import Dict, Optional
import time

# Import the complete Chic India platform
from .chic_india_demo import ChicIndiaAREngine

class ChicIndiaLiveDemo:
    """Live camera demo for Chic India AR Platform"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("CHIC INDIA AR FASHION PLATFORM")
        print("Live Camera Demo")
        print("="*70)
        
        # Initialize the AR engine
        print("\nInitializing AR engine...")
        self.engine = ChicIndiaAREngine()
        
        # Load learned corrections
        try:
            self.engine.sku_learner.load()
            print("✓ Loaded SKU corrections")
        except:
            print("⚠ No SKU corrections found")
        
        # Set up initial garment
        self.engine.sizing_pipeline.set_garment("SKU-001")
        
        # Demo state
        self.show_measurements = True
        self.show_recommendations = True
        self.show_stats = True
        self.user_id = "live_demo_user"
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("CONTROLS:")
        print("="*70)
        print("  SPACE  - Add/remove shirt overlay")
        print("  P      - Add/remove pants overlay")
        print("  R      - Show/hide style recommendations")
        print("  M      - Show/hide measurements")
        print("  S      - Show/hide stats")
        print("  Q      - Quit")
        print("="*70 + "\n")
    
    def run(self):
        """Run the live demo"""
        
        # Open camera
        # Open camera (Async for low latency)
        from src.core.async_camera import AsyncCamera
        cap = AsyncCamera(src=0, width=640, height=480).start()
        # cap.set... handled in AsyncCamera init
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return
        
        print("✓ Camera opened successfully")
        print("✓ Platform running...\n")
        
        shirt_active = False
        pants_active = False
        
        while True:
            frame = cap.read()
            if frame is None:
                # print("Waiting for camera...")
                time.sleep(0.01)
                continue
            
            ret = True # Async camera always returns valid frame if not None
            
            # Process frame through Chic India AR engine
            result = self.engine.process_frame(frame, self.user_id)
            
            # Get rendered frame
            display_frame = result['rendered_frame']
            
            # Calculate FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
            
            # Draw UI overlays
            if result['status'] == 'success':
                display_frame = self._draw_ui(display_frame, result)
            else:
                cv2.putText(display_frame, "Position yourself in frame", 
                           (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw stats
            if self.show_stats:
                display_frame = self._draw_stats(display_frame)
            
            # Show frame
            cv2.imshow('Chic India AR Platform', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nShutting down...")
                break
            
            elif key == ord(' '):  # Space - toggle shirt
                if not shirt_active:
                    self.engine.add_garment("shirt_demo", "shirt", "upper")
                    shirt_active = True
                    print("✓ Added shirt overlay")
                else:
                    self.engine.remove_garment("shirt_demo")
                    shirt_active = False
                    print("✓ Removed shirt overlay")
            
            elif key == ord('p'):  # P - toggle pants
                if not pants_active:
                    self.engine.add_garment("pants_demo", "pants", "lower")
                    pants_active = True
                    print("✓ Added pants overlay")
                else:
                    self.engine.remove_garment("pants_demo")
                    pants_active = False
                    print("✓ Removed pants overlay")
            
            elif key == ord('r'):  # R - toggle recommendations
                self.show_recommendations = not self.show_recommendations
                print(f"✓ Style recommendations: {'ON' if self.show_recommendations else 'OFF'}")
            
            elif key == ord('m'):  # M - toggle measurements
                self.show_measurements = not self.show_measurements
                print(f"✓ Measurements: {'ON' if self.show_measurements else 'OFF'}")
            
            elif key == ord('s'):  # S - toggle stats
                self.show_stats = not self.show_stats
                print(f"✓ Stats: {'ON' if self.show_stats else 'OFF'}")
        
        # Cleanup
        cap.stop()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("Session Complete")
        print("="*70)
        print(f"Platform status: Operational")
        print(f"Active garments: {len(self.engine.layer_manager.get_active_garments())}")
        print(f"Recommendations generated: {len(self.engine.recommendations)}")
        print("="*70)
    
    def _draw_ui(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw UI overlays on frame"""
        
        h, w = frame.shape[:2]
        
        # Background panel for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        y_offset = 30
        
        # Title
        cv2.putText(frame, "CHIC INDIA AR", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 40
        
        # Measurements
        if self.show_measurements and result['measurements']:
            cv2.putText(frame, "MEASUREMENTS", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            upper = result['measurements']['upper_body']
            cv2.putText(frame, f"Shoulder: {upper['shoulder_cm']:.1f} cm", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
            cv2.putText(frame, f"Chest: {upper['chest_cm']:.1f} cm", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
            cv2.putText(frame, f"Torso: {upper['torso_cm']:.1f} cm", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 25
            
            # Lower body measurements if available
            if result['measurements']['lower_body']:
                lower = result['measurements']['lower_body']
                cv2.putText(frame, f"Waist: {lower['waist_cm']:.1f} cm", 
                           (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
                cv2.putText(frame, f"Hip: {lower['hip_cm']:.1f} cm", 
                           (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
                cv2.putText(frame, f"Inseam: {lower['inseam_cm']:.1f} cm", 
                           (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 25
        
        # Fit decision
        if 'fit_decisions' in result and result['fit_decisions']:
            cv2.putText(frame, "FIT DECISION", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            decision = result['fit_decisions']['decision']
            color = (0, 255, 0) if decision == 'GOOD' else (0, 165, 255) if decision == 'LOOSE' else (0, 0, 255)
            cv2.putText(frame, f"{decision}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
            
            conf = result['fit_decisions']['confidence']
            cv2.putText(frame, f"Confidence: {conf:.0%}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 30
        
        # Style recommendations
        if self.show_recommendations and result['style_recommendations']:
            cv2.putText(frame, "STYLE ADVICE", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            for i, rec in enumerate(result['style_recommendations'][:2]):  # Top 2
                text = f"{rec['subcategory'].replace('_', ' ').title()}"
                cv2.putText(frame, f"{i+1}. {text}", 
                           (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
                y_offset += 18
            
            y_offset += 15
        
        # Active garments
        active_garments = result['active_garments']
        if active_garments:
            cv2.putText(frame, f"TRYING ON ({len(active_garments)})", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            for garment in active_garments[:3]:
                cv2.putText(frame, f"- {garment}", 
                           (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                y_offset += 18
        
        return frame
    
    def _draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance stats"""
        
        h, w = frame.shape[:2]
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Platform status
        cv2.putText(frame, "LIVE", 
                   (w - 120, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame


def main():
    """Launch the Chic India AR Platform"""
    
    try:
        demo = ChicIndiaLiveDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nThank you for using Chic India AR Platform!")


if __name__ == "__main__":
    main()
