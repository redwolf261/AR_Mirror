"""
Enhanced UI Renderer
Provides rich information panels and overlays
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RenderConfig:
    """Configuration for UI rendering"""
    panel_alpha: float = 0.6
    text_color: Tuple[int, int, int] = (255, 255, 255)
    accent_color: Tuple[int, int, int] = (0, 255, 0)
    warning_color: Tuple[int, int, int] = (0, 165, 255)
    error_color: Tuple[int, int, int] = (0, 0, 255)
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.5
    thickness: int = 1
    line_height: int = 25


class InfoPanelRenderer:
    """Renders information panels on video frames"""
    
    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
    
    def render_main_panel(
        self,
        frame: np.ndarray,
        fps: float,
        body_measurements: Optional[Dict] = None,
        shape_info: Optional[Dict] = None,
        garment_info: Optional[Dict] = None,
        fit_info: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Render main information panel.
        
        Args:
            frame: Input frame (RGB)
            fps: Current FPS
            body_measurements: Dict with shoulder_cm, chest_cm, etc.
            shape_info: Dict with cluster, confidence
            garment_info: Dict with sku, brand
            fit_info: Dict with size, confidence, overall
        
        Returns:
            Frame with rendered panel
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Panel background
        panel_w = 420
        panel_h = 320
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.config.panel_alpha, output, 1 - self.config.panel_alpha, 0, output)
        
        # Draw border
        cv2.rectangle(output, (10, 10), (panel_w, panel_h), self.config.accent_color, 2)
        
        y_pos = 40
        
        # Title
        cv2.putText(output, "AR TRY-ON SYSTEM", (20, y_pos),
                   self.config.font, 0.7, self.config.accent_color, 2)
        y_pos += self.config.line_height + 10
        
        # FPS
        fps_color = self.config.accent_color if fps > 20 else self.config.warning_color
        cv2.putText(output, f"FPS: {fps:.1f}", (20, y_pos),
                   self.config.font, self.config.font_scale, fps_color, self.config.thickness)
        y_pos += self.config.line_height
        
        # Body measurements section
        if body_measurements:
            y_pos += 5
            cv2.putText(output, "BODY MEASUREMENTS", (20, y_pos),
                       self.config.font, 0.5, (200, 200, 255), 1)
            y_pos += self.config.line_height
            
            shoulder = body_measurements.get('shoulder_cm', 0)
            cv2.putText(output, f"  Shoulder: {shoulder:.1f} cm", (20, y_pos),
                       self.config.font, self.config.font_scale, self.config.text_color, 1)
            y_pos += self.config.line_height
            
            chest = body_measurements.get('chest_cm', 0)
            cv2.putText(output, f"  Chest: {chest:.1f} cm", (20, y_pos),
                       self.config.font, self.config.font_scale, self.config.text_color, 1)
            y_pos += self.config.line_height
            
            yaw = body_measurements.get('yaw_signal', 0)
            yaw_deg = yaw * 45  # Approximate degrees
            cv2.putText(output, f"  Rotation: {yaw_deg:.1f}°", (20, y_pos),
                       self.config.font, self.config.font_scale, self.config.text_color, 1)
            y_pos += self.config.line_height
        
        # Shape classification
        if shape_info:
            y_pos += 5
            cv2.putText(output, "BODY SHAPE", (20, y_pos),
                       self.config.font, 0.5, (255, 200, 100), 1)
            y_pos += self.config.line_height
            
            cluster = shape_info.get('cluster', 'Unknown')
            confidence = shape_info.get('confidence', 0)
            cv2.putText(output, f"  {cluster} ({confidence:.1%})", (20, y_pos),
                       self.config.font, self.config.font_scale, self.config.text_color, 1)
            y_pos += self.config.line_height
        
        # Garment info
        if garment_info:
            y_pos += 5
            cv2.putText(output, "CURRENT GARMENT", (20, y_pos),
                       self.config.font, 0.5, (100, 200, 255), 1)
            y_pos += self.config.line_height
            
            sku = garment_info.get('sku', 'None')
            brand = garment_info.get('brand', '')
            cv2.putText(output, f"  {sku} - {brand}", (20, y_pos),
                       self.config.font, self.config.font_scale, self.config.text_color, 1)
            y_pos += self.config.line_height
        
        # Fit assessment
        if fit_info:
            y_pos += 5
            cv2.putText(output, "FIT ASSESSMENT", (20, y_pos),
                       self.config.font, 0.5, (150, 255, 150), 1)
            y_pos += self.config.line_height
            
            size = fit_info.get('size', '?')
            overall = fit_info.get('overall', 'UNKNOWN')
            fit_color = self.config.accent_color if overall == "GOOD" else self.config.warning_color
            
            cv2.putText(output, f"  Size: {size} ({overall})", (20, y_pos),
                       self.config.font, self.config.font_scale, fit_color, self.config.thickness)
            y_pos += self.config.line_height
            
            confidence = fit_info.get('confidence', 'UNKNOWN')
            cv2.putText(output, f"  Confidence: {confidence}", (20, y_pos),
                       self.config.font, self.config.font_scale, self.config.text_color, 1)
            y_pos += self.config.line_height
        
        # Controls help (at bottom)
        y_pos = panel_h - 10
        cv2.putText(output, "← → : Garment | G: Overlay | Q: Quit", (20, y_pos),
                   self.config.font, 0.4, (180, 180, 180), 1)
        
        return output
    
    def render_performance_overlay(
        self,
        frame: np.ndarray,
        timings: Dict[str, float]
    ) -> np.ndarray:
        """
        Render performance metrics overlay (top-right).
        
        Args:
            frame: Input frame
            timings: Dict with pose_ms, depth_ms, render_ms, etc.
        
        Returns:
            Frame with performance overlay
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Panel position (top-right)
        panel_w = 220
        panel_h = 150
        x_start = w - panel_w - 10
        y_start = 10
        
        # Background
        overlay = output.copy()
        cv2.rectangle(overlay, (x_start, y_start), (w - 10, y_start + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.config.panel_alpha, output, 1 - self.config.panel_alpha, 0, output)
        
        # Border
        cv2.rectangle(output, (x_start, y_start), (w - 10, y_start + panel_h), (100, 100, 255), 2)
        
        y_pos = y_start + 30
        
        # Title
        cv2.putText(output, "PERFORMANCE", (x_start + 10, y_pos),
                   self.config.font, 0.5, (100, 100, 255), 1)
        y_pos += self.config.line_height
        
        # Timings
        for key, value in timings.items():
            if key.endswith('_ms'):
                label = key.replace('_ms', '').capitalize()
                cv2.putText(output, f"{label}: {value:.1f}ms", (x_start + 10, y_pos),
                           self.config.font, 0.4, self.config.text_color, 1)
                y_pos += 20
        
        return output
    
    def render_fit_details(
        self,
        frame: np.ndarray,
        fit_result
    ) -> np.ndarray:
        """
        Render detailed fit breakdown (bottom-left).
        
        Args:
            frame: Input frame
            fit_result: FitResult object with component fits
        
        Returns:
            Frame with fit details overlay
        """
        if fit_result is None:
            return frame
        
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Panel position (bottom-left)
        panel_w = 300
        panel_h = 180
        x_start = 10
        y_start = h - panel_h - 10
        
        # Background
        overlay = output.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.config.panel_alpha, output, 1 - self.config.panel_alpha, 0, output)
        
        # Border
        cv2.rectangle(output, (x_start, y_start), (x_start + panel_w, h - 10), (150, 255, 150), 2)
        
        y_pos = y_start + 30
        
        # Title
        cv2.putText(output, "FIT DETAILS", (x_start + 10, y_pos),
                   self.config.font, 0.5, (150, 255, 150), 1)
        y_pos += self.config.line_height
        
        # Component fits
        components = {
            'shoulder': fit_result.shoulder_fit,
            'chest': fit_result.chest_fit,
            'waist': fit_result.waist_fit
        }
        
        for component, fit_value in components.items():
            # Color based on fit
            if fit_value == "GOOD":
                color = self.config.accent_color
            elif fit_value in ["LOOSE", "TIGHT"]:
                color = self.config.warning_color
            else:
                color = self.config.error_color
            
            cv2.putText(output, f"{component.capitalize()}: {fit_value}", (x_start + 10, y_pos),
                       self.config.font, 0.4, color, 1)
            y_pos += 20
        
        # Overall recommendation
        y_pos += 5
        cv2.putText(output, f"→ {fit_result.overall_decision}", (x_start + 10, y_pos),
                   self.config.font, 0.5, self.config.accent_color, 2)
        
        return output


if __name__ == "__main__":
    # Test renderer
    import numpy as np
    
    renderer = InfoPanelRenderer()
    
    # Create test frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    
    # Test data
    body_measurements = {
        'shoulder_cm': 45.2,
        'chest_cm': 98.5,
        'yaw_signal': 0.3
    }
    
    shape_info = {
        'cluster': 'Athletic',
        'confidence': 0.87
    }
    
    garment_info = {
        'sku': 'TSH-001',
        'brand': 'Nike'
    }
    
    fit_info = {
        'size': 'M',
        'overall': 'GOOD',
        'confidence': 'HIGH'
    }
    
    timings = {
        'pose_ms': 15.2,
        'depth_ms': 8.5,
        'render_ms': 12.3
    }
    
    # Render panels
    frame = renderer.render_main_panel(frame, 30.5, body_measurements, shape_info, garment_info, fit_info)
    frame = renderer.render_performance_overlay(frame, timings)
    
    # Display
    cv2.imshow("Renderer Test", frame)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✓ Renderer test complete")
