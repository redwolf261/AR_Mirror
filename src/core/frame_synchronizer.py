"""
Frame synchronizer for aligning multi-modal inputs (pose, segmentation, depth)
Ensures temporal consistency across different inference speeds
"""

import numpy as np
from typing import Dict, Optional, Any
from collections import deque
import time


class FrameSynchronizer:
    """Synchronize pose, segmentation, and depth results across frames"""
    
    def __init__(self, max_age_ms: float = 100.0):
        """
        Initialize synchronizer.
        
        Args:
            max_age_ms: Maximum age of cached results before they're considered stale
        """
        self.max_age_ms = max_age_ms
        self.cache: Dict[str, Optional[Any]] = {
            'pose': None,
            'segmentation': None,
            'depth': None
        }
        self.timestamps = {
            'pose': 0.0,
            'segmentation': 0.0,
            'depth': 0.0
        }
        self.frame_count = 0
    
    def update(
        self,
        frame_id: int,
        pose: Optional[Any] = None,
        segmentation: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None
    ):
        """
        Update cache with new results.
        
        Args:
            frame_id: Current frame number
            pose: Pose detection result
            segmentation: Segmentation mask
            depth: Depth map
        """
        current_time = time.time() * 1000  # milliseconds
        
        if pose is not None:
            self.cache['pose'] = pose
            self.timestamps['pose'] = current_time
        
        if segmentation is not None:
            self.cache['segmentation'] = segmentation
            self.timestamps['segmentation'] = current_time
        
        if depth is not None:
            self.cache['depth'] = depth
            self.timestamps['depth'] = current_time
        
        self.frame_count = frame_id
    
    def get_synchronized(self) -> Dict[str, Any]:
        """
        Get synchronized multi-modal data.
        
        Returns:
            Dictionary with current best estimates for pose, segmentation, depth
            Each value may be None if not available or stale
        """
        current_time = time.time() * 1000
        result = {}
        
        for key in ['pose', 'segmentation', 'depth']:
            age = current_time - self.timestamps[key]
            if age < self.max_age_ms:
                result[key] = self.cache[key]
            else:
                result[key] = None  # Too old, consider unavailable
        
        return result
    
    def is_complete(self) -> bool:
        """Check if all modalities are available and fresh"""
        synchronized = self.get_synchronized()
        return all(v is not None for v in synchronized.values())
    
    def get_staleness(self) -> Dict[str, float]:
        """Get age of each cached result in milliseconds"""
        current_time = time.time() * 1000
        return {
            key: current_time - self.timestamps[key]
            for key in ['pose', 'segmentation', 'depth']
        }


if __name__ == "__main__":
    # Test synchronizer
    sync = FrameSynchronizer(max_age_ms=100.0)
    
    # Simulate different inference speeds
    sync.update(frame_id=0, pose={'landmarks': []}, depth=np.zeros((480, 640)))
    time.sleep(0.03)  # 30ms
    
    sync.update(frame_id=1, segmentation=np.zeros((480, 640), dtype=np.uint8))
    
    result = sync.get_synchronized()
    print(f"✓ Synchronized data keys: {list(result.keys())}")
    print(f"✓ Complete: {sync.is_complete()}")
    print(f"✓ Staleness: {sync.get_staleness()}")
