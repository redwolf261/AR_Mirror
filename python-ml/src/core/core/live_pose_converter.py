import numpy as np
import cv2

class LivePoseConverter:
    """ 
    Converts MediaPipe Pose landmarks (33 points) to OpenPose Heatmaps (18 channels)
    Required for CP-VTON+ input.
    """
    def __init__(self, height=256, width=192):
        self.height = height
        self.width = width
        # Mapping MediaPipe index -> OpenPose index
        # MP: 0:nose, 11:left_shoulder, 12:right_shoulder, 23:left_hip, 24:right_hip, ...
        # OP: 0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist, 5:LShoulder, ...
        self.mp_to_op = {
            0: 0,   # Nose
            # Neck is calculated (avg of shoulders)
            12: 2,  # RShoulder
            14: 3,  # RElbow
            16: 4,  # RWrist
            11: 5,  # LShoulder
            13: 6,  # LElbow
            15: 7,  # LWrist
            24: 8,  # RHip
            26: 9,  # RKnee
            28: 10, # RAnkle
            23: 11, # LHip
            25: 12, # LKnee
            27: 13, # LAnkle
            # Eyes/Ears mapped if needed
        }

    def convert_to_heatmap(self, landmarks):
        """
        Args:
            landmarks: List of {x, y, visibility} from MediaPipe
        Returns:
            heatmap: (18, height, width) numpy array
        """
        heatmap = np.zeros((18, self.height, self.width), dtype=np.float32)
        
        if not landmarks:
            return heatmap

        # Extract keypoints
        points = {}
        for mp_idx, op_idx in self.mp_to_op.items():
            lm = landmarks[mp_idx]
            points[op_idx] = (int(lm.x * self.width), int(lm.y * self.height))

        # Calculate Neck (Average of shoulders)
        if 5 in points and 2 in points:
            neck_x = (points[5][0] + points[2][0]) // 2
            neck_y = (points[5][1] + points[2][1]) // 2
            points[1] = (neck_x, neck_y)

        # Draw Gaussians
        for idx, (x, y) in points.items():
            if 0 <= x < self.width and 0 <= y < self.height:
                self._add_gaussian(heatmap[idx], x, y)
                
        return heatmap

    def _add_gaussian(self, heatmap_channel, x, y, sigma=4):
        tmp_size = sigma * 3
        ul = [int(x - tmp_size), int(y - tmp_size)]
        br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
        
        size = 2 * tmp_size + 1
        x0 = np.arange(0, size, 1, dtype=float)
        y0 = x0[:, np.newaxis]
        x00 = size // 2
        y00 = size // 2
        g = np.exp(- ((x0 - x00) ** 2 + (y0 - y00) ** 2) / (2 * sigma ** 2))
        
        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], self.width) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], self.height) - ul[1]
        
        # Image range
        img_x = max(0, ul[0]), min(br[0], self.width)
        img_y = max(0, ul[1]), min(br[1], self.height)

        heatmap_channel[img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
