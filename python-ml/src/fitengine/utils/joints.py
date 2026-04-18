"""
Joint index maps.

RTMPOSE_133_TO_BODY33: selects 33-point body subset from RTMPose-133 output.
STAR_24_TO_BODY33:     maps STAR's 24 joints to body-33 positions (for reprojection loss).

Body-33 convention (used everywhere in FitEngine):
  0  nose          1  left_eye        2  right_eye
  3  left_ear      4  right_ear       5  left_shoulder
  6  right_shoulder 7 left_elbow      8  right_elbow
  9  left_wrist   10  right_wrist    11  left_hip
  12 right_hip    13  left_knee      14  right_knee
  15 left_ankle   16  right_ankle    17  left_heel
  18 right_heel   19  left_foot_idx  20  right_foot_idx
  21 left_pinky   22  right_pinky    23  left_index
  24 right_index  25  left_thumb     26  right_thumb
  27 neck         28  left_big_toe   29  right_big_toe
  30 left_small_toe 31 right_small_toe 32 mid_hip
"""

import numpy as np

# ---------------------------------------------------------------------------
# RTMPose-133 → Body-33
# RTMPose-133 keypoint ordering (subset relevant here):
#   0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
#   5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
#   9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
#   13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle,
#   17:left_big_toe, 18:left_small_toe, 19:left_heel,
#   20:right_big_toe, 21:right_small_toe, 22:right_heel,
#   ...133 hand/face landmarks follow
# ---------------------------------------------------------------------------
RTMPOSE_133_TO_BODY33: list[int] = [
    0,   # 0  nose
    1,   # 1  left_eye
    2,   # 2  right_eye
    3,   # 3  left_ear
    4,   # 4  right_ear
    5,   # 5  left_shoulder
    6,   # 6  right_shoulder
    7,   # 7  left_elbow
    8,   # 8  right_elbow
    9,   # 9  left_wrist
    10,  # 10 right_wrist
    11,  # 11 left_hip
    12,  # 12 right_hip
    13,  # 13 left_knee
    14,  # 14 right_knee
    15,  # 15 left_ankle
    16,  # 16 right_ankle
    19,  # 17 left_heel
    22,  # 18 right_heel
    17,  # 19 left_foot_idx (big toe)
    20,  # 20 right_foot_idx (big toe)
    # Approximate hand landmarks via wrist copies (refined in Phase 2)
    9,   # 21 left_pinky  ← wrist proxy
    10,  # 22 right_pinky ← wrist proxy
    9,   # 23 left_index  ← wrist proxy
    10,  # 24 right_index ← wrist proxy
    9,   # 25 left_thumb  ← wrist proxy
    10,  # 26 right_thumb ← wrist proxy
    -1,  # 27 neck        ← synthesized below
    17,  # 28 left_big_toe
    20,  # 29 right_big_toe
    18,  # 30 left_small_toe
    21,  # 31 right_small_toe
    -1,  # 32 mid_hip     ← synthesized below
]

# Joints that must be synthesized (not directly available in RTMPose-133)
# neck  = midpoint(left_shoulder, right_shoulder)
# mid_hip = midpoint(left_hip, right_hip)
_SYNTHESIZED = {27: (5, 6), 32: (11, 12)}


def rtmpose133_to_body33(kp133: np.ndarray) -> np.ndarray:
    """
    Map RTMPose-133 keypoints [133, 3] → body-33 [33, 3] (x, y, conf).
    Synthesized joints (neck, mid_hip) are averaged from their parents.
    """
    assert kp133.shape == (133, 3), f"Expected [133,3], got {kp133.shape}"
    out = np.zeros((33, 3), dtype=np.float32)
    for dst, src in enumerate(RTMPOSE_133_TO_BODY33):
        if src == -1:
            a, b = _SYNTHESIZED[dst]
            out[dst] = (kp133[a] + kp133[b]) / 2.0
        else:
            out[dst] = kp133[src]
    return out


# ---------------------------------------------------------------------------
# STAR-24 → Body-33
# STAR joint ordering (24 joints):
#   0 pelvis, 1 left_hip, 2 right_hip, 3 spine1, 4 left_knee,
#   5 right_knee, 6 spine2, 7 left_ankle, 8 right_ankle, 9 spine3,
#   10 left_foot, 11 right_foot, 12 neck, 13 left_collar, 14 right_collar,
#   15 head, 16 left_shoulder, 17 right_shoulder, 18 left_elbow,
#   19 right_elbow, 20 left_wrist, 21 right_wrist, 22 left_hand,
#   23 right_hand
# ---------------------------------------------------------------------------
STAR_24_TO_BODY33: dict[int, int] = {
    # body33_idx : star_joint_idx
    5:  16,  # left_shoulder
    6:  17,  # right_shoulder
    7:  18,  # left_elbow
    8:  19,  # right_elbow
    9:  20,  # left_wrist
    10: 21,  # right_wrist
    11:  1,  # left_hip
    12:  2,  # right_hip
    13:  4,  # left_knee
    14:  5,  # right_knee
    15:  7,  # left_ankle
    16:  8,  # right_ankle
    27: 12,  # neck
    32:  0,  # mid_hip (pelvis)
}


def normalize_keypoints(
    kp: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Normalize keypoints [N, 3] from pixel coords to [0, 1] range.
    x divided by img_w, y divided by img_h, confidence unchanged.
    """
    out = kp.copy().astype(np.float32)
    out[:, 0] /= float(img_w)
    out[:, 1] /= float(img_h)
    return out
