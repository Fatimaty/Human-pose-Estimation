
# utils_vis.py
import cv2
import numpy as np

# COCO order for 17 keypoints
COCO_KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# Skeleton connections (pairs of indices in the COCO order)
SKELETON = [
    (5, 6),  # shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8,10),   # right arm
    (11,12),          # hips
    (5,11), (6,12),   # torso
    (11,13), (13,15), # left leg
    (12,14), (14,16)  # right leg
]

def draw_keypoints_and_skeleton(image, keypoints, scores=None, kp_thresh=0.2):
    """Draw keypoints & skeleton on a BGR image.
    keypoints: (N, 17, 3) [x,y,score] per person OR (17,3)
    """
    out = image.copy()
    persons = keypoints
    if persons.ndim == 2:
        persons = persons[None, ...]

    for i, kps in enumerate(persons):
        color = (0, 255, 0)
        # draw keypoints
        for j, (x, y, v) in enumerate(kps):
            if v >= kp_thresh:
                cv2.circle(out, (int(x), int(y)), 3, (0, 255, 255), -1)
        # draw bones
        for a, b in SKELETON:
            xa, ya, va = kps[a]
            xb, yb, vb = kps[b]
            if va >= kp_thresh and vb >= kp_thresh:
                cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)
    return out
