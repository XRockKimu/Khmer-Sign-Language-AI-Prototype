"""
Ported from backend/notebooks/live_testing_script.ipynb, cell 1
("LANDMARK EXTRACTION MODULE (For Live Testing)").

Converts one MediaPipe Holistic result into the (258,) raw position
vector this backend's pipeline expects: 33-point pose (132, hip-
centered, x/y/z/visibility) + left hand (63, wrist-relative) +
right hand (63, wrist-relative).

Only the pieces the notebook itself marks "KEEP (Works for live
frames)" are ported here -- extract_pose/extract_hand/
extract_raw_features and SmartAdaptiveExtractor. Everything else in
that cell (FrameBuffer, add_velocity) belongs to buffering/formatting,
already handled elsewhere (feature_formatter.py) or explicitly out of
scope for this milestone.

Math preserved verbatim from the notebook -- verified byte-for-byte
identical against the original notebook source using mock landmark
objects (no camera required) before this file was written into the
verification path.
"""

from collections import deque

import cv2
import numpy as np


def extract_pose(pose_lm):
    """Extract 132 pose features"""
    if pose_lm is None:
        return np.zeros(132, dtype=np.float32)
    arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_lm.landmark], dtype=np.float32)
    # Normalize relative to hip midpoint
    hip = (arr[23, :3] + arr[24, :3]) / 2
    arr[:, :3] -= hip
    return arr.flatten()


def extract_hand(hand_lm):
    """Extract 63 hand features"""
    if hand_lm is None:
        return np.zeros(63, dtype=np.float32)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
    return (pts - pts[0]).flatten()


def extract_raw_features(results):
    """
    Extract 258 raw features from a single frame
    This is what you'll use in live testing
    """
    pose_f = extract_pose(results.pose_landmarks)          # 132
    left_f = extract_hand(results.left_hand_landmarks)     # 63
    right_f = extract_hand(results.right_hand_landmarks)   # 63
    return np.concatenate([pose_f, left_f, right_f])        # 258


class SmartAdaptiveExtractor:
    def __init__(self):
        self.detection_history = deque(maxlen=10)
        self.hand_size_history = deque(maxlen=10)
        self.frames_since_detection = 0
        self.stats = {
            'normal': 0,
            'adaptive': 0,
            'failed': 0,
            'total': 0,
        }

    def _hand_size(self, results, h, w):
        sizes = []
        for lm_set in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if lm_set:
                xs = [lm.x * w for lm in lm_set.landmark]
                ys = [lm.y * h for lm in lm_set.landmark]
                sizes.append(max(max(xs) - min(xs), max(ys) - min(ys)))
        return np.mean(sizes) if sizes else 0

    def _has_hands(self, results):
        return bool(results.left_hand_landmarks or results.right_hand_landmarks)

    def _scale_back(self, results, scale):
        for lm_set in [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
            if lm_set:
                for lm in lm_set.landmark:
                    lm.x /= scale
                    lm.y /= scale
        return results

    def extract(self, frame, holistic):
        h, w = frame.shape[:2]
        self.stats['total'] += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        if self._has_hands(results):
            sz = self._hand_size(results, h, w)
            self.hand_size_history.append(sz)
            self.frames_since_detection = 0
            self.stats['normal'] += 1
            return results

        self.frames_since_detection += 1
        avg_sz = np.mean(self.hand_size_history) if self.hand_size_history else 0
        scales = [1.3, 1.6, 2.0] if avg_sz < 80 else [1.3]

        for scale in scales:
            if scale * min(h, w) > 2000:
                continue
            up = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            res = holistic.process(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
            if self._has_hands(res):
                self._scale_back(res, scale)
                self.stats['adaptive'] += 1
                return res

        self.stats['failed'] += 1
        return results

    def get_stats(self):
        t = max(self.stats['total'], 1)
        return {
            'normal_detection_rate': self.stats['normal'] / t,
            'adaptive_detection_rate': self.stats['adaptive'] / t,
            'failed_detection_rate': self.stats['failed'] / t,
        }
