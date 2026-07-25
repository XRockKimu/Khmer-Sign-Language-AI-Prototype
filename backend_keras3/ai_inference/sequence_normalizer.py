"""
Ported from backend/notebooks/live_testing_script.ipynb, cell 2
("NORMALIZATION MODULE (For Live Testing)"), class SignLanguageNormalizer.

Math preserved verbatim from the notebook. The only omission is
normalize_single_frame(), the notebook's helper for normalizing a
single frame before a live capture buffer has filled -- that is a
streaming/frame-buffer concern, explicitly out of scope for this
milestone (no frame buffering yet), and unused by the batch
normalize_sequence() path this backend needs.
"""

import numpy as np


class SignLanguageNormalizer:
    """
    Normalizer specifically for 516-feature sign language data
    Feature structure: [POSE(132) | HANDS(126) | VELOCITY(258)]
    """

    def __init__(self,
                 use_root_center=True,
                 use_shoulder_width=True,
                 use_hand_normalization=True,
                 use_scale_normalization=True,
                 use_clipping=True,
                 target_hand_size=0.3,
                 clip_bounds=(-2.0, 2.0)):
        self.use_root_center = use_root_center
        self.use_shoulder_width = use_shoulder_width
        self.use_hand_normalization = use_hand_normalization
        self.use_scale_normalization = use_scale_normalization
        self.use_clipping = use_clipping

        self.target_hand_size = target_hand_size
        self.clip_bounds = clip_bounds

        # Feature indices
        self.POSE_START = 0
        self.POSE_END = 132  # 33 landmarks x 4
        self.HANDS_START = 132
        self.HANDS_END = 258  # 126 features
        self.VELOCITY_START = 258
        self.VELOCITY_END = 516  # 258 features

        # MediaPipe landmark indices for pose
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24

        # Hand landmark indices (0-20)
        self.WRIST = 0
        self.FINGERTIPS = [4, 8, 12, 16, 20]

    def extract_pose_landmarks(self, features):
        """Extract pose landmarks from features. Returns: (33, 4) array"""
        pose = features[self.POSE_START:self.POSE_END].reshape(33, 4)
        return pose

    def extract_hand_landmarks(self, features, hand='left'):
        """Extract hand landmarks from features. Returns: (21, 3) array"""
        hand_features = features[self.HANDS_START:self.HANDS_END]

        if hand == 'left':
            hand_lms = hand_features[:63].reshape(21, 3)
        else:  # right
            hand_lms = hand_features[63:].reshape(21, 3)

        return hand_lms

    def extract_velocity(self, features):
        """Extract velocity features"""
        velocity = features[self.VELOCITY_START:self.VELOCITY_END]
        return velocity

    def _normalize_hand_scale(self, hand_landmarks):
        """Normalize hand to target size"""
        wrist = hand_landmarks[self.WRIST]

        distances = []
        for idx in self.FINGERTIPS:
            if idx < len(hand_landmarks):
                dist = np.linalg.norm(hand_landmarks[idx] - wrist)
                distances.append(dist)

        hand_size = np.mean(distances) if distances else 1.0

        if hand_size > 0.001:
            scale_factor = self.target_hand_size / hand_size
            hand_landmarks = hand_landmarks * scale_factor

        return hand_landmarks

    def _compute_velocity(self, pose_hands):
        """Compute velocity from normalized pose+hands"""
        T = pose_hands.shape[0]
        velocity = np.zeros_like(pose_hands)

        # Compute velocity as difference between consecutive frames
        velocity[1:] = pose_hands[1:] - pose_hands[:-1]
        # First frame velocity is zero

        return velocity

    def normalize_sequence(self, features_sequence):
        """
        Normalize a full sequence (30 frames) of 516-feature vectors

        Args:
            features_sequence: (T, 516) array where T = sequence_length (30)

        Returns:
            normalized_sequence: (T, 516) array with normalized features
        """
        T = features_sequence.shape[0]
        normalized_features = []

        # -- First pass: collect shoulder widths --
        shoulder_widths = []
        for t in range(T):
            features = features_sequence[t]
            pose = self.extract_pose_landmarks(features)

            if self.use_shoulder_width:
                left_shoulder = pose[self.LEFT_SHOULDER][:3]
                right_shoulder = pose[self.RIGHT_SHOULDER][:3]
                width = np.linalg.norm(left_shoulder - right_shoulder)
                shoulder_widths.append(width)
            else:
                shoulder_widths.append(1.0)

        shoulder_widths = np.array(shoulder_widths)

        # -- Second pass: normalize each frame --
        for t in range(T):
            features = features_sequence[t].copy()

            # Extract components
            pose = self.extract_pose_landmarks(features)
            left_hand = self.extract_hand_landmarks(features, 'left')
            right_hand = self.extract_hand_landmarks(features, 'right')
            velocity = self.extract_velocity(features)

            # -- 1. Normalize Pose --
            if self.use_root_center:
                left_hip = pose[self.LEFT_HIP][:3]
                right_hip = pose[self.RIGHT_HIP][:3]
                hip_center = (left_hip + right_hip) / 2
                pose[:, :3] = pose[:, :3] - hip_center

            if self.use_shoulder_width:
                width = shoulder_widths[t]
                if width > 0.001:
                    pose[:, :3] = pose[:, :3] / width
                else:
                    pose[:, :3] = 0

            # -- 2. Normalize Hands --
            if self.use_root_center:
                left_hand = left_hand - left_hand[self.WRIST]
                right_hand = right_hand - right_hand[self.WRIST]

            if self.use_scale_normalization:
                left_hand = self._normalize_hand_scale(left_hand)
                right_hand = self._normalize_hand_scale(right_hand)

            if self.use_hand_normalization:
                left_hand[:, 0] = -left_hand[:, 0]

            if self.use_clipping:
                pose[:, :3] = np.clip(pose[:, :3],
                                       self.clip_bounds[0],
                                       self.clip_bounds[1])
                left_hand = np.clip(left_hand,
                                     self.clip_bounds[0],
                                     self.clip_bounds[1])
                right_hand = np.clip(right_hand,
                                      self.clip_bounds[0],
                                      self.clip_bounds[1])

            # -- 3. Combine Features --
            pose_flat = pose.flatten()
            hands_flat = np.concatenate([left_hand.flatten(), right_hand.flatten()])
            combined = np.concatenate([pose_flat, hands_flat])
            combined = np.concatenate([combined, velocity])

            normalized_features.append(combined)

        normalized_features = np.array(normalized_features)

        # -- 4. Recompute Velocity --
        pose_hands = normalized_features[:, :258]
        velocity = self._compute_velocity(pose_hands)
        normalized_features = np.concatenate([pose_hands, velocity], axis=1)

        return normalized_features
