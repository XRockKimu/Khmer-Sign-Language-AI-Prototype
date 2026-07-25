"""
Ported from backend/notebooks/live_testing_script.ipynb, cell 3
("FEATURE ENGINEERING MODULE (For Live Testing)"), class
FeatureEngineeringEngine.

Math preserved verbatim from the notebook, including method names.
One method is intentionally NOT ported: compute_finger_relative_positions().
It is defined in the notebook but never called from process_frame() or
process_sequence() -- confirmed by reading the full method body -- so
porting it would add dead code with no effect on the 686-feature output.
"""

import numpy as np


class FeatureEngineeringEngine:
    """
    Feature engineering specifically for 516-feature sign language data
    Data structure: [POSE(132) | HANDS(126) | VELOCITY(258)]
    """

    def __init__(self):
        # Feature indices
        self.POSE_START = 0
        self.POSE_END = 132
        self.HANDS_START = 132
        self.HANDS_END = 258
        self.VELOCITY_START = 258
        self.VELOCITY_END = 516

        # MediaPipe landmark indices for pose
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.NOSE = 0
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14

        # Hand fingertip indices (0-20)
        self.HAND_INDICES = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20,
            'thumb_base': 1,
            'index_base': 5,
            'middle_base': 9,
            'ring_base': 13,
            'pinky_base': 17
        }

        # Joint chains for angle calculation
        self.JOINT_CHAINS = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

        self.feature_dims = {}

        # For velocity/acceleration calculation across a sequence
        self.prev_features = None
        self.prev_velocity = None
        self.frame_count = 0

    # ============================================================
    # EXTRACT COMPONENTS FROM 516 FEATURES
    # ============================================================

    def extract_pose_landmarks(self, features):
        """Extract pose landmarks (33, 4) from 516 features"""
        return features[self.POSE_START:self.POSE_END].reshape(33, 4)

    def extract_left_hand(self, features):
        """Extract left hand landmarks (21, 3) from 516 features"""
        hand_features = features[self.HANDS_START:self.HANDS_END]
        return hand_features[:63].reshape(21, 3)

    def extract_right_hand(self, features):
        """Extract right hand landmarks (21, 3) from 516 features"""
        hand_features = features[self.HANDS_START:self.HANDS_END]
        return hand_features[63:].reshape(21, 3)

    def extract_velocity(self, features):
        """Extract velocity (258) from 516 features"""
        return features[self.VELOCITY_START:self.VELOCITY_END]

    # ============================================================
    # 1. RELATIVE JOINT COORDINATES
    # ============================================================

    def compute_relative_coordinates(self, hand_landmarks):
        """Compute coordinates relative to wrist (21, 3) -> 63 features"""
        wrist = hand_landmarks[0]
        relative = hand_landmarks - wrist
        return relative.flatten()

    # ============================================================
    # 2. HAND-TO-BODY DISTANCES
    # ============================================================

    def compute_hand_to_body_distances(self, hand_landmarks, pose_landmarks):
        """Compute distances from hand center to body parts (7 distances)"""
        hand_center = np.mean(hand_landmarks, axis=0)

        body_keypoints = {
            'nose': pose_landmarks[self.NOSE][:3],
            'left_shoulder': pose_landmarks[self.LEFT_SHOULDER][:3],
            'right_shoulder': pose_landmarks[self.RIGHT_SHOULDER][:3],
            'left_hip': pose_landmarks[self.LEFT_HIP][:3],
            'right_hip': pose_landmarks[self.RIGHT_HIP][:3],
            'left_elbow': pose_landmarks[self.LEFT_ELBOW][:3],
            'right_elbow': pose_landmarks[self.RIGHT_ELBOW][:3]
        }

        distances = []
        for name, pos in body_keypoints.items():
            dist = np.linalg.norm(hand_center - pos)
            distances.append(dist)

        return np.array(distances)

    def compute_wrist_to_shoulder_ratio(self, hand_landmarks, pose_landmarks, hand_type='left'):
        """Compute ratio of wrist distance to shoulder width (1 feature)"""
        wrist = hand_landmarks[0]

        if hand_type == 'left':
            shoulder = pose_landmarks[self.LEFT_SHOULDER][:3]
        else:
            shoulder = pose_landmarks[self.RIGHT_SHOULDER][:3]

        wrist_to_shoulder = np.linalg.norm(wrist - shoulder)

        shoulder_width = np.linalg.norm(
            pose_landmarks[self.LEFT_SHOULDER][:3] -
            pose_landmarks[self.RIGHT_SHOULDER][:3]
        )

        if shoulder_width > 0.001:
            ratio = wrist_to_shoulder / shoulder_width
        else:
            ratio = 0

        return np.array([ratio])

    # ============================================================
    # 3. JOINT ANGLES
    # ============================================================

    def compute_hand_angles(self, hand_landmarks):
        """Compute angles for all finger joints (15 angles)"""
        angles = []

        for finger_name, indices in self.JOINT_CHAINS.items():
            if len(indices) >= 3:
                for i in range(len(indices) - 2):
                    p1 = hand_landmarks[indices[i]]
                    p2 = hand_landmarks[indices[i + 1]]
                    p3 = hand_landmarks[indices[i + 2]]
                    angle = self._compute_angle(p1, p2, p3)
                    angles.append(angle)

        return np.array(angles)

    def _compute_angle(self, p1, p2, p3):
        """Compute angle between three points in degrees"""
        v1 = p1 - p2
        v2 = p3 - p2

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 0.001 and v2_norm > 0.001:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
        else:
            angle = 0

        return angle

    def compute_hand_orientation_angles(self, hand_landmarks):
        """Compute overall hand orientation (2 features)"""
        wrist = hand_landmarks[0]
        index_base = hand_landmarks[5]
        pinky_base = hand_landmarks[17]

        v1 = index_base - wrist
        v2 = pinky_base - wrist

        palm_normal = np.cross(v1, v2)
        if np.linalg.norm(palm_normal) > 0.001:
            palm_normal = palm_normal / np.linalg.norm(palm_normal)

        roll = np.arctan2(palm_normal[1], palm_normal[2]) * 180 / np.pi
        pitch = np.arctan2(-palm_normal[0], np.sqrt(palm_normal[1] ** 2 + palm_normal[2] ** 2)) * 180 / np.pi

        return np.array([roll, pitch])

    # ============================================================
    # 4. VELOCITY & ACCELERATION
    # ============================================================

    def process_frame(self, features_516):
        """
        Process a single frame of 516 features to 686 engineered features

        Args:
            features_516: (516,) array

        Returns:
            features_686: (686,) array
        """
        # Extract components
        pose = self.extract_pose_landmarks(features_516)
        left_hand = self.extract_left_hand(features_516)
        right_hand = self.extract_right_hand(features_516)
        velocity = self.extract_velocity(features_516)

        # -- Static Features --
        static_features = []

        # 1. Relative coordinates (left: 63, right: 63)
        left_relative = self.compute_relative_coordinates(left_hand)
        right_relative = self.compute_relative_coordinates(right_hand)
        static_features.append(left_relative)
        static_features.append(right_relative)

        # 2. Hand-to-body distances (left: 7, right: 7)
        left_distances = self.compute_hand_to_body_distances(left_hand, pose)
        right_distances = self.compute_hand_to_body_distances(right_hand, pose)
        static_features.append(left_distances)
        static_features.append(right_distances)

        # 3. Joint angles (left: 15, right: 15)
        left_angles = self.compute_hand_angles(left_hand)
        right_angles = self.compute_hand_angles(right_hand)
        static_features.append(left_angles)
        static_features.append(right_angles)

        # 4. Hand orientation (left: 2, right: 2)
        left_orientation = self.compute_hand_orientation_angles(left_hand)
        right_orientation = self.compute_hand_orientation_angles(right_hand)
        static_features.append(left_orientation)
        static_features.append(right_orientation)

        # 5. Wrist-to-shoulder ratio (left: 1, right: 1)
        left_ratio = self.compute_wrist_to_shoulder_ratio(left_hand, pose, 'left')
        right_ratio = self.compute_wrist_to_shoulder_ratio(right_hand, pose, 'right')
        static_features.append(left_ratio)
        static_features.append(right_ratio)

        # Combine static features (166 features)
        static_combined = np.concatenate(static_features)

        # -- Dynamic Features --
        self.frame_count += 1

        # 6. Base features for velocity (pose + hands): 258
        pose_flat = pose.flatten()
        left_flat = left_hand.flatten()
        right_flat = right_hand.flatten()
        base = np.concatenate([pose_flat, left_flat, right_flat])

        # 7. Velocity: Use existing velocity (258)
        velocity = self.extract_velocity(features_516)

        # 8. Speed (1)
        speed = np.array([np.linalg.norm(velocity)], dtype=np.float32)

        # 9. Acceleration (258)
        if self.prev_features is not None:
            acceleration = velocity - self.prev_velocity
        else:
            acceleration = np.zeros(258, dtype=np.float32)

        # 10. Acceleration magnitude (1)
        accel_mag = np.array([np.linalg.norm(acceleration)], dtype=np.float32)

        # 11. Motion direction (2)
        if np.linalg.norm(velocity) > 0.001:
            direction = velocity / np.linalg.norm(velocity)
            azimuth = np.arctan2(direction[1], direction[0]) * 180 / np.pi
            elevation = np.arctan2(direction[2], np.sqrt(direction[0] ** 2 + direction[1] ** 2)) * 180 / np.pi
            direction_vec = np.array([azimuth, elevation], dtype=np.float32)
        else:
            direction_vec = np.zeros(2, dtype=np.float32)

        # -- Combine ALL Features (686) --
        all_features = np.concatenate([
            static_combined,   # 166
            velocity,          # 258
            speed,             # 1
            acceleration,      # 258
            accel_mag,         # 1
            direction_vec      # 2
        ])

        # Update previous values
        self.prev_features = base.copy()
        self.prev_velocity = velocity.copy()

        return all_features

    def process_sequence(self, features_sequence):
        """
        Process a full sequence of 516-feature frames

        Args:
            features_sequence: (T, 516) array where T = 30

        Returns:
            engineered_sequence: (T, 686) array
        """
        T = features_sequence.shape[0]
        engineered_frames = []

        # Reset state for sequence processing
        self.prev_features = None
        self.prev_velocity = None
        self.frame_count = 0

        for t in range(T):
            engineered = self.process_frame(features_sequence[t])
            engineered_frames.append(engineered)

        return np.array(engineered_frames)

    def reset(self):
        """Reset the state for a new sequence"""
        self.prev_features = None
        self.prev_velocity = None
        self.frame_count = 0
