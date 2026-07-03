import cv2
import mediapipe as mp
import numpy as np


class HandKeypointExtractor:
    """
    Extracts left and right hand landmarks using MediaPipe Holistic.

    Returns:
        left_hand  -> numpy array (63,)
        right_hand -> numpy array (63,)
    """

    HAND_FEATURES = 63

    def __init__(
        self,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_holistic = mp.solutions.holistic

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def extract_native_hand(hand_landmarks):
        """
        Convert one MediaPipe hand into a 63-dimensional feature vector.
        The wrist is used as the origin so the hand becomes translation-invariant.
        """

        if hand_landmarks is None:
            return np.zeros(
                HandKeypointExtractor.HAND_FEATURES,
                dtype=np.float32,
            )

        pts = []

        for lm in hand_landmarks.landmark:
            pts.extend([lm.x, lm.y, lm.z])

        pts = np.array(
            pts,
            dtype=np.float32,
        ).reshape(-1, 3)

        # Wrist becomes origin
        pts -= pts[0].copy()

        return pts.flatten()

    def extract(self, frame):
        """
        Extract hand landmarks from ONE OpenCV frame.

        Parameters
        ----------
        frame : numpy.ndarray (BGR)

        Returns
        -------
        dict
        {
            "left_hand": np.ndarray(63,),
            "right_hand": np.ndarray(63,),
            "results": MediaPipe Results
        }
        """

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = self.holistic.process(rgb)

        left_hand = self.extract_native_hand(
            results.left_hand_landmarks
        )

        right_hand = self.extract_native_hand(
            results.right_hand_landmarks
        )

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "results": results,
        }

    def close(self):
        """
        Release MediaPipe resources.
        """
        self.holistic.close()