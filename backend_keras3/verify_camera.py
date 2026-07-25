"""
Milestone 4 verification script -- manual and interactive, requires a
real webcam. Unlike verify_setup.py (fully offline, synthetic data,
no hardware dependency), this script cannot be run unattended: it
opens a real camera and expects a person to perform a gesture in
front of it while it captures.

Pipeline exercised end to end on real captured data:

    webcam frame
      -> SmartAdaptiveExtractor.extract() + extract_raw_features()   (258,)
      -> accumulate 30 frames                                        (30, 258)
      -> feature_formatter.format_sequence()                         (30, 686)
      -> np.expand_dims(..., axis=0)                                 (1, 30, 686)
      -> model.predict()                                             (1, 20)

Still not a FastAPI app, no frame-buffer class (accumulation here is
a plain list, matching the "no frame buffering yet" scope of this
milestone), no logging, no routes.

Run directly:

    python verify_camera.py
"""

import cv2
import numpy as np
import mediapipe as mp

from ai_inference import model_loader, label_loader, feature_formatter
from ai_inference.pose_hand_extractor import SmartAdaptiveExtractor, extract_raw_features

SEQUENCE_LENGTH = 30
CAMERA_INDEX = 0


def main():
    # This script verifies the LSTM model specifically, unchanged in
    # purpose since Milestone 4 -- model_loader.load_model() now takes
    # a model_id (Milestone 9, multi-model), so this is passed
    # explicitly rather than relying on a since-removed default.
    model = model_loader.load_model("lstm")
    labels = label_loader.load_labels()

    print()
    print("=" * 50)
    print("OPENING CAMERA")
    print("=" * 50)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {CAMERA_INDEX}.")
        print("Likely causes: no camera attached, camera in use by another")
        print("application, or (on some systems) a missing camera driver/permission.")
        return

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    extractor = SmartAdaptiveExtractor()

    print(f"Camera opened. Capturing {SEQUENCE_LENGTH} frames -- perform a gesture now.")
    print()

    frames = []
    try:
        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read a frame from the camera mid-capture.")
                break

            results = extractor.extract(frame, holistic)
            position = extract_raw_features(results)

            if position.shape != (258,):
                print(f"[ERROR] Unexpected extractor output shape: {position.shape}")
                break

            frames.append(position)
            print(f"  frame {len(frames):2d}/{SEQUENCE_LENGTH}  "
                  f"position shape={position.shape}  "
                  f"hand_detected={bool(results.left_hand_landmarks or results.right_hand_landmarks)}")
    finally:
        cap.release()
        holistic.close()

    if len(frames) < SEQUENCE_LENGTH:
        print()
        print(f"[ERROR] Only captured {len(frames)}/{SEQUENCE_LENGTH} frames -- stopping.")
        return

    print()
    print("=" * 50)
    print("PIPELINE STAGE SHAPES (real captured data)")
    print("=" * 50)

    sequence_258 = np.array(frames, dtype=np.float32)
    print(f"1. Accumulated positions : {sequence_258.shape} (expected (30, 258))")

    sequence_686 = feature_formatter.format_sequence(sequence_258)
    print(f"2. Engineered features   : {sequence_686.shape} (expected (30, 686))")

    model_input = np.expand_dims(sequence_686, axis=0)
    print(f"3. Model input           : {model_input.shape} (expected (1, 30, 686))")

    print()
    print("=" * 50)
    print("RUNNING PREDICTION")
    print("=" * 50)
    prediction = model.predict(model_input, verbose=0)
    predicted_index = int(np.argmax(prediction[0]))
    predicted_label = labels[predicted_index]
    confidence = float(prediction[0][predicted_index])

    print(f"Predicted label: {predicted_label}")
    print(f"Confidence     : {confidence:.4f}")


if __name__ == "__main__":
    main()
