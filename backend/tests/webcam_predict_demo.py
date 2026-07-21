"""
Manual, visual webcam-to-prediction demo.

Wires together HandKeypointExtractor -> FrameSampleBuffer -> a live
POST /predict call, so a signed gesture in front of the webcam can be
turned into a Khmer label prediction end to end.

Not a pytest test (no assertions) -- run directly:

    python tests/webcam_predict_demo.py

Requires the backend server to already be running separately:

    uvicorn main:app --reload

Controls:
    p -- sample the last 30 accumulated frames and send them to /predict
    q -- quit
"""

import json
import urllib.error
import urllib.request

import cv2

from ai_inference.frame_sample_buffer import FrameSampleBuffer
from ai_inference.keypoint_extractor import HandKeypointExtractor

PREDICT_URL = "http://127.0.0.1:8000/predict"


def send_prediction(sequence):
    payload = json.dumps({"sequence": sequence.tolist()}).encode("utf-8")
    request = urllib.request.Request(
        PREDICT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"HTTP {exc.code} from /predict: {detail}") from exc


def main():
    extractor = HandKeypointExtractor()
    buffer = FrameSampleBuffer()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'p' to sample the last 30 frames and predict. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame")
                break

            data = extractor.extract(frame)
            buffer.add_frame(data["left_hand"], data["right_hand"])

            cv2.imshow("Webcam Predict Demo", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p"):
                try:
                    sequence = buffer.sample_sequence()
                    print("Sending sequence to /predict ...")
                    result = send_prediction(sequence)
                    print(result)
                except Exception as exc:
                    print(f"Prediction failed: {exc}")
                finally:
                    buffer.reset()
            elif key == ord("q"):
                break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
