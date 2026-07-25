"""
/keypoints/extract endpoint.

Accepts a single image frame from the browser and returns its 258-
feature pose+hands position vector, using the already-verified
ai_inference.pose_hand_extractor module (SmartAdaptiveExtractor +
extract_raw_features, verified byte-identical against the notebook
in Milestone 4). No keypoint math lives here -- this module owns only
the HTTP boundary: decoding the uploaded image, delegating to the
extractor, translating exceptions into HTTP responses.

Compatibility with the existing backend's routes/keypoints.py:
  - same request field name ("frame", multipart/form-data)
  - same response field names ("position", "hand_detected")
  - same HTTP status codes for the same failure modes (422 for an
    undecodable image, 500 for an unexpected extraction error)

The one unavoidable difference: "position" here has length 258, not
126 -- this backend's models expect pose+hands, not the existing
backend's hands-only pipeline. See this milestone's summary for the
frontend-side implication (POSITION_FEATURES in keypointsApi.ts is
hardcoded to 126 and would need to become model-aware; not changed
here, since frontend integration is explicitly out of scope for this
milestone).
"""

import threading

import cv2
import numpy as np
import mediapipe as mp
from fastapi import APIRouter, HTTPException, UploadFile

from ai_inference.pose_hand_extractor import SmartAdaptiveExtractor, extract_raw_features

router = APIRouter()

_holistic = None
_extractor: SmartAdaptiveExtractor | None = None
_extractor_lock = threading.Lock()


def get_holistic_and_extractor():
    """
    Lazily builds one shared Holistic instance and one shared
    SmartAdaptiveExtractor, mirroring the existing backend's
    get_extractor() singleton pattern.

    Both are held persistently, not recreated per request: Holistic's
    static_image_mode=False relies on temporal continuity between
    successive frames of the same gesture capture to track hands
    correctly across calls, and SmartAdaptiveExtractor accumulates
    hand-size history across frames to inform its adaptive
    re-detection scaling -- recreating either per frame would defeat
    both.
    """
    global _holistic, _extractor

    if _holistic is None:
        _holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    if _extractor is None:
        _extractor = SmartAdaptiveExtractor()

    return _holistic, _extractor


@router.post("/keypoints/extract")
def extract_keypoints(frame: UploadFile) -> dict:
    image_bytes = frame.file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if decoded is None:
        raise HTTPException(
            status_code=422, detail="Could not decode uploaded image."
        )

    holistic, extractor = get_holistic_and_extractor()

    try:
        # MediaPipe's Holistic.process() is not documented as thread-safe;
        # FastAPI's sync routes run in a threadpool, so concurrent requests
        # against the one shared holistic/extractor are a real possibility.
        with _extractor_lock:
            results = extractor.extract(decoded, holistic)
            position = extract_raw_features(results)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Internal keypoint extraction error."
        )

    hand_detected = (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )

    return {"position": position.tolist(), "hand_detected": hand_detected}
