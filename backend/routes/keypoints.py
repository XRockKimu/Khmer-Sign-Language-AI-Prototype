"""
/keypoints/extract endpoint.

Accepts a single image frame from the browser and returns its 126-feature
hand-position vector (left hand + right hand, wrist-relative), using the
existing, unmodified ai_inference.keypoint_extractor.HandKeypointExtractor.
This is the missing link that lets the frontend build a REAL (30, 126)
sequence from the user's actual camera feed, one frame at a time, instead
of a mock sequence -- the resulting sequence is submitted to the existing
/predict endpoint exactly as before. /predict's request/response contract
is not touched by this module at all.

This module owns only the HTTP boundary: decoding the uploaded image,
delegating to HandKeypointExtractor, and translating exceptions into HTTP
responses. No keypoint math lives here.
"""

import threading

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile

from ai_inference.keypoint_extractor import HandKeypointExtractor

router = APIRouter()

_extractor: HandKeypointExtractor | None = None
_extractor_lock = threading.Lock()


def get_extractor() -> HandKeypointExtractor:
    """
    Lazily builds one shared HandKeypointExtractor, mirroring
    routes.predict's InferenceService singleton pattern.

    A single persistent instance is used deliberately, not a fresh one per
    request: HandKeypointExtractor defaults to static_image_mode=False,
    which relies on temporal continuity between successive frames of the
    same gesture capture to track hands correctly across calls --
    recreating the extractor per frame would defeat that.
    """
    global _extractor

    if _extractor is None:
        _extractor = HandKeypointExtractor()

    return _extractor


@router.post("/keypoints/extract")
def extract_keypoints(frame: UploadFile) -> dict:
    image_bytes = frame.file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if decoded is None:
        raise HTTPException(
            status_code=422, detail="Could not decode uploaded image."
        )

    extractor = get_extractor()

    try:
        # MediaPipe's Holistic.process() is not documented as thread-safe;
        # FastAPI's sync routes run in a threadpool, so concurrent requests
        # against the one shared extractor are a real possibility.
        with _extractor_lock:
            result = extractor.extract(decoded)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Internal keypoint extraction error."
        )

    # Order matches frame_sample_buffer.FrameSampleBuffer.add_frame's
    # combined = np.concatenate([left_hand, right_hand]).
    position = np.concatenate([result["left_hand"], result["right_hand"]])

    mediapipe_results = result["results"]
    hand_detected = (
        mediapipe_results.left_hand_landmarks is not None
        or mediapipe_results.right_hand_landmarks is not None
    )

    return {"position": position.tolist(), "hand_detected": hand_detected}
