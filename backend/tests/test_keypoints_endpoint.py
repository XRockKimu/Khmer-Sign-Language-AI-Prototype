"""
Integration tests for the /keypoints/extract endpoint (routes/keypoints.py).

Mounts only routes.keypoints's router into a local, test-only FastAPI app,
following the same pattern as test_predict_endpoint.py. Uses the REAL
HandKeypointExtractor (real MediaPipe, not mocked) against a synthetic
blank image -- this is intentional: a blank image has no hand in it, so
MediaPipe is expected to return no landmarks, which should surface as a
successful response with an all-zero 126-length position vector rather
than an error. This proves the HTTP boundary and the "no hand visible"
path both work, without needing a real webcam or a real hand photo.
"""

import io

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.keypoints as keypoints_route


@pytest.fixture(autouse=True)
def _reset_extractor():
    keypoints_route._extractor = None
    yield
    if keypoints_route._extractor is not None:
        keypoints_route._extractor.close()
    keypoints_route._extractor = None


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(keypoints_route.router)
    return TestClient(app, raise_server_exceptions=False)


def _blank_jpeg_bytes() -> bytes:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_extract_keypoints_returns_200_with_correct_shape(client):
    response = client.post(
        "/keypoints/extract",
        files={"frame": ("frame.jpg", io.BytesIO(_blank_jpeg_bytes()), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "position" in body
    assert len(body["position"]) == 126
    assert all(isinstance(v, float) for v in body["position"])


def test_extract_keypoints_no_hand_returns_zero_vector(client):
    response = client.post(
        "/keypoints/extract",
        files={"frame": ("frame.jpg", io.BytesIO(_blank_jpeg_bytes()), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["position"] == [0.0] * 126
    assert body["hand_detected"] is False


def test_extract_keypoints_invalid_image_returns_422(client):
    response = client.post(
        "/keypoints/extract",
        files={"frame": ("frame.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
    )

    assert response.status_code == 422


def test_extract_keypoints_reuses_shared_extractor_across_calls(client):
    assert keypoints_route._extractor is None

    client.post(
        "/keypoints/extract",
        files={"frame": ("frame.jpg", io.BytesIO(_blank_jpeg_bytes()), "image/jpeg")},
    )
    first_extractor = keypoints_route._extractor
    assert first_extractor is not None

    client.post(
        "/keypoints/extract",
        files={"frame": ("frame.jpg", io.BytesIO(_blank_jpeg_bytes()), "image/jpeg")},
    )

    assert keypoints_route._extractor is first_extractor
