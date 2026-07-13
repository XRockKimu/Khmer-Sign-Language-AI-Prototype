"""
Integration tests for the /predict endpoint (routes/predict.py).

Mounts only routes.predict's router into a local, test-only FastAPI
app -- backend/main.py's real app wiring is Day 12 scope and is
intentionally not touched here. The model and label map singletons
are monkeypatched directly, bypassing real model/label file loading.
"""

import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ai_inference.model_loader as model_loader
import ai_inference.label_loader as label_loader
import routes.predict as predict_route


class _MockModel:
    def predict(self, batch, verbose=0):
        probs = np.zeros((1, 25), dtype=np.float32)
        probs[0, 3] = 0.9
        probs[0, 0] = 0.1
        return probs


@pytest.fixture(autouse=True)
def _wire_mock_dependencies():
    model_loader._model = _MockModel()
    label_loader._labels = {f"label_{i}": i for i in range(25)}
    predict_route._inference_service = None
    yield
    model_loader._model = None
    label_loader._labels = None
    predict_route._inference_service = None


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(predict_route.router)
    return TestClient(app, raise_server_exceptions=False)


def _valid_sequence():
    return np.random.default_rng(0).random((30, 126)).tolist()


def test_predict_valid_request_returns_200(client):
    response = client.post("/predict", json={"sequence": _valid_sequence()})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class_index"] == 3
    assert body["predicted_label"] == "label_3"
    assert body["sequence_shape"] == [30, 252]
    assert len(body["top_k"]) == 3


def test_predict_wrong_row_count_returns_422(client):
    bad_sequence = np.zeros((29, 126)).tolist()
    response = client.post("/predict", json={"sequence": bad_sequence})

    assert response.status_code == 422


def test_predict_wrong_column_count_returns_422(client):
    bad_sequence = np.zeros((30, 100)).tolist()
    response = client.post("/predict", json={"sequence": bad_sequence})

    assert response.status_code == 422


def test_predict_nan_in_sequence_returns_422(client):
    bad = np.zeros((30, 126))
    bad[0, 0] = float("nan")
    raw_body = json.dumps({"sequence": bad.tolist()})
    response = client.post(
        "/predict",
        content=raw_body.encode("utf-8"),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


def test_predict_missing_sequence_field_returns_422(client):
    response = client.post("/predict", json={"wrong_key": []})

    assert response.status_code == 422


def test_predict_model_exception_returns_500(client):
    class _FailingModel:
        def predict(self, batch, verbose=0):
            raise RuntimeError("simulated TensorFlow failure")

    model_loader._model = _FailingModel()
    predict_route._inference_service = None

    response = client.post("/predict", json={"sequence": _valid_sequence()})

    assert response.status_code == 500
    assert "internal" in response.json()["detail"].lower()


def test_predict_not_loaded_returns_503(client):
    model_loader._model = None
    label_loader._labels = None
    predict_route._inference_service = None

    response = client.post("/predict", json={"sequence": _valid_sequence()})

    assert response.status_code == 503