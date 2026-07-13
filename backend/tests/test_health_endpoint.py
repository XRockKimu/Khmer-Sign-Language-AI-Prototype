"""
Integration tests for the /health/model endpoint (routes/health.py).

Mounts only routes.health's router into a local, test-only FastAPI
app, following the same pattern as test_predict_endpoint.py. The
model and label map singletons are monkeypatched directly to simulate
both the loaded and not-loaded states, bypassing real model/label
file loading.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ai_inference.label_loader as label_loader
import ai_inference.model_loader as model_loader
import routes.health as health_route


class _MockModel:
    def predict(self, batch, verbose=0):
        raise NotImplementedError("health check should never call predict()")


@pytest.fixture(autouse=True)
def _reset_singletons():
    model_loader._model = None
    label_loader._labels = None
    yield
    model_loader._model = None
    label_loader._labels = None


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(health_route.router)
    return TestClient(app, raise_server_exceptions=False)


def test_health_model_returns_200_when_both_loaded(client):
    model_loader._model = _MockModel()
    label_loader._labels = {f"label_{i}": i for i in range(25)}

    response = client.get("/health/model")

    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["label_map_loaded"] is True
    assert body["num_classes"] == 25


def test_health_model_returns_503_when_model_not_loaded(client):
    model_loader._model = None
    label_loader._labels = {f"label_{i}": i for i in range(25)}

    response = client.get("/health/model")

    assert response.status_code == 503
    body = response.json()
    assert body["model_loaded"] is False
    assert body["label_map_loaded"] is True


def test_health_model_returns_503_when_labels_not_loaded(client):
    model_loader._model = _MockModel()
    label_loader._labels = None

    response = client.get("/health/model")

    assert response.status_code == 503
    body = response.json()
    assert body["model_loaded"] is True
    assert body["label_map_loaded"] is False


def test_health_model_returns_503_when_neither_loaded(client):
    model_loader._model = None
    label_loader._labels = None

    response = client.get("/health/model")

    assert response.status_code == 503
    body = response.json()
    assert body["model_loaded"] is False
    assert body["label_map_loaded"] is False
    assert body["num_classes"] is None


def test_health_model_never_calls_model_predict(client):
    calls = []

    class _TrackingModel(_MockModel):
        def predict(self, batch, verbose=0):
            calls.append(batch)
            return super().predict(batch, verbose=verbose)

    model_loader._model = _TrackingModel()
    label_loader._labels = {f"label_{i}": i for i in range(25)}

    client.get("/health/model")

    assert calls == []
