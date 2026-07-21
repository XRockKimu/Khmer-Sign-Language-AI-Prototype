"""
Unit tests for services.prediction_logging_service.

The database layer (db.connection.get_connection, db.prediction_repository)
is monkeypatched directly, following the same direct-attribute-assignment
pattern used by test_predict_endpoint.py and test_health_endpoint.py --
no real PostgreSQL connection is used. These tests verify the one
requirement Day 20 cares about most: log_prediction() must never raise,
regardless of what fails underneath it.
"""

from contextlib import contextmanager

import pytest

import services.prediction_logging_service as prediction_logging_service
from db import prediction_repository as repository


class _FakeConnection:
    def __init__(self):
        self.committed = False

    def commit(self):
        self.committed = True


@pytest.fixture(autouse=True)
def _reset_cache():
    prediction_logging_service._active_model_version_cache = None
    yield
    prediction_logging_service._active_model_version_cache = None


@contextmanager
def _fake_get_connection(conn):
    yield conn


def test_log_prediction_happy_path_calls_repository_and_commits(monkeypatch):
    conn = _FakeConnection()
    calls = {}

    monkeypatch.setattr(
        prediction_logging_service, "get_connection", lambda: _fake_get_connection(conn)
    )
    monkeypatch.setattr(
        repository,
        "fetch_active_model_version",
        lambda c: {"id": 1, "version_label": "v1", "num_classes": 25},
    )

    def _create_session(c, client_identifier=None):
        calls["client_identifier"] = client_identifier
        return "session-123"

    def _insert_event(c, **kwargs):
        calls["event_kwargs"] = kwargs
        return 99

    monkeypatch.setattr(repository, "create_session", _create_session)
    monkeypatch.setattr(repository, "insert_prediction_event", _insert_event)

    prediction_logging_service.log_prediction(
        gesture_class_index=3,
        confidence=0.9,
        top_k=[{"label": "ក", "confidence": 0.9}],
        inference_ms=12.5,
    )

    assert conn.committed is True
    assert calls["event_kwargs"]["session_id"] == "session-123"
    assert calls["event_kwargs"]["model_version_id"] == 1
    assert calls["event_kwargs"]["gesture_class_index"] == 3
    assert calls["event_kwargs"]["confidence"] == 0.9
    assert calls["event_kwargs"]["inference_ms"] == 12.5


def test_log_prediction_caches_active_model_version(monkeypatch):
    conn = _FakeConnection()
    fetch_calls = []

    monkeypatch.setattr(
        prediction_logging_service, "get_connection", lambda: _fake_get_connection(conn)
    )

    def _fetch(c):
        fetch_calls.append(c)
        return {"id": 1, "version_label": "v1", "num_classes": 25}

    monkeypatch.setattr(repository, "fetch_active_model_version", _fetch)
    monkeypatch.setattr(repository, "create_session", lambda c, client_identifier=None: "s")
    monkeypatch.setattr(repository, "insert_prediction_event", lambda c, **kw: 1)

    for _ in range(3):
        prediction_logging_service.log_prediction(
            gesture_class_index=0,
            confidence=1.0,
            top_k=[],
            inference_ms=1.0,
        )

    assert len(fetch_calls) == 1


def test_log_prediction_swallows_connection_failure(monkeypatch):
    def _raise_connection_error():
        raise RuntimeError("DATABASE_URL is not set")

    monkeypatch.setattr(prediction_logging_service, "get_connection", _raise_connection_error)

    # Must not raise.
    prediction_logging_service.log_prediction(
        gesture_class_index=0,
        confidence=1.0,
        top_k=[],
        inference_ms=1.0,
    )


def test_log_prediction_swallows_repository_failure(monkeypatch):
    conn = _FakeConnection()

    monkeypatch.setattr(
        prediction_logging_service, "get_connection", lambda: _fake_get_connection(conn)
    )

    def _raise(*args, **kwargs):
        raise RuntimeError("no active model_versions row")

    monkeypatch.setattr(repository, "fetch_active_model_version", _raise)

    # Must not raise, and must not commit a half-finished transaction.
    prediction_logging_service.log_prediction(
        gesture_class_index=0,
        confidence=1.0,
        top_k=[],
        inference_ms=1.0,
    )

    assert conn.committed is False
