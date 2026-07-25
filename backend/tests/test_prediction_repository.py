"""
Unit tests for db.prediction_repository.

No real PostgreSQL connection is used -- a fake cursor/connection records
the SQL and parameters it was called with, following the same
"recording mock" pattern as test_inference_service.py's _RecordingModel.
These tests verify the SQL this module issues and how it maps Python
values to columns, not real database execution.
"""

import pytest
from psycopg2.extras import Json

from db import prediction_repository as repository


class _RecordingCursor:
    def __init__(self, fetchone_result):
        self._fetchone_result = fetchone_result
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._fetchone_result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _RecordingConnection:
    def __init__(self, fetchone_result):
        self._cursor = _RecordingCursor(fetchone_result)

    def cursor(self):
        return self._cursor


def test_fetch_active_model_version_returns_mapped_dict():
    conn = _RecordingConnection(fetchone_result=(7, "best_model_25class_fix", 25))

    result = repository.fetch_active_model_version(
        conn, backend="backend", model_id="original"
    )

    assert result == {
        "id": 7,
        "version_label": "best_model_25class_fix",
        "num_classes": 25,
    }
    sql, params = conn._cursor.executed[0]
    assert "model_versions" in sql
    assert "is_active" in sql
    assert params == ("backend", "original")


def test_fetch_active_model_version_raises_when_none_active():
    conn = _RecordingConnection(fetchone_result=None)

    with pytest.raises(RuntimeError):
        repository.fetch_active_model_version(
            conn, backend="backend", model_id="original"
        )


def test_create_session_returns_id_as_string():
    conn = _RecordingConnection(fetchone_result=("11111111-1111-1111-1111-111111111111",))

    session_id = repository.create_session(conn, client_identifier="browser-abc")

    assert session_id == "11111111-1111-1111-1111-111111111111"
    sql, params = conn._cursor.executed[0]
    assert "prediction_sessions" in sql
    assert params == ("browser-abc",)


def test_create_session_defaults_client_identifier_to_none():
    conn = _RecordingConnection(fetchone_result=("22222222-2222-2222-2222-222222222222",))

    repository.create_session(conn)

    _, params = conn._cursor.executed[0]
    assert params == (None,)


def test_insert_prediction_event_passes_correct_params_and_serializes_top_k():
    conn = _RecordingConnection(fetchone_result=(42,))
    top_k = [{"label": "ក", "confidence": 0.9}, {"label": "ខ", "confidence": 0.05}]

    event_id = repository.insert_prediction_event(
        conn,
        session_id="11111111-1111-1111-1111-111111111111",
        model_version_id=1,
        gesture_class_index=3,
        confidence=0.9,
        top_k=top_k,
        inference_ms=12.5,
    )

    assert event_id == 42
    sql, params = conn._cursor.executed[0]
    assert "prediction_events" in sql
    (
        session_id_param,
        model_version_id_param,
        status_param,
        gesture_class_index_param,
        confidence_param,
        top_k_param,
        inference_ms_param,
    ) = params
    assert session_id_param == "11111111-1111-1111-1111-111111111111"
    assert model_version_id_param == 1
    assert status_param == "success"
    assert gesture_class_index_param == 3
    assert confidence_param == 0.9
    assert isinstance(top_k_param, Json)
    assert inference_ms_param == 12.5


def test_insert_prediction_event_error_status_leaves_prediction_fields_null():
    conn = _RecordingConnection(fetchone_result=(43,))

    event_id = repository.insert_prediction_event(
        conn,
        session_id="11111111-1111-1111-1111-111111111111",
        model_version_id=1,
        status="error",
    )

    assert event_id == 43
    _, params = conn._cursor.executed[0]
    (
        _session_id,
        _model_version_id,
        status_param,
        gesture_class_index_param,
        confidence_param,
        top_k_param,
        inference_ms_param,
    ) = params
    assert status_param == "error"
    assert gesture_class_index_param is None
    assert confidence_param is None
    assert top_k_param is None
    assert inference_ms_param is None
