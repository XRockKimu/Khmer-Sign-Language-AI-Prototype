"""
Orchestrates prediction logging for routes/predict.py: creating a
prediction_sessions row and one prediction_events row for every /predict
call, success or failure, for whichever model_id served it.

This is the only module routes/predict.py calls for logging, and it is
called from exactly one place there -- _run_prediction(), shared by both
the /predict and /predict/{model_id} routes -- so every model (lstm, gru,
bgru, blstm) logs through this same code path with no per-model logging
logic duplicated anywhere.

It never lets a failure escape: log_prediction()/log_prediction_error()
catch everything -- a missing DATABASE_URL, an unreachable database, a
missing active model version, anything -- log it, and return. A broken
database must degrade prediction logging, never prediction serving.

Session model: the current frontend and /predict contract carry no
client-supplied session identifier, so there is no way to correlate
multiple /predict calls into one session yet (same constraint as
backend/services/prediction_logging_service.py). Consistent with that,
this module creates one new prediction_sessions row per logged
prediction -- a session of exactly one event.

Milestone 12: this backend shares its prediction_events/model_versions
tables with backend/ (see
database/migrations/004_multi_model_unified_logging.sql), scoped by
backend="backend_keras3" and whichever model_id served the request. Unlike
backend/services/prediction_logging_service.py, which hardcodes a single
model_id ("original"), this module caches one active model version per
model_id, since backend_keras3 serves four models concurrently and each
has its own active model_versions row.
"""

import logging
from typing import Optional

from db import prediction_repository as repository
from db.connection import get_connection

logger = logging.getLogger(__name__)

BACKEND_NAME = "backend_keras3"

_active_model_version_cache: dict = {}


def _get_active_model_version(conn, model_id: str) -> dict:
    """
    Caches the active model version per model_id for the life of the
    process, mirroring ai_inference.model_loader's load-once-and-cache
    singleton pattern. Not invalidated if the active row changes later --
    switching a model's active version currently requires restarting the
    process, matching how deploying a new model file also requires a
    restart today.
    """
    if model_id not in _active_model_version_cache:
        _active_model_version_cache[model_id] = repository.fetch_active_model_version(
            conn, backend=BACKEND_NAME, model_id=model_id
        )

    return _active_model_version_cache[model_id]


def log_prediction(
    *,
    model_id: str,
    gesture_class_index: int,
    confidence: float,
    top_k: list,
    inference_ms: float,
    client_identifier: Optional[str] = None,
) -> None:
    """
    Best-effort logging of one successful prediction for `model_id`.
    Never raises: any failure is caught and logged at ERROR level, so
    callers can invoke this directly with no try/except of their own and
    no risk to the prediction response already computed.
    """
    try:
        with get_connection() as conn:
            model_version = _get_active_model_version(conn, model_id)
            session_id = repository.create_session(conn, client_identifier)
            repository.insert_prediction_event(
                conn,
                session_id=session_id,
                model_version_id=model_version["id"],
                status="success",
                gesture_class_index=gesture_class_index,
                confidence=confidence,
                top_k=top_k,
                inference_ms=inference_ms,
            )
            conn.commit()
    except Exception as exc:
        logger.error(
            "Prediction logging failed for model_id=%s; prediction result "
            "is unaffected: %s",
            model_id,
            exc,
        )


def log_prediction_error(
    *, model_id: str, client_identifier: Optional[str] = None
) -> None:
    """
    Best-effort logging of a failed /predict/{model_id} call: a
    prediction_events row with status="error" and no predicted class,
    confidence, or top_k.

    Added so a bad-input (422) or inference-failure (500) request is
    still visible in the unified prediction log, not silently dropped --
    mirroring log_prediction()'s never-raises guarantee.
    """
    try:
        with get_connection() as conn:
            model_version = _get_active_model_version(conn, model_id)
            session_id = repository.create_session(conn, client_identifier)
            repository.insert_prediction_event(
                conn,
                session_id=session_id,
                model_version_id=model_version["id"],
                status="error",
            )
            conn.commit()
    except Exception as exc:
        logger.error(
            "Prediction error logging failed for model_id=%s; response is "
            "unaffected: %s",
            model_id,
            exc,
        )
