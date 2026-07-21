"""
Orchestrates prediction logging for routes/predict.py: creating a
prediction_sessions row and one prediction_events row for every
successful prediction.

This is the only module routes/predict.py calls for logging. It never
lets a failure escape: log_prediction() catches everything -- a missing
DATABASE_URL, an unreachable database, a missing active model version,
anything -- logs it, and returns. A broken database must degrade
prediction logging, never prediction serving.

Session model: the current frontend and /predict contract carry no
client-supplied session identifier (per Day 20's requirements, neither may
change), so there is no way to correlate multiple /predict calls into one
session yet. Consistent with that constraint, this module creates one new
prediction_sessions row per logged prediction -- a session of exactly one
event. Correlating a browser's full demo session into a single
prediction_sessions row is a natural follow-up once the frontend or API
contract gains an actual session concept, but is out of scope here.
"""

import logging
from typing import Optional

from db import prediction_repository as repository
from db.connection import get_connection

logger = logging.getLogger(__name__)

_active_model_version_cache: Optional[dict] = None


def _get_active_model_version(conn) -> dict:
    """
    Caches the active model version for the life of the process, mirroring
    ai_inference.model_loader / ai_inference.label_loader's own
    load-once-and-cache singleton pattern. Not invalidated if the active
    row changes later -- switching the active model currently requires
    restarting the process, matching how deploying a new model file also
    requires a restart today.
    """
    global _active_model_version_cache

    if _active_model_version_cache is None:
        _active_model_version_cache = repository.fetch_active_model_version(conn)

    return _active_model_version_cache


def log_prediction(
    *,
    gesture_class_index: int,
    confidence: float,
    top_k: list,
    inference_ms: float,
    client_identifier: Optional[str] = None,
) -> None:
    """
    Best-effort logging of one successful prediction. Never raises: any
    failure is caught and logged at ERROR level, so callers can invoke
    this directly with no try/except of their own and no risk to the
    prediction response already computed.
    """
    try:
        with get_connection() as conn:
            model_version = _get_active_model_version(conn)
            session_id = repository.create_session(conn, client_identifier)
            repository.insert_prediction_event(
                conn,
                session_id=session_id,
                model_version_id=model_version["id"],
                gesture_class_index=gesture_class_index,
                confidence=confidence,
                top_k=top_k,
                inference_ms=inference_ms,
            )
            conn.commit()
    except Exception as exc:
        logger.error(
            "Prediction logging failed; prediction result is unaffected: %s", exc
        )
