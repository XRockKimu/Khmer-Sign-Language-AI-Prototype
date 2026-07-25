"""
Raw SQL data-access layer for prediction logging.

Owns every query against gesture_classes, model_versions,
prediction_sessions, and prediction_events -- the same tables
backend/db/prediction_repository.py owns, shared across both backends as
of Milestone 12 (see database/migrations/004_multi_model_unified_logging.sql).
No SQL for these tables should be written anywhere outside this module;
callers (see services.prediction_logging_service) pass an open connection
in and get plain Python values back.

This is a deliberate copy of backend/db/prediction_repository.py, not an
import across backends -- the two are independent deployable services
with separate virtualenvs (see db/connection.py's docstring for the same
reasoning). The functions here are identical: nothing about logging a
prediction differs between an "original" prediction and an "lstm"/"gru"/
"bgru"/"blstm" one, only which (backend, model_id) is passed in.

None of these functions commit or roll back -- that is the caller's
decision, so multiple calls sharing one connection can be composed into a
single transaction.
"""

from typing import Optional

from psycopg2.extensions import connection as Psycopg2Connection
from psycopg2.extras import Json


def fetch_active_model_version(
    conn: Psycopg2Connection, *, backend: str, model_id: str
) -> dict:
    """
    Returns {"id": int, "version_label": str, "num_classes": int} for the
    model_versions row currently marked is_active for this (backend,
    model_id) pair.

    backend_keras3 serves four models (lstm/gru/bgru/blstm), each with its
    own active model_versions row -- callers must always pass the specific
    model_id being predicted, not assume a single system-wide active row.

    Raises RuntimeError if no model version is marked active for this
    pair -- this indicates
    database/migrations/005_seed_keras3_model_versions.sql was never
    applied, or the version has since been deactivated. Either way it is a
    configuration problem the caller should surface as a logging failure,
    not silently guess a model version for.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, version_label, num_classes
            FROM model_versions
            WHERE is_active = TRUE AND backend = %s AND model_id = %s
            LIMIT 1
            """,
            (backend, model_id),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"No active model_versions row found for backend={backend!r}, "
            f"model_id={model_id!r}."
        )

    model_version_id, version_label, num_classes = row
    return {
        "id": model_version_id,
        "version_label": version_label,
        "num_classes": num_classes,
    }


def create_session(
    conn: Psycopg2Connection, client_identifier: Optional[str] = None
) -> str:
    """
    Inserts a new prediction_sessions row and returns its id as a string.

    prediction_sessions.id is UUID, but no caller needs a real uuid.UUID
    instance -- the value is only ever round-tripped back into another
    UUID column (prediction_events.session_id).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prediction_sessions (client_identifier)
            VALUES (%s)
            RETURNING id
            """,
            (client_identifier,),
        )
        (session_id,) = cur.fetchone()

    return str(session_id)


def insert_prediction_event(
    conn: Psycopg2Connection,
    *,
    session_id: str,
    model_version_id: int,
    status: str = "success",
    gesture_class_index: Optional[int] = None,
    confidence: Optional[float] = None,
    top_k: Optional[list] = None,
    inference_ms: Optional[float] = None,
) -> int:
    """
    Inserts one prediction_events row and returns its id.

    top_k is serialized via psycopg2's Json adapter into the JSONB column
    verbatim -- callers should pass the /predict response's own top_k list
    ([{"label": str, "confidence": float}, ...]) unmodified.

    gesture_class_index/confidence/top_k/inference_ms are optional and
    status defaults to "success", so a failed prediction attempt can be
    logged as status="error" with all four left NULL -- matching the
    CHECK constraint added in
    database/migrations/004_multi_model_unified_logging.sql, which
    requires exactly that pairing.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prediction_events
                (session_id, model_version_id, status, gesture_class_index,
                 confidence, top_k, inference_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_id,
                model_version_id,
                status,
                gesture_class_index,
                confidence,
                Json(top_k) if top_k is not None else None,
                inference_ms,
            ),
        )
        (event_id,) = cur.fetchone()

    return event_id
