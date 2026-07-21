"""
Raw SQL data-access layer for prediction logging.

Owns every query against gesture_classes, model_versions,
prediction_sessions, and prediction_events -- the tables created in Day 19
(database/migrations/001_initial_schema.sql). No SQL for these tables
should be written anywhere outside this module; callers (see
services.prediction_logging_service) pass an open connection in and get
plain Python values back.

None of these functions commit or roll back -- that is the caller's
decision, so multiple calls sharing one connection can be composed into a
single transaction.
"""

from typing import Optional

from psycopg2.extensions import connection as Psycopg2Connection
from psycopg2.extras import Json


def fetch_active_model_version(conn: Psycopg2Connection) -> dict:
    """
    Returns {"id": int, "version_label": str, "num_classes": int} for the
    single model_versions row currently marked is_active.

    Raises RuntimeError if no model version is marked active -- this
    indicates database/migrations/003_seed_model_version.sql was never
    applied, or every version has since been deactivated. Either way it
    is a configuration problem the caller should surface as a logging
    failure, not silently guess a model version for.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, version_label, num_classes
            FROM model_versions
            WHERE is_active = TRUE
            LIMIT 1
            """
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            "No active model_versions row found. Apply "
            "database/migrations/003_seed_model_version.sql or mark a "
            "model version active."
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
    gesture_class_index: int,
    confidence: float,
    top_k: list,
    inference_ms: float,
) -> int:
    """
    Inserts one prediction_events row and returns its id.

    top_k is serialized via psycopg2's Json adapter into the JSONB column
    verbatim -- callers should pass the /predict response's own top_k list
    ([{"label": str, "confidence": float}, ...]) unmodified.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prediction_events
                (session_id, model_version_id, gesture_class_index,
                 confidence, top_k, inference_ms)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_id,
                model_version_id,
                gesture_class_index,
                confidence,
                Json(top_k),
                inference_ms,
            ),
        )
        (event_id,) = cur.fetchone()

    return event_id
