"""
PostgreSQL connection pool for prediction logging.

Reads DATABASE_URL from the environment (via python-dotenv, consistent
with ai_inference.model_loader / ai_inference.label_loader) and exposes a
thread-safe connection pool via get_connection().

Unlike model_loader/label_loader, this module is deliberately NOT
initialized at application startup and never fails fast: /predict must
keep returning successful predictions even if the database is
unreachable or DATABASE_URL is unset, so the pool is only ever created
lazily, the first time a prediction actually needs to be logged.
"""

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from dotenv import load_dotenv
from psycopg2.extensions import connection as Psycopg2Connection
from psycopg2.pool import ThreadedConnectionPool

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

_pool: Optional[ThreadedConnectionPool] = None


def get_pool() -> ThreadedConnectionPool:
    """
    Lazily creates the shared connection pool on first use.

    Raises RuntimeError if DATABASE_URL is not set. Callers (see
    services.prediction_logging_service) are expected to catch this --
    a missing or invalid DATABASE_URL degrades prediction logging, not
    prediction serving.
    """
    global _pool

    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set; cannot connect to PostgreSQL for "
                "prediction logging."
            )
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)

    return _pool


@contextmanager
def get_connection() -> Iterator[Psycopg2Connection]:
    """
    Borrows a connection from the pool for the duration of the `with`
    block. Rolls back on any exception before returning the connection to
    the pool, so a failed request never leaves a poisoned
    idle-in-transaction connection behind for the next borrower.
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
