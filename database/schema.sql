-- schema.sql
--
-- Consolidated structural snapshot of the current database schema, kept in
-- sync with database/migrations/*.sql. This file reflects structure only
-- (tables, constraints, indexes, views) -- it does not include seed data,
-- which lives in database/seed.sql and should be applied separately after
-- this file.
--
-- Two installation paths exist, and they are not meant to be mixed:
--
--   Fresh install (new, empty database) -- this is the supported path for
--   standing up a new environment, and the only one this header documents
--   the steps for:
--     psql "$DATABASE_URL" -f database/schema.sql
--     psql "$DATABASE_URL" -f database/seed.sql
--
--   Historical upgrade (an existing pre-Milestone-12 database) -- replay
--   database/migrations/ in order instead of using this file:
--     001_initial_schema.sql -> 002_seed_gesture_classes.sql ->
--     003_seed_model_version.sql -> 004_multi_model_unified_logging.sql ->
--     005_seed_keras3_model_versions.sql ->
--     006_seed_keras3_gesture_classes.sql
--     migrations/ is kept purely as the ordered historical record of how
--     the schema reached its current shape; 002 and 003 in particular only
--     make sense in that sequence -- they insert against the
--     pre-Milestone-12 shape (a bare class_index primary key on
--     gesture_classes, and a model_versions row with no backend/model_id),
--     which stops existing the moment 004 runs. Do not run any
--     migrations/ file against a database created from this file --
--     seed.sql is their fresh-install replacement, not a supplement to
--     them.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------
-- model_versions
-- ---------------------------------------------------------------------
-- One row per trained/deployed model artifact, so every prediction_events
-- row can be traced back to exactly which model produced it.
--
-- Milestone 12: backend/model_id identify which FastAPI service and which
-- registered model (ai_inference.model_registry.MODEL_FILENAMES key, or
-- "original") a version belongs to. is_active is now scoped per
-- (backend, model_id) rather than system-wide, because backend_keras3
-- alone serves four concurrently-active models.
CREATE TABLE model_versions (
    id             BIGSERIAL PRIMARY KEY,
    version_label  TEXT NOT NULL UNIQUE,
    file_path      TEXT NOT NULL,
    num_classes    INTEGER NOT NULL CHECK (num_classes > 0),
    is_active      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    backend        TEXT NOT NULL CHECK (backend IN ('backend', 'backend_keras3')),
    model_id       TEXT NOT NULL
);

COMMENT ON TABLE model_versions IS
    'One row per trained/deployed model artifact (.h5 file), for traceability of which model produced a given prediction.';
COMMENT ON COLUMN model_versions.version_label IS
    'Human-assigned identifier for this model artifact, e.g. "best_model_25class_fix".';
COMMENT ON COLUMN model_versions.file_path IS
    'Path to the model artifact on disk, relative to its serving backend''s root.';
COMMENT ON COLUMN model_versions.backend IS
    'Which FastAPI service served this model: "backend" (original Keras 2, 25 classes) or "backend_keras3" (multi-model Keras 3).';
COMMENT ON COLUMN model_versions.model_id IS
    'Stable model identifier: matches ai_inference.model_registry.MODEL_FILENAMES keys ("lstm","gru","bgru","blstm") in backend_keras3, or "original" for backend''s sole model.';
COMMENT ON COLUMN model_versions.is_active IS
    'True while this version is the one currently serving predictions for its (backend, model_id) pair. Scoped per model, not system-wide -- enforced by idx_model_versions_active_per_model.';

-- Only one model version may be active per (backend, model_id) pair.
CREATE UNIQUE INDEX idx_model_versions_active_per_model
    ON model_versions (backend, model_id)
    WHERE is_active;

-- ---------------------------------------------------------------------
-- gesture_classes
-- ---------------------------------------------------------------------
-- One row per recognizable sign, scoped to the model_version whose label
-- map it came from. Milestone 12: class_index and label are only unique
-- *within* a model_version, not globally -- two label maps can and do
-- disagree on what class_index N or label X means (e.g. class_index 0 is
-- "No_action" for the original model but "កាតាប" for every Keras-3 model).
CREATE TABLE gesture_classes (
    model_version_id  BIGINT NOT NULL REFERENCES model_versions (id) ON DELETE RESTRICT,
    class_index       INTEGER NOT NULL,
    label             TEXT NOT NULL,
    is_active         BOOLEAN NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_version_id, class_index)
);

COMMENT ON TABLE gesture_classes IS
    'Reference list of recognizable KSL gesture classes, one row per (model_version, model output index).';
COMMENT ON COLUMN gesture_classes.model_version_id IS
    'Which model_versions row this class index/label belongs to. Class taxonomies are per-model, not global.';
COMMENT ON COLUMN gesture_classes.class_index IS
    'Model output index for this class (0..num_classes-1) within its model_version; matches predicted_class_index returned by /predict.';
COMMENT ON COLUMN gesture_classes.is_active IS
    'False once a class is retired from future training/inference, without deleting historical prediction_events rows referencing it.';

ALTER TABLE gesture_classes ADD CONSTRAINT gesture_classes_model_version_label_key
    UNIQUE (model_version_id, label);

-- ---------------------------------------------------------------------
-- prediction_sessions
-- ---------------------------------------------------------------------
-- One row per continuous demo session (from the frontend flow entering
-- HAND_DETECTED through however many capture/predict cycles it runs).
-- Groups prediction_events for session-level analytics (events per
-- session, session duration, drop-off) without tying that grouping to any
-- particular model version -- a session is about user activity, not
-- inference.
CREATE TABLE prediction_sessions (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_identifier  TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at           TIMESTAMPTZ,
    CONSTRAINT prediction_sessions_ended_after_created
        CHECK (ended_at IS NULL OR ended_at >= created_at)
);

COMMENT ON TABLE prediction_sessions IS
    'One row per continuous frontend demo session; groups prediction_events for session-level analytics.';
COMMENT ON COLUMN prediction_sessions.client_identifier IS
    'Optional opaque client-supplied identifier (e.g. a browser-generated token); no user accounts exist in this system.';
COMMENT ON COLUMN prediction_sessions.ended_at IS
    'Set when the session is closed; NULL while still in progress.';

CREATE INDEX idx_prediction_sessions_created_at ON prediction_sessions (created_at);

-- ---------------------------------------------------------------------
-- prediction_events
-- ---------------------------------------------------------------------
-- One row per individual /predict call, from any backend or model: the
-- granular prediction log this entire schema exists to support.
--
-- Milestone 12: gesture_class_index/confidence/top_k/inference_ms are
-- nullable and status was added, so a *failed* /predict call (bad input,
-- inference exception) can also be logged, with no predicted class.
CREATE TABLE prediction_events (
    id                     BIGSERIAL PRIMARY KEY,
    session_id             UUID NOT NULL REFERENCES prediction_sessions (id) ON DELETE CASCADE,
    model_version_id       BIGINT NOT NULL REFERENCES model_versions (id) ON DELETE RESTRICT,
    gesture_class_index    INTEGER,
    confidence             DOUBLE PRECISION CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    top_k                  JSONB CHECK (top_k IS NULL OR jsonb_typeof(top_k) = 'array'),
    inference_ms           DOUBLE PRECISION CHECK (inference_ms IS NULL OR inference_ms >= 0),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    status                 TEXT NOT NULL DEFAULT 'success' CHECK (status IN ('success', 'error')),
    CONSTRAINT prediction_events_model_gesture_fkey
        FOREIGN KEY (model_version_id, gesture_class_index)
        REFERENCES gesture_classes (model_version_id, class_index)
        ON DELETE RESTRICT,
    CONSTRAINT prediction_events_status_fields_check
        CHECK (
            (status = 'success' AND gesture_class_index IS NOT NULL
                AND confidence IS NOT NULL AND top_k IS NOT NULL)
            OR
            (status = 'error' AND gesture_class_index IS NULL
                AND confidence IS NULL AND top_k IS NULL)
        )
);

COMMENT ON TABLE prediction_events IS
    'One row per individual /predict call from any backend or model: predicted class, confidence, top-k predictions, inference time, status, and provenance (session + model version).';
COMMENT ON COLUMN prediction_events.session_id IS
    'The prediction_sessions row this prediction belongs to. Cascades on session delete: an event has no meaning without its parent session.';
COMMENT ON COLUMN prediction_events.model_version_id IS
    'Which model produced this prediction. RESTRICT, not CASCADE: a model_versions row must never be deleted while historical events reference it -- retire it via is_active instead.';
COMMENT ON COLUMN prediction_events.gesture_class_index IS
    'The predicted gesture class within model_version_id''s taxonomy. NULL for a failed prediction attempt (status=''error'').';
COMMENT ON COLUMN prediction_events.top_k IS
    'The API''s top_k array verbatim: [{"label": string, "confidence": number}, ...]. NULL for status=''error''.';
COMMENT ON COLUMN prediction_events.status IS
    '"success" for a completed prediction, "error" for a failed /predict call. Error rows carry no predicted class, confidence, or top_k.';

CREATE INDEX idx_prediction_events_session_id ON prediction_events (session_id);
CREATE INDEX idx_prediction_events_model_version_id ON prediction_events (model_version_id);
CREATE INDEX idx_prediction_events_gesture_class_index ON prediction_events (gesture_class_index);
CREATE INDEX idx_prediction_events_created_at ON prediction_events (created_at);
CREATE INDEX idx_prediction_events_status ON prediction_events (status);

-- ---------------------------------------------------------------------
-- prediction_log_unified
-- ---------------------------------------------------------------------
-- Milestone 12: flat, backend/model-agnostic prediction log -- one row
-- per /predict call from any backend or model, success or error, with no
-- join required by the caller.
CREATE VIEW prediction_log_unified AS
SELECT
    pe.id            AS event_id,
    pe.created_at    AS "timestamp",
    mv.backend        AS backend,
    mv.model_id        AS model_id,
    gc.label            AS predicted_label,
    pe.confidence        AS confidence,
    pe.inference_ms      AS inference_ms,
    pe.status            AS status,
    pe.session_id        AS session_id
FROM prediction_events pe
JOIN model_versions mv
    ON mv.id = pe.model_version_id
LEFT JOIN gesture_classes gc
    ON gc.model_version_id = pe.model_version_id
   AND gc.class_index = pe.gesture_class_index
ORDER BY pe.created_at DESC;
