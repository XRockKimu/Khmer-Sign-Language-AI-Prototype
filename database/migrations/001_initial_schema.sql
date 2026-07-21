-- 001_initial_schema.sql
--
-- Initial schema for the KSL prediction system: gesture classes, model
-- versions, prediction sessions, and prediction events. This migration
-- creates structure only -- no seed data (see 002 and 003) and no
-- application logic. Inserting into or reading from these tables is Day 20
-- scope, not this migration's.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------
-- gesture_classes
-- ---------------------------------------------------------------------
-- One row per recognizable sign, sourced from the model's label map
-- (backend/labels/label_map_25class.json). class_index is the model's own
-- output index for this class, not a synthetic surrogate key: it is fixed
-- by the frozen model/feature-formatter contract (see
-- backend/ai_inference/feature_formatter.py) and is exactly the value the
-- backend returns as predicted_class_index, so using it directly as the
-- primary key avoids an unnecessary indirection layer.
CREATE TABLE gesture_classes (
    class_index   INTEGER PRIMARY KEY,
    label         TEXT NOT NULL UNIQUE,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE gesture_classes IS
    'Static reference list of recognizable KSL gesture classes, one row per model output index.';
COMMENT ON COLUMN gesture_classes.class_index IS
    'Model output index for this class (0..num_classes-1); matches predicted_class_index returned by /predict.';
COMMENT ON COLUMN gesture_classes.is_active IS
    'False once a class is retired from future training/inference, without deleting historical prediction_events rows referencing it.';

-- ---------------------------------------------------------------------
-- model_versions
-- ---------------------------------------------------------------------
-- One row per trained/deployed model artifact, so every prediction_events
-- row can be traced back to exactly which model produced it.
CREATE TABLE model_versions (
    id             BIGSERIAL PRIMARY KEY,
    version_label  TEXT NOT NULL UNIQUE,
    file_path      TEXT NOT NULL,
    num_classes    INTEGER NOT NULL CHECK (num_classes > 0),
    is_active      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE model_versions IS
    'One row per trained/deployed model artifact (.h5 file), for traceability of which model produced a given prediction.';
COMMENT ON COLUMN model_versions.version_label IS
    'Human-assigned identifier for this model artifact, e.g. "best_model_25class_fix".';
COMMENT ON COLUMN model_versions.file_path IS
    'Path to the model artifact on disk, matching MODEL_PATH in backend/.env at the time this version was registered.';
COMMENT ON COLUMN model_versions.is_active IS
    'True for the single model version currently serving predictions; enforced unique by idx_model_versions_single_active.';

-- Only one model version may be active at a time.
CREATE UNIQUE INDEX idx_model_versions_single_active
    ON model_versions (is_active)
    WHERE is_active;

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
    'Set when the session is closed; NULL while still in progress. Populated by Day 20 application logic, not this schema.';

CREATE INDEX idx_prediction_sessions_created_at ON prediction_sessions (created_at);

-- ---------------------------------------------------------------------
-- prediction_events
-- ---------------------------------------------------------------------
-- One row per individual /predict call: the granular prediction log this
-- entire schema exists to support. Not populated by any code yet --
-- inserting into this table is Day 20 scope.
CREATE TABLE prediction_events (
    id                     BIGSERIAL PRIMARY KEY,
    session_id             UUID NOT NULL REFERENCES prediction_sessions (id) ON DELETE CASCADE,
    model_version_id       BIGINT NOT NULL REFERENCES model_versions (id) ON DELETE RESTRICT,
    gesture_class_index    INTEGER NOT NULL REFERENCES gesture_classes (class_index) ON DELETE RESTRICT,
    confidence             DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    top_k                  JSONB NOT NULL CHECK (jsonb_typeof(top_k) = 'array'),
    inference_ms           DOUBLE PRECISION NOT NULL CHECK (inference_ms >= 0),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE prediction_events IS
    'One row per individual /predict call: predicted class, confidence, top-k predictions, inference time, and provenance (session + model version).';
COMMENT ON COLUMN prediction_events.session_id IS
    'The prediction_sessions row this prediction belongs to. Cascades on session delete: an event has no meaning without its parent session.';
COMMENT ON COLUMN prediction_events.model_version_id IS
    'Which model produced this prediction. RESTRICT, not CASCADE: a model_versions row must never be deleted while historical events reference it -- retire it via is_active instead.';
COMMENT ON COLUMN prediction_events.gesture_class_index IS
    'The predicted gesture class. RESTRICT for the same reason as model_version_id: retire a class via gesture_classes.is_active, never delete it out from under prediction history.';
COMMENT ON COLUMN prediction_events.top_k IS
    'The API''s top_k array verbatim: [{"label": string, "confidence": number}, ...].';

CREATE INDEX idx_prediction_events_session_id ON prediction_events (session_id);
CREATE INDEX idx_prediction_events_model_version_id ON prediction_events (model_version_id);
CREATE INDEX idx_prediction_events_gesture_class_index ON prediction_events (gesture_class_index);
CREATE INDEX idx_prediction_events_created_at ON prediction_events (created_at);
