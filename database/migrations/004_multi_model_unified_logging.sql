-- 004_multi_model_unified_logging.sql
--
-- Milestone 12: backend_keras3 now serves four concurrently-active models
-- (lstm/gru/bgru/blstm) alongside this system's original 25-class model --
-- five active models at once, not one. Two assumptions baked into
-- 001_initial_schema.sql no longer hold:
--
--   1. idx_model_versions_single_active enforced exactly one active
--      model_versions row *system-wide*. That was correct when only one
--      backend/model existed; it must now allow one active version *per
--      model*, since five models are legitimately active simultaneously.
--
--   2. gesture_classes used class_index as a global primary key and label
--      as a global unique key, on the assumption that "class index N" and
--      "label X" mean the same gesture everywhere. That assumption breaks
--      the moment a second label map exists: class_index 0 is "No_action"
--      in the original 25-class map but "កាតាប" in the shared 20-class
--      Keras-3 label map (backend_keras3/labels/label_map_lstm.json), and
--      at least one label ("ប៊ិកក្រហម") appears in both maps at different
--      indices. Classes must be scoped per model_version, not global.
--
-- prediction_events also could not represent a *failed* prediction attempt
-- at all -- gesture_class_index/confidence/top_k were all NOT NULL, so
-- there was no way to store "status" for a request that never produced a
-- predicted class. Milestone 12 requires recording success/error status
-- for every backend, so those columns become nullable and a status column
-- is added.
--
-- backend/model_id are added to model_versions, not duplicated onto every
-- prediction_events row: they describe *which model produced the
-- prediction*, a fact of the model version, not of the individual event.
-- Making them event-level columns would just denormalize this same fact
-- across 180+ existing rows (and every row after). The prediction_log_unified
-- view at the bottom exists precisely so callers never have to hand-write
-- that join to get backend/model_id/predicted label/confidence/inference_ms/
-- timestamp/status in one flat query.

-- ---------------------------------------------------------------------
-- model_versions: identify backend + stable model_id; allow one active
-- version per model instead of one active version system-wide.
-- ---------------------------------------------------------------------
ALTER TABLE model_versions ADD COLUMN backend TEXT;
ALTER TABLE model_versions ADD COLUMN model_id TEXT;

UPDATE model_versions
SET backend = 'backend', model_id = 'original'
WHERE version_label = 'best_model_25class_fix';

ALTER TABLE model_versions ALTER COLUMN backend SET NOT NULL;
ALTER TABLE model_versions ALTER COLUMN model_id SET NOT NULL;

ALTER TABLE model_versions ADD CONSTRAINT model_versions_backend_check
    CHECK (backend IN ('backend', 'backend_keras3'));

DROP INDEX idx_model_versions_single_active;

CREATE UNIQUE INDEX idx_model_versions_active_per_model
    ON model_versions (backend, model_id)
    WHERE is_active;

COMMENT ON COLUMN model_versions.backend IS
    'Which FastAPI service served this model: "backend" (original Keras 2, 25 classes) or "backend_keras3" (multi-model Keras 3).';
COMMENT ON COLUMN model_versions.model_id IS
    'Stable model identifier: matches ai_inference.model_registry.MODEL_FILENAMES keys ("lstm","gru","bgru","blstm") in backend_keras3, or "original" for backend''s sole model.';
COMMENT ON COLUMN model_versions.is_active IS
    'True while this version is the one currently serving predictions for its (backend, model_id) pair. Milestone 12: scoped per model, not system-wide -- enforced by idx_model_versions_active_per_model, not a single global flag.';

-- ---------------------------------------------------------------------
-- prediction_events: drop its FK into gesture_classes first -- it
-- depends on gesture_classes_pkey, which the next section replaces.
-- ---------------------------------------------------------------------
ALTER TABLE prediction_events
    DROP CONSTRAINT prediction_events_gesture_class_index_fkey;

-- ---------------------------------------------------------------------
-- gesture_classes: scope class_index/label per model_version, since two
-- label maps can (and do) disagree on what index/label N means.
-- ---------------------------------------------------------------------
ALTER TABLE gesture_classes ADD COLUMN model_version_id BIGINT
    REFERENCES model_versions (id) ON DELETE RESTRICT;

UPDATE gesture_classes
SET model_version_id = (SELECT id FROM model_versions WHERE model_id = 'original');

ALTER TABLE gesture_classes ALTER COLUMN model_version_id SET NOT NULL;

ALTER TABLE gesture_classes DROP CONSTRAINT gesture_classes_pkey;
ALTER TABLE gesture_classes ADD PRIMARY KEY (model_version_id, class_index);

ALTER TABLE gesture_classes DROP CONSTRAINT gesture_classes_label_key;
ALTER TABLE gesture_classes ADD CONSTRAINT gesture_classes_model_version_label_key
    UNIQUE (model_version_id, label);

COMMENT ON COLUMN gesture_classes.model_version_id IS
    'Which model_versions row this class index/label belongs to. Milestone 12: class taxonomies are per-model, not global -- class_index and label are now only unique within a model_version, not across the whole table.';

-- ---------------------------------------------------------------------
-- prediction_events: re-add the FK through the new composite key, and
-- allow a row to represent a failed (non-predicting) attempt.
-- ---------------------------------------------------------------------
ALTER TABLE prediction_events ALTER COLUMN gesture_class_index DROP NOT NULL;
ALTER TABLE prediction_events ALTER COLUMN confidence DROP NOT NULL;
ALTER TABLE prediction_events ALTER COLUMN top_k DROP NOT NULL;
ALTER TABLE prediction_events ALTER COLUMN inference_ms DROP NOT NULL;

ALTER TABLE prediction_events
    ADD CONSTRAINT prediction_events_model_gesture_fkey
    FOREIGN KEY (model_version_id, gesture_class_index)
    REFERENCES gesture_classes (model_version_id, class_index)
    ON DELETE RESTRICT;

ALTER TABLE prediction_events ADD COLUMN status TEXT NOT NULL DEFAULT 'success';

ALTER TABLE prediction_events
    ADD CONSTRAINT prediction_events_status_check
    CHECK (status IN ('success', 'error'));

ALTER TABLE prediction_events
    ADD CONSTRAINT prediction_events_status_fields_check
    CHECK (
        (status = 'success' AND gesture_class_index IS NOT NULL
            AND confidence IS NOT NULL AND top_k IS NOT NULL)
        OR
        (status = 'error' AND gesture_class_index IS NULL
            AND confidence IS NULL AND top_k IS NULL)
    );

COMMENT ON COLUMN prediction_events.status IS
    '"success" for a completed prediction, "error" for a failed /predict call (bad input, model not loaded, inference exception). Error rows carry no predicted class, confidence, or top_k.';

CREATE INDEX idx_prediction_events_status ON prediction_events (status);

-- ---------------------------------------------------------------------
-- Unified read-time view: one row per logged prediction across every
-- backend/model, in the flat shape Milestone 12 asks for -- no caller
-- needs to know about sessions/model_versions/gesture_classes to answer
-- "every prediction, which backend, which model, what label, how
-- confident, how long, when, success or error."
-- ---------------------------------------------------------------------
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

COMMENT ON VIEW prediction_log_unified IS
    'Milestone 12: flat, backend/model-agnostic prediction log -- one row per /predict call from any backend or model, success or error, with no join required by the caller.';
