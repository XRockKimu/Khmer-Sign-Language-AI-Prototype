-- 005_seed_keras3_model_versions.sql
--
-- Registers the four model artifacts backend_keras3 currently serves
-- (see ai_inference/model_registry.py's MODEL_FILENAMES) as active model
-- versions, so Milestone 12's logging has a model_versions.id to attach
-- prediction_events to for every model_id, not just "original".
--
-- All four share backend='backend_keras3' and num_classes=20 (the shared
-- 686-feature pipeline and label_map_lstm.json label map -- see
-- ai_inference/model_registry.py's own docstring). Each gets its own row
-- because idx_model_versions_active_per_model (004) enforces uniqueness
-- per (backend, model_id), not system-wide, so all four can be active at
-- once alongside "original".
--
-- Idempotent: safe to re-run against a database that already has these
-- rows.

INSERT INTO model_versions (version_label, file_path, num_classes, is_active, backend, model_id)
VALUES
    ('lstm_best_model_v1', 'models/lstm_best_model_v1.h5', 20, TRUE, 'backend_keras3', 'lstm'),
    ('gru_best_model_v2', 'models/gru_best_model_v2.h5', 20, TRUE, 'backend_keras3', 'gru'),
    ('bgru_best_model_v1', 'models/bgru_best_model_v1.h5', 20, TRUE, 'backend_keras3', 'bgru'),
    ('blstm_best_model_v1', 'models/blstm_best_model_v1.h5', 20, TRUE, 'backend_keras3', 'blstm')
ON CONFLICT (version_label) DO NOTHING;
