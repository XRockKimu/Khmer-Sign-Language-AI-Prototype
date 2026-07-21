-- 003_seed_model_version.sql
--
-- Registers the model artifact currently referenced by backend/.env
-- (MODEL_PATH=models/best_model_25class_fix.h5) as the active model
-- version. Without this row, Day 20's prediction logging would have no
-- model_versions.id to attach prediction_events to.
--
-- Idempotent: safe to re-run against a database that already has this row.

INSERT INTO model_versions (version_label, file_path, num_classes, is_active)
VALUES ('best_model_25class_fix', 'models/best_model_25class_fix.h5', 25, TRUE)
ON CONFLICT (version_label) DO NOTHING;
