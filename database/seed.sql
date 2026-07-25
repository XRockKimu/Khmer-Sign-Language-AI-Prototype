-- seed.sql
--
-- Full initial data set for a fresh Milestone-12 database: every active
-- model version this system serves, and the gesture-class label map for
-- each. Run once, immediately after schema.sql, against a newly created
-- database -- see schema.sql's header for the two supported installation
-- paths.
--
-- This is a snapshot of the *current* data, not a replay of history: it
-- supersedes 002_seed_gesture_classes.sql, 003_seed_model_version.sql,
-- 005_seed_keras3_model_versions.sql, and 006_seed_keras3_gesture_classes.sql
-- for fresh installs the same way schema.sql supersedes 001/004 for
-- structure. Those migrations remain the authoritative record for
-- upgrading an existing pre-Milestone-12 database and are not used
-- together with this file.
--
-- Sourced from:
--   backend/.env (MODEL_PATH)                        -> the original model
--   backend/labels/label_map_25class.json             -> its label map
--   backend_keras3/ai_inference/model_registry.py     -> the four Keras-3 models
--   backend_keras3/labels/label_map_lstm.json          -> their shared label map
--
-- Idempotent: safe to re-run against a database that already has this data.

-- ---------------------------------------------------------------------
-- model_versions: one active row per model this system serves -- the
-- original 25-class model (backend), and the four Keras-3 models
-- (backend_keras3), all active simultaneously per
-- idx_model_versions_active_per_model.
-- ---------------------------------------------------------------------
INSERT INTO model_versions (version_label, file_path, num_classes, is_active, backend, model_id)
VALUES
    ('best_model_25class_fix', 'models/best_model_25class_fix.h5', 25, TRUE, 'backend',       'original'),
    ('lstm_best_model_v1',     'models/lstm_best_model_v1.h5',     20, TRUE, 'backend_keras3', 'lstm'),
    ('gru_best_model_v2',      'models/gru_best_model_v2.h5',      20, TRUE, 'backend_keras3', 'gru'),
    ('bgru_best_model_v1',     'models/bgru_best_model_v1.h5',     20, TRUE, 'backend_keras3', 'bgru'),
    ('blstm_best_model_v1',    'models/blstm_best_model_v1.h5',    20, TRUE, 'backend_keras3', 'blstm')
ON CONFLICT (version_label) DO NOTHING;

-- ---------------------------------------------------------------------
-- gesture_classes: the "original" model's 25-class label map
-- (label_map_25class.json), scoped to its model_version.
-- ---------------------------------------------------------------------
INSERT INTO gesture_classes (model_version_id, class_index, label)
SELECT mv.id, c.class_index, c.label
FROM model_versions mv
CROSS JOIN (VALUES
    (0, 'No_action'),
    (1, 'កាតាប'),
    (2, 'កាតាបស្ពាយក្រោយ'),
    (3, 'កុំព្យូទ័រ'),
    (4, 'កៅអី'),
    (5, 'ក្ដារខៀន'),
    (6, 'ខ្មៅដៃ'),
    (7, 'ជ័រលុប'),
    (8, 'ដីស'),
    (9, 'តុ'),
    (10, 'ទឹកលុប'),
    (11, 'នាយករង'),
    (12, 'នាយិកា'),
    (13, 'បន្ទាត់'),
    (14, 'ប៊ិក'),
    (15, 'ប៊ិកក្រហម'),
    (16, 'ប៊ិកខៀវ'),
    (17, 'លោកគ្រូ'),
    (18, 'សាលារៀន'),
    (19, 'សៀវភៅ'),
    (20, 'សៀវភៅពុម្ភ'),
    (21, 'ហ្វឺតក្រហម'),
    (22, 'ហ្វឺតខៀវ'),
    (23, 'ហ្វឺតខ្មៅ'),
    (24, 'អ្នកគ្រូ')
) AS c(class_index, label)
WHERE mv.backend = 'backend' AND mv.model_id = 'original'
ON CONFLICT (model_version_id, class_index) DO NOTHING;

-- ---------------------------------------------------------------------
-- gesture_classes: backend_keras3's shared 20-class label map
-- (label_map_lstm.json), scoped to each of its four model versions. One
-- row per (model_version, class_index) -- class_index 0 means "កាតាប" for
-- all four models, but each gets its own row, mirroring how the models
-- themselves are separate model_versions rows despite sharing a label map.
-- ---------------------------------------------------------------------
INSERT INTO gesture_classes (model_version_id, class_index, label)
SELECT mv.id, c.class_index, c.label
FROM model_versions mv
CROSS JOIN (VALUES
    (0, 'កាតាប'),
    (1, 'កាតាបស្ពាយក្រោយ'),
    (2, 'កុំព្យូទ័រ'),
    (3, 'កៅអី'),
    (4, 'ក្ដារខៀន'),
    (5, 'ខ្មៅដៃ'),
    (6, 'ជ័រលុប'),
    (7, 'ដីស'),
    (8, 'តុ'),
    (9, 'ទឹកលុប'),
    (10, 'នាយករង'),
    (11, 'នាយិកា'),
    (12, 'បន្ទាត់'),
    (13, 'ប៊ិក'),
    (14, 'ប៊ិកក្រហម'),
    (15, 'ប៊ិកខៀវ'),
    (16, 'សៀវភៅ'),
    (17, 'ហ្វឺតក្រហម'),
    (18, 'ហ្វឺតខៀវ'),
    (19, 'ហ្វឺតខ្មៅ')
) AS c(class_index, label)
WHERE mv.backend = 'backend_keras3'
ON CONFLICT (model_version_id, class_index) DO NOTHING;
