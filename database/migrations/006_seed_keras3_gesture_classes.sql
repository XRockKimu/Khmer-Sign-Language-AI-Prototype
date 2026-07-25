-- 006_seed_keras3_gesture_classes.sql
--
-- Seeds gesture_classes for each of the four backend_keras3 model versions
-- registered in 005, from backend_keras3/labels/label_map_lstm.json -- the
-- single label map all four models share (see model_registry.py's
-- docstring). Generated programmatically from that file rather than
-- transcribed by hand, to guarantee the Khmer label text is byte-exact.
--
-- One (model_version_id, class_index) row per model version, per 004's
-- composite primary key: class_index 0 means "កាតាប" for every Keras-3
-- model, but it is a *different* row per model_version_id, not a shared
-- one -- mirroring how the models themselves are separate rows in
-- model_versions even though they share a label map.
--
-- Depends on 005_seed_keras3_model_versions.sql having already run.
--
-- Idempotent: safe to re-run against a database that already has these
-- rows.

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
  AND mv.model_id IN ('lstm', 'gru', 'bgru', 'blstm')
ON CONFLICT (model_version_id, class_index) DO NOTHING;
