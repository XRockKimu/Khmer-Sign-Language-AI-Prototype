-- 002_seed_gesture_classes.sql
--
-- Seeds gesture_classes from backend/labels/label_map_25class.json, the
-- label map the currently deployed model (best_model_25class_fix.h5) was
-- trained against. Generated programmatically from that file rather than
-- transcribed by hand, to guarantee the Khmer label text is byte-exact.
--
-- Idempotent: safe to re-run against a database that already has these
-- rows.

INSERT INTO gesture_classes (class_index, label) VALUES
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
ON CONFLICT (class_index) DO NOTHING;
