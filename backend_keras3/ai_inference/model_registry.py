"""
Registry of every model backend_keras3 can serve, keyed by model_id.

All four Keras-3 models (lstm/gru/bgru/blstm) share the identical
686-feature pipeline (ai_inference.feature_formatter) and the
identical label map (labels/label_map_lstm.json) -- established during
the migration-planning review of live_testing_script.ipynb, where
every "Lt for X" cell loads labels from the same hardcoded path and
uses the same SignLanguageNormalizer/FeatureEngineeringEngine
configuration. Only the .h5 weights file differs per model, so this
registry is deliberately just a name -> filename mapping, not a
per-model pipeline configuration -- there is nothing else that varies.

Adding bgru/blstm later means adding one line here; nothing else in
ai_inference/ needs to change for them.
"""

MODEL_FILENAMES = {
    "lstm": "lstm_best_model_v1.h5",
    "gru": "gru_best_model_v2.h5",
    "bgru": "bgru_best_model_v1.h5",
    "blstm": "blstm_best_model_v1.h5",
}
