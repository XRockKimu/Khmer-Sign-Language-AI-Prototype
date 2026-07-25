"""
Model loader for the Keras-3 model family.

Milestone 9: generalized from a single global model to a dict keyed by
model_id, since backend_keras3 now serves more than one model (lstm,
gru, ...) from the same process. Still no custom_objects handling --
none of these models use custom layers (confirmed during the earlier
Keras-3 compatibility testing).

get_model() distinguishes two failure modes deliberately, because
callers (see routes/predict.py, routes/health.py) need to react to
them differently:
  - KeyError: model_id isn't a model this backend even knows about
    (maps to 404 -- there's nothing to wait for).
  - RuntimeError: model_id is known but load_model()/load_all_models()
    hasn't been called yet (maps to 503 -- try again once started).
"""

from pathlib import Path
import os
import tensorflow as tf
from dotenv import load_dotenv

from ai_inference.model_registry import MODEL_FILENAMES

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = BASE_DIR / os.getenv("MODELS_DIR", "models")

_models: dict = {}


def load_model(model_id: str):
    """
    Loads and caches the model for `model_id` if not already loaded.
    Idempotent: safe to call repeatedly for the same model_id.
    """
    if model_id in _models:
        return _models[model_id]

    if model_id not in MODEL_FILENAMES:
        raise KeyError(
            f"Unknown model_id: {model_id!r}. Known models: {sorted(MODEL_FILENAMES)}"
        )

    model_path = MODELS_DIR / MODEL_FILENAMES[model_id]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    print(f"[INFO] Loading model '{model_id}' from:\n{model_path}")

    _models[model_id] = tf.keras.models.load_model(model_path, compile=False)

    print(f"[SUCCESS] Model '{model_id}' loaded successfully!")

    return _models[model_id]


def load_all_models():
    """Loads every model in MODEL_FILENAMES. Called once at app startup."""
    for model_id in MODEL_FILENAMES:
        load_model(model_id)


def get_model(model_id: str):
    if model_id not in MODEL_FILENAMES:
        raise KeyError(f"Unknown model_id: {model_id!r}")

    if model_id not in _models:
        raise RuntimeError(
            f"Model '{model_id}' has not been loaded.\n"
            f"Call load_model({model_id!r}) first."
        )

    return _models[model_id]
