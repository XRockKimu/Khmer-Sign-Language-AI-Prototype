from pathlib import Path
import tensorflow as tf
import os
from dotenv import load_dotenv

from ai_inference.temporal_attention import TemporalAttention

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / os.getenv(
    "MODEL_PATH",
    "models/best_model_25class_fix.h5"
)

_model = None


def load_model():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    print(f"[INFO] Loading model from:\n{MODEL_PATH}")

    _model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "TemporalAttention": TemporalAttention
        }
    )

    print("[SUCCESS] Model loaded successfully!")

    return _model


def get_model():
    if _model is None:
        raise RuntimeError(
            "Model has not been loaded.\n"
            "Call load_model() first."
        )

    return _model