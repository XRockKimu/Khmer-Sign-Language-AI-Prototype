"""
Milestone 2: label loader for the Keras-3 model family.

Unlike the existing backend's label map (label_map_25class.json,
{label: index}), label_map_lstm.json is already stored as
{index: label} -- e.g. {"0": "កាតាប", ...}. JSON object keys are
always strings on disk, so this loader converts them to int once at
load time; the key is semantically an integer class index, not a
string, and callers should never have to know or care that the file
on disk stored it as one. That is the one piece of normalization this
loader does -- it does not invert label<->index like the existing
backend's InferenceService does, because there is nothing to invert
here: the file already maps index -> label directly.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

LABEL_PATH = BASE_DIR / os.getenv(
    "LABEL_MAP_PATH",
    "labels/label_map_lstm.json"
)

_labels = None


def load_labels():
    global _labels

    if _labels is not None:
        return _labels

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Label map not found:\n{LABEL_PATH}"
        )

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    _labels = {int(index): label for index, label in raw.items()}

    print("=" * 50)
    print("Label Map Loaded")
    print("=" * 50)
    print(f"Total Classes: {len(_labels)}")

    return _labels


def get_labels():
    if _labels is None:
        raise RuntimeError(
            "Labels have not been loaded.\n"
            "Call load_labels() first."
        )

    return _labels
