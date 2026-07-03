import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

LABEL_PATH = BASE_DIR / os.getenv(
    "LABEL_MAP_PATH",
    "labels/label_map_25class.json"
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
        _labels = json.load(f)

    print("=" * 50)
    print("Label Map Loaded")
    print("=" * 50)

    print(f"Total Classes: {len(_labels)}")

    return _labels


def get_labels():

    if _labels is None:
        raise RuntimeError(
            "Labels have not been loaded."
        )

    return _labels