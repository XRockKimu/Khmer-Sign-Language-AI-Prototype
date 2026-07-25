"""
Top-level orchestration for the 686-feature pipeline, equivalent in role
to the existing backend's ai_inference/feature_formatter.py (which builds
the (30, 252) tensor), but for this backend's (30, 686) contract instead.

Ported from backend/notebooks/live_testing_script.ipynb's "Lt for lstm"
cell (cell 11), function process_sequence(seq_258), which itself
composes:
  - the velocity step from cell 1's add_velocity(sequence_258)
  - sequence_normalizer.SignLanguageNormalizer.normalize_sequence()
  - feature_engineering.FeatureEngineeringEngine.process_sequence()

into one call: (30, 258) -> (30, 516) -> (30, 686).

This module does not perform pose/hand extraction (that step produces
the (30, 258) input this module expects) and does not buffer frames --
both are deferred to the milestone that introduces real frame capture.
"""

import numpy as np

from ai_inference.sequence_normalizer import SignLanguageNormalizer
from ai_inference.feature_engineering import FeatureEngineeringEngine

# Fixed by the trained model's input contract, mirroring the existing
# backend's feature_formatter.py convention of naming these explicitly
# rather than leaving them as unexplained literals.
SEQUENCE_LENGTH = 30
POSITION_FEATURES = 258
VELOCITY_FEATURES = 258
POSITION_AND_VELOCITY_FEATURES = POSITION_FEATURES + VELOCITY_FEATURES  # 516
TOTAL_FEATURES = 686

_normalizer = SignLanguageNormalizer(
    use_root_center=True,
    use_shoulder_width=True,
    use_hand_normalization=True,
    use_scale_normalization=True,
    use_clipping=True,
    target_hand_size=0.3,
    clip_bounds=(-2.0, 2.0),
)


def _add_velocity(sequence_258: np.ndarray) -> np.ndarray:
    """
    Ported from cell 1's add_velocity(sequence_258): frame-to-frame
    delta, first frame's velocity is zero, concatenated onto the
    original positions.
    """
    velocity = np.zeros_like(sequence_258)
    velocity[1:] = sequence_258[1:] - sequence_258[:-1]
    return np.concatenate([sequence_258, velocity], axis=1)


def format_sequence(positions: np.ndarray) -> np.ndarray:
    """
    Build the (30, 686) model-input tensor from a (30, 258) raw
    pose+hands position sequence.

    Parameters
    ----------
    positions : np.ndarray
        Shape (30, 258): 33-point pose (132, hip-centered, x/y/z/visibility)
        followed by left hand (63, wrist-relative) and right hand
        (63, wrist-relative), per frame.

    Returns
    -------
    np.ndarray
        Shape (30, 686), dtype float32.

    Raises
    ------
    ValueError
        If `positions` is not shape (30, 258), or contains any NaN or
        Inf value -- same fail-fast contract as the existing backend's
        feature_formatter.format_sequence.
    """
    positions = np.asarray(positions)

    if positions.shape != (SEQUENCE_LENGTH, POSITION_FEATURES):
        raise ValueError(
            "format_sequence expected an array of shape "
            f"({SEQUENCE_LENGTH}, {POSITION_FEATURES}), got "
            f"{positions.shape}."
        )

    if not np.isfinite(positions).all():
        raise ValueError(
            "format_sequence received non-finite values (NaN or Inf) "
            "in the input position sequence."
        )

    positions = positions.astype(np.float32, copy=True)

    sequence_516 = _add_velocity(positions)
    sequence_516_normalized = _normalizer.normalize_sequence(sequence_516)

    feature_engine = FeatureEngineeringEngine()
    sequence_686 = feature_engine.process_sequence(sequence_516_normalized)

    return sequence_686.astype(np.float32)
