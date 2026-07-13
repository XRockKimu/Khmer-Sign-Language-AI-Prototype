"""
Feature formatter for the KSL AI inference pipeline.

Converts a sampled (30, 126) hand-position sequence into the (30, 252)
model-input tensor, reproducing the training notebook's feature
construction exactly (Cell 5 of landmark_extract.ipynb):

    positions (30, 126)
        -> velocity[0]  = 0
        -> velocity[1:] = positions[1:] - positions[:-1]
        -> features     = concatenate([positions, velocity], axis=1)

This module owns only the "Compute velocity" and "Construct 252-feature
tensor" steps of the Architecture Invariant. It has no knowledge of
capture, buffering, sampling, or the model itself, and performs no
gesture-lifecycle or MediaPipe logic.
"""

import numpy as np

# Fixed by the frozen architecture contract and the trained model's
# input shape -- not configurable, since any deviation here would break
# compatibility with best_model_25class_fix.h5.
SEQUENCE_LENGTH = 30
POSITION_FEATURES = 126


def format_sequence(positions: np.ndarray) -> np.ndarray:
    """
    Build the (30, 252) model-input tensor from a sampled position sequence.

    This is a pure, stateless transformation: it performs no sampling,
    buffering, or capture, and it never mutates the caller's array.

    Parameters
    ----------
    positions : np.ndarray
        Shape (30, 126). Row order must already reflect the training
        pipeline's downsampling (see FrameSampleBuffer.sample_sequence),
        and each frame's 126 columns must be ordered
        [left_hand(63), right_hand(63)] wrist-relative, per
        keypoint_extractor.HandKeypointExtractor.

    Returns
    -------
    np.ndarray
        Shape (30, 252), dtype float32. Columns 0:126 are the unmodified
        input positions; columns 126:252 are frame-to-frame velocity,
        with row 0's velocity set to zero.

    Raises
    ------
    ValueError
        If `positions` is not shape (30, 126), or contains any NaN or
        Inf value. Invalid input is treated as an upstream pipeline
        defect (e.g. a failed MediaPipe extraction or a corrupted
        buffer read) -- this function fails fast rather than
        silently coercing, ignoring, or forwarding bad data to the model.
    """
    positions = np.asarray(positions)

    if positions.shape != (SEQUENCE_LENGTH, POSITION_FEATURES):
        raise ValueError(
            "format_sequence expected an array of shape "
            f"({SEQUENCE_LENGTH}, {POSITION_FEATURES}), got "
            f"{positions.shape}. This indicates an upstream contract "
            "violation -- check FrameSampleBuffer.sample_sequence()."
        )

    if not np.isfinite(positions).all():
        raise ValueError(
            "format_sequence received non-finite values (NaN or Inf) "
            "in the input position sequence. This is treated as an "
            "upstream pipeline defect -- check the keypoint extraction "
            "and buffering stages that produced this sequence, rather "
            "than the model input itself."
        )

    # Validate before casting, so a dtype conversion never has a chance
    # to mask or alter a NaN/Inf value that should have already failed.
    positions = positions.astype(np.float32, copy=True)

    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]

    return np.concatenate([positions, velocity], axis=1)