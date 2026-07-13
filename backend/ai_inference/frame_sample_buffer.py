"""
Frame sample buffer for the KSL AI inference pipeline.

Accumulates raw per-frame hand-position vectors and, on demand,
produces a fixed-length (30, 126) sequence by reproducing the training
notebook's sampling/padding strategy (Cell 5 of landmark_extract.ipynb):

    if accumulated_frames >= 30:
        indices = np.linspace(0, N - 1, 30, dtype=int)
        sequence = accumulated[indices]
    else:
        pad by repeating the last accumulated frame

This module owns only the "Accumulate position frames" and "Downsample
to 30 frames" steps of the Architecture Invariant. It has no knowledge
of gesture start/stop timing (owned by the caller), velocity, or
252-feature tensor construction (owned by feature_formatter).
"""

import numpy as np

# Fixed by the frozen architecture contract and the trained model's
# expected sequence length -- not configurable.
SEQUENCE_LENGTH = 30
HAND_FEATURES = 63
POSITION_FEATURES = HAND_FEATURES * 2  # left + right = 126


class FrameSampleBuffer:
    """
    Accumulates raw (126,) position frames and samples them to a fixed
    (30, 126) sequence.

    This class does not decide when a gesture starts or stops -- the
    caller (a test harness today, a frontend-driven capture loop in
    Week 3) owns that lifecycle entirely. It only buffers whatever
    frames it is given, in the order given, and samples on request.
    """

    def __init__(self, target_length: int = SEQUENCE_LENGTH) -> None:
        self._target_length = target_length
        self._frames = []

    def reset(self) -> None:
        """Clear all accumulated frames. Call before starting a new capture."""
        self._frames = []

    def add_frame(self, left_hand: np.ndarray, right_hand: np.ndarray) -> None:
        """
        Append one frame's hand-position vectors to the buffer.

        Parameters
        ----------
        left_hand, right_hand : np.ndarray
            Each shape (63,), as produced by
            HandKeypointExtractor.extract().

        Raises
        ------
        ValueError
            If either array is not shape (63,). Treated as an upstream
            contract violation, consistent with the fail-fast standard
            established in feature_formatter.
        """
        left_hand = np.asarray(left_hand)
        right_hand = np.asarray(right_hand)

        if left_hand.shape != (HAND_FEATURES,):
            raise ValueError(
                f"add_frame expected left_hand of shape ({HAND_FEATURES},), "
                f"got {left_hand.shape}. Check HandKeypointExtractor output."
            )

        if right_hand.shape != (HAND_FEATURES,):
            raise ValueError(
                f"add_frame expected right_hand of shape ({HAND_FEATURES},), "
                f"got {right_hand.shape}. Check HandKeypointExtractor output."
            )

        combined = np.concatenate([left_hand, right_hand])
        self._frames.append(combined)

    def sample_sequence(self) -> np.ndarray:
        """
        Produce a (30, 126) sequence from the accumulated frames.

        If 30 or more frames have been accumulated, selects 30 evenly
        spaced frames across the full accumulated history via
        np.linspace, matching the training notebook exactly. If fewer
        than 30 frames have been accumulated, pads by repeating the
        last accumulated frame, also matching the notebook's fallback.

        Does NOT clear the buffer -- call reset() explicitly before
        starting the next capture window. Calling this repeatedly
        without an intervening reset() returns the same result each
        time, since it operates on whatever has been accumulated so far.

        Returns
        -------
        np.ndarray
            Shape (30, 126).

        Raises
        ------
        RuntimeError
            If no frames have been accumulated. This is a buffer-state
            violation (nothing to sample), distinct from an invalid-
            input violation, but is still treated as fail-fast per the
            same doctrine as feature_formatter's input validation.
        """
        total_frames = len(self._frames)

        if total_frames == 0:
            raise RuntimeError(
                "sample_sequence called with no accumulated frames. "
                "Call add_frame() at least once after reset()/__init__ "
                "before sampling."
            )

        accumulated = np.stack(self._frames, axis=0)  # (N, 126)

        if total_frames >= self._target_length:
            indices = np.linspace(
                0, total_frames - 1, self._target_length, dtype=int
            )
            sequence = accumulated[indices]
        else:
            pad_needed = self._target_length - total_frames
            last_frame = accumulated[-1:]
            padding = np.repeat(last_frame, pad_needed, axis=0)
            sequence = np.concatenate([accumulated, padding], axis=0)

        return sequence