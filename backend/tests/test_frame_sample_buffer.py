"""
Unit tests for ai_inference.frame_sample_buffer.FrameSampleBuffer.

Covers the full Day 10 test plan:
  - exact-30, downsampling, and padding sampling behavior
  - padding uses the last accumulated frame, not an average or zero
  - reset() clears state correctly
  - sample_sequence() is idempotent without an intervening reset()
  - frame ordering is preserved
  - empty-buffer and malformed-input fail-fast behavior
"""

import numpy as np
import pytest

from ai_inference.frame_sample_buffer import FrameSampleBuffer


def _hand(value, dtype=np.float32):
    """A (63,) array filled with a single distinguishable value."""
    return np.full(63, value, dtype=dtype)


def test_sample_sequence_exact_30_frames():
    buf = FrameSampleBuffer()
    for i in range(30):
        buf.add_frame(_hand(i), _hand(i))

    result = buf.sample_sequence()

    assert result.shape == (30, 126)
    # linspace(0, 29, 30) is the identity permutation -- output must
    # equal the input frames in original order, unchanged.
    assert np.array_equal(result[:, 0], np.arange(30))
    assert np.array_equal(result[:, 63], np.arange(30))


def test_sample_sequence_downsamples_correctly():
    buf = FrameSampleBuffer()
    for i in range(60):
        buf.add_frame(_hand(i), _hand(i))

    result = buf.sample_sequence()
    expected_indices = np.linspace(0, 59, 30, dtype=int)

    assert result.shape == (30, 126)
    assert np.array_equal(result[:, 0], expected_indices)


def test_sample_sequence_pads_when_fewer_than_30():
    buf = FrameSampleBuffer()
    for i in range(10):
        buf.add_frame(_hand(i), _hand(i))

    result = buf.sample_sequence()

    assert result.shape == (30, 126)
    # First 10 rows match the real accumulated frames, in order.
    assert np.array_equal(result[:10, 0], np.arange(10))
    # Remaining 20 rows must equal the last real frame (value 9),
    # not zero and not an average.
    assert np.all(result[10:, 0] == 9)


def test_sample_sequence_pad_uses_last_accumulated_frame():
    buf = FrameSampleBuffer()
    buf.add_frame(_hand(5), _hand(5))
    buf.add_frame(_hand(42), _hand(42))  # last accumulated frame

    result = buf.sample_sequence()

    assert np.all(result[2:, 0] == 42)
    assert not np.any(result[2:, 0] == 5)


def test_single_frame_padding():
    buf = FrameSampleBuffer()
    buf.add_frame(_hand(7), _hand(7))

    result = buf.sample_sequence()

    assert result.shape == (30, 126)
    assert np.all(result[:, 0] == 7)
    assert np.all(result[:, 63] == 7)


def test_reset_clears_buffer():
    buf = FrameSampleBuffer()
    for i in range(30):
        buf.add_frame(_hand(i), _hand(i))

    buf.reset()

    for i in range(5):
        buf.add_frame(_hand(100 + i), _hand(100 + i))

    result = buf.sample_sequence()

    # Only post-reset data should be reflected; padded with the last
    # post-reset frame (104), not any pre-reset value.
    assert np.all(result[:, 0] >= 100)
    assert result[-1, 0] == 104


def test_sample_sequence_idempotent_without_reset():
    buf = FrameSampleBuffer()
    for i in range(45):
        buf.add_frame(_hand(i), _hand(i))

    first_call = buf.sample_sequence()
    second_call = buf.sample_sequence()

    assert np.array_equal(first_call, second_call)


def test_add_frame_accumulates_in_order():
    buf = FrameSampleBuffer()
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    for v in values:
        buf.add_frame(_hand(v), _hand(v))

    # Fewer than 30 frames -> first len(values) rows must preserve
    # insertion order exactly.
    result = buf.sample_sequence()
    assert np.array_equal(result[: len(values), 0], values)


def test_empty_buffer_raises():
    buf = FrameSampleBuffer()
    with pytest.raises(RuntimeError):
        buf.sample_sequence()


def test_add_frame_wrong_left_hand_shape_raises():
    buf = FrameSampleBuffer()
    with pytest.raises(ValueError):
        buf.add_frame(np.zeros(10), _hand(0))


def test_add_frame_wrong_right_hand_shape_raises():
    buf = FrameSampleBuffer()
    with pytest.raises(ValueError):
        buf.add_frame(_hand(0), np.zeros(10))