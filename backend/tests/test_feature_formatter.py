"""
Unit tests for ai_inference.feature_formatter.format_sequence.

Covers the full Day 9 test plan (Revision 2):
  - output shape and position-block correctness
  - velocity computation, including the zeroed first row
  - padded-frame (identical consecutive rows) velocity behavior
  - shape-contract violations
  - dtype coercion
  - fail-fast rejection of NaN / Inf input
  - purity (input array is never mutated)
"""

import numpy as np
import pytest

from ai_inference.feature_formatter import format_sequence


def _make_positions(seed=0):
    """Deterministic (30, 126) fixture for shape/purity-style tests."""
    rng = np.random.default_rng(seed)
    return rng.random((30, 126), dtype=np.float64)


# ---------------------------------------------------------------------
# Shape and structure
# ---------------------------------------------------------------------

def test_output_shape():
    positions = _make_positions()
    result = format_sequence(positions)
    assert result.shape == (30, 252)


def test_position_block_unchanged():
    positions = _make_positions()
    result = format_sequence(positions)
    # Columns 0:126 must equal the input exactly (allowing for the
    # float64 -> float32 cast, not a logical change).
    np.testing.assert_allclose(
        result[:, :126], positions, rtol=1e-6, atol=1e-6
    )


# ---------------------------------------------------------------------
# Velocity computation
# ---------------------------------------------------------------------

def test_velocity_first_row_zero():
    positions = _make_positions()
    result = format_sequence(positions)
    assert np.all(result[0, 126:] == 0.0)


def test_velocity_computation():
    """
    Hand-checked velocity values on a small, explicit fixture.

    Only rows 0-2 carry meaningful values; rows 3-29 are constant
    (equal to row 2) so their expected velocity is exactly zero.
    This directly protects against a silent train/inference mismatch
    in the velocity formula itself.
    """
    positions = np.zeros((30, 126), dtype=np.float64)
    positions[0] = 1.0
    positions[1] = 3.0
    positions[2] = 2.0
    positions[3:] = 2.0  # hold constant for remaining frames

    result = format_sequence(positions)

    expected_velocity = np.zeros((30, 126), dtype=np.float32)
    expected_velocity[1] = 3.0 - 1.0   # = 2.0
    expected_velocity[2] = 2.0 - 3.0   # = -1.0
    expected_velocity[3:] = 0.0        # constant frames -> zero velocity

    np.testing.assert_allclose(result[:, 126:], expected_velocity)


def test_padded_frame_zero_velocity():
    """
    Simulates FrameSampleBuffer's last-frame-repeat padding: identical
    consecutive rows must produce exactly zero velocity, not
    floating-point near-zero noise.
    """
    positions = np.zeros((30, 126), dtype=np.float64)
    positions[:20] = np.random.default_rng(1).random(126)
    positions[20:] = positions[19]  # repeat the last real frame

    result = format_sequence(positions)

    velocity_padded_region = result[21:, 126:]
    assert np.array_equal(
        velocity_padded_region, np.zeros_like(velocity_padded_region)
    )


# ---------------------------------------------------------------------
# Shape-contract violations
# ---------------------------------------------------------------------

def test_wrong_row_count_raises():
    positions = np.zeros((29, 126), dtype=np.float32)
    with pytest.raises(ValueError):
        format_sequence(positions)


def test_wrong_column_count_raises():
    positions = np.zeros((30, 125), dtype=np.float32)
    with pytest.raises(ValueError):
        format_sequence(positions)


# ---------------------------------------------------------------------
# Dtype coercion
# ---------------------------------------------------------------------

def test_dtype_coercion():
    positions = _make_positions()  # float64
    result = format_sequence(positions)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------
# Fail-fast: NaN / Inf
# ---------------------------------------------------------------------

@pytest.mark.parametrize("row", [0, 15, 29])
def test_nan_input_raises(row):
    positions = _make_positions()
    positions[row, 0] = np.nan
    with pytest.raises(ValueError):
        format_sequence(positions)


@pytest.mark.parametrize("row,sign", [(0, 1), (15, -1), (29, 1)])
def test_inf_input_raises(row, sign):
    positions = _make_positions()
    positions[row, 0] = sign * np.inf
    with pytest.raises(ValueError):
        format_sequence(positions)


# ---------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------

def test_input_not_modified():
    positions = _make_positions()
    original = positions.copy()

    format_sequence(positions)

    assert np.array_equal(positions, original)