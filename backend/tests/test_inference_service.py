"""
Unit tests for ai_inference.inference_service.InferenceService.

The model is mocked throughout -- these tests verify orchestration
logic (format -> infer -> label-map -> top-k -> response schema), not
the real Keras model itself.
"""

import numpy as np
import pytest

from ai_inference.inference_service import InferenceService


class _RecordingModel:
    """Mock model that records the batch it was called with."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.last_batch = None
        self.call_count = 0

    def predict(self, batch, verbose=0):
        self.last_batch = batch
        self.call_count += 1
        return self.probabilities.reshape(1, -1)


def _labels(n=25):
    return {f"label_{i}": i for i in range(n)}


def _positions():
    return np.random.default_rng(0).random((30, 126)).astype(np.float32)


def test_predict_calls_model_with_formatted_shape():
    probs = np.zeros(25)
    probs[3] = 1.0
    model = _RecordingModel(probs)
    service = InferenceService(model, _labels())

    service.predict(_positions())

    assert model.call_count == 1
    assert model.last_batch.shape == (1, 30, 252)


def test_predict_returns_expected_schema():
    probs = np.zeros(25)
    probs[3] = 1.0
    model = _RecordingModel(probs)
    service = InferenceService(model, _labels())

    result = service.predict(_positions())

    expected_keys = {
        "predicted_class_index",
        "predicted_label",
        "confidence",
        "top_k",
        "inference_ms",
        "sequence_shape",
    }
    assert set(result.keys()) == expected_keys
    assert result["sequence_shape"] == [30, 252]


def test_predict_label_mapping_correct():
    probs = np.zeros(25)
    probs[7] = 1.0
    labels = _labels()
    model = _RecordingModel(probs)
    service = InferenceService(model, labels)

    result = service.predict(_positions())

    assert result["predicted_class_index"] == 7
    assert result["predicted_label"] == "label_7"
    assert result["confidence"] == pytest.approx(1.0)


def test_predict_label_mapping_survives_shuffled_dict_order():
    """
    Guards against relying on dict insertion order: the label dict is
    built in reverse index order, and the mapping must still resolve
    the correct index -> label pair via the explicit reverse map.
    """
    shuffled_labels = {f"label_{i}": i for i in reversed(range(25))}
    probs = np.zeros(25)
    probs[12] = 1.0
    model = _RecordingModel(probs)
    service = InferenceService(model, shuffled_labels)

    result = service.predict(_positions())

    assert result["predicted_label"] == "label_12"


def test_predict_top_k_ordering_and_length():
    probs = np.zeros(25)
    probs[3] = 0.87
    probs[19] = 0.08
    probs[14] = 0.03
    model = _RecordingModel(probs)
    service = InferenceService(model, _labels(), top_k=3)

    result = service.predict(_positions())

    assert len(result["top_k"]) == 3
    confidences = [entry["confidence"] for entry in result["top_k"]]
    assert confidences == sorted(confidences, reverse=True)
    assert result["top_k"][0]["label"] == "label_3"
    assert result["top_k"][0]["confidence"] == pytest.approx(0.87)


def test_predict_respects_custom_top_k():
    probs = np.linspace(0, 1, 25)
    model = _RecordingModel(probs)
    service = InferenceService(model, _labels(), top_k=5)

    result = service.predict(_positions())

    assert len(result["top_k"]) == 5


def test_predict_propagates_feature_formatter_valueerror():
    model = _RecordingModel(np.zeros(25))
    service = InferenceService(model, _labels())

    with pytest.raises(ValueError):
        service.predict(np.zeros((29, 126)))  # wrong shape

    # The model must never be called if formatting already failed.
    assert model.call_count == 0


def test_predict_propagates_valueerror_on_nan_input():
    model = _RecordingModel(np.zeros(25))
    service = InferenceService(model, _labels())

    bad_positions = _positions()
    bad_positions[0, 0] = np.nan

    with pytest.raises(ValueError):
        service.predict(bad_positions)

    assert model.call_count == 0


def test_predict_records_nonnegative_inference_ms():
    model = _RecordingModel(np.zeros(25))
    service = InferenceService(model, _labels())

    result = service.predict(_positions())

    assert result["inference_ms"] >= 0