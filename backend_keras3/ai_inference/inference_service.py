"""
Inference orchestration service, equivalent in role to the existing
backend's ai_inference/inference_service.py, adapted for this
backend's (30, 258) -> (30, 686) contract.

One deliberate difference from the existing backend's version: that
one builds index_to_label by inverting a {label: index} map. This
backend's label_loader.get_labels() already returns a plain
{int: str} index -> label dict (see label_loader.py's docstring from
Milestone 2), so there is nothing to invert here -- using it directly
is correct, not a simplification that changes behavior.
"""

import time

import numpy as np

from ai_inference.feature_formatter import format_sequence

DEFAULT_TOP_K = 3


class InferenceService:
    """
    Orchestrates: raw (30, 258) positions -> engineered (30, 686)
    tensor -> model prediction -> label-mapped, top-k response.
    """

    def __init__(self, model, labels: dict, top_k: int = DEFAULT_TOP_K) -> None:
        self._model = model
        self._top_k = top_k
        self._index_to_label = labels

    def predict(self, positions: np.ndarray) -> dict:
        """
        Run one prediction from a raw (30, 258) position sequence.

        Raises
        ------
        ValueError
            Propagated, unmodified, from feature_formatter.format_sequence
            if `positions` violates its shape or finiteness contract.
        """
        features = format_sequence(positions)

        batch = np.expand_dims(features, axis=0)  # (1, 30, 686)

        start = time.perf_counter()
        raw_predictions = self._model.predict(batch, verbose=0)
        inference_ms = (time.perf_counter() - start) * 1000.0

        probabilities = np.asarray(raw_predictions)[0]  # (num_classes,)

        predicted_index = int(np.argmax(probabilities))
        predicted_label = self._index_to_label[predicted_index]
        confidence = float(probabilities[predicted_index])

        top_k_indices = np.argsort(probabilities)[::-1][: self._top_k]
        top_k = [
            {
                "label": self._index_to_label[int(i)],
                "confidence": float(probabilities[i]),
            }
            for i in top_k_indices
        ]

        return {
            "predicted_class_index": predicted_index,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "top_k": top_k,
            "inference_ms": inference_ms,
            "sequence_shape": list(features.shape),
        }
