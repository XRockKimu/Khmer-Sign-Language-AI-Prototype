"""
Inference orchestration service for the KSL AI inference pipeline.

Owns the seam between a raw sampled (30, 126) position sequence and a
finished prediction response. This is the single place where the
Architecture Invariant's remaining two stages are stitched together in
code:

    Downsample to 30 frames   (caller/FrameSampleBuffer -- upstream of this)
        -> Compute velocity          )
        -> Construct 252-feature     ) done here, via feature_formatter
           tensor                    )
        -> Model inference           ) done here, via the injected model

InferenceService does not load the model or label map itself -- both
are injected via the constructor, keeping this class testable with
mocks and consistent with model_loader/label_loader owning their own
singleton lifecycles.
"""

import time

import numpy as np

from ai_inference.feature_formatter import format_sequence

DEFAULT_TOP_K = 3


class InferenceService:
    """
    Orchestrates: raw (30, 126) positions -> formatted tensor -> model
    prediction -> label-mapped, top-k response.
    """

    def __init__(self, model, labels: dict, top_k: int = DEFAULT_TOP_K) -> None:
        """
        Parameters
        ----------
        model : object with a .predict(batch) method
            Typically the singleton returned by
            ai_inference.model_loader.get_model(), but this class has
            no import-level dependency on that module -- any object
            exposing .predict() and accepting a (1, 30, 252) batch,
            returning (1, num_classes) probabilities, will work.
        labels : dict
            Maps label string -> class index, as loaded by
            ai_inference.label_loader (e.g. {"No_action": 0, ...}).
        top_k : int
            Number of top predictions to include in the response.
        """
        self._model = model
        self._top_k = top_k

        # Build an explicit index -> label reverse map from the label
        # dict's values, rather than relying on dict insertion order
        # matching class index order. This is safer than assuming
        # list(labels.keys())[index] holds indefinitely.
        self._index_to_label = {index: label for label, index in labels.items()}

    def predict(self, positions: np.ndarray) -> dict:
        """
        Run one prediction from a raw sampled position sequence.

        Parameters
        ----------
        positions : np.ndarray
            Shape (30, 126) -- the output of
            FrameSampleBuffer.sample_sequence(), NOT yet velocity- or
            tensor-formatted.

        Returns
        -------
        dict
            {
              "predicted_class_index": int,
              "predicted_label": str,
              "confidence": float,
              "top_k": [{"label": str, "confidence": float}, ...],
              "inference_ms": float,
              "sequence_shape": [30, 252],
            }

        Raises
        ------
        ValueError
            Propagated, unmodified, from feature_formatter.format_sequence
            if `positions` violates its shape or finiteness contract.
            This method does not catch or reinterpret that error --
            translating it into an HTTP response is routes_predict.py's
            job, not this service's.
        """
        # "Construct 252-feature tensor" -- the one call site where
        # feature_formatter is invoked, per the frozen architecture.
        features = format_sequence(positions)

        batch = np.expand_dims(features, axis=0)  # (1, 30, 252)

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