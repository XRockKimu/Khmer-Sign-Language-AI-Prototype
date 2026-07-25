"""
/predict endpoint.

Accepts a raw, already-sampled (30, 126) position sequence and returns
a prediction. Per the locked API contract, clients never construct or
send the (30, 252) velocity-augmented tensor -- that construction is
hidden inside InferenceService, which this route depends on exclusively.

This module owns only the HTTP boundary: request parsing, delegating
to InferenceService, and translating exceptions into HTTP responses.
No feature formatting, label lookup, or model prediction logic lives
here -- that is ai_inference's responsibility, not routes/.

After every prediction attempt -- success or failure -- this route also
asks services.prediction_logging_service to log it to PostgreSQL. That
call is best-effort and cannot affect the response: prediction_logging_service
never raises, and the try/except below is a second, defensive layer in
case of a bug in the logging code itself -- either way, database logging
failures never turn a successful prediction into an error response, and
never delay the error response for a genuinely bad request either.

Milestone 12: a 422 (bad input) or 500 (inference failure) is now also
logged, as status="error", so the unified prediction log reflects every
/predict call, not just the successful ones. A 503 (model not loaded) is
not logged -- it means no prediction was ever attempted, so there is
nothing to log.
"""

import logging
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai_inference import model_loader, label_loader
from ai_inference.inference_service import InferenceService
from services import prediction_logging_service

logger = logging.getLogger(__name__)

router = APIRouter()

_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """
    Lazily builds the shared InferenceService from the loaded model and
    label map singletons. Raises RuntimeError if either has not been
    loaded yet -- this route maps that to a 503, distinguishing
    "not ready yet" from a genuine 500 error.
    """
    global _inference_service

    if _inference_service is None:
        model = model_loader.get_model()
        labels = label_loader.get_labels()
        _inference_service = InferenceService(model, labels)

    return _inference_service


class PredictRequest(BaseModel):
    sequence: List[List[float]]


def _log_error_safely() -> None:
    """
    Defense-in-depth wrapper around log_prediction_error(), mirroring the
    try/except already around log_prediction() below: that function never
    raises either, so reaching this except branch means a bug in the
    logging service itself, not a normal database failure.
    """
    try:
        prediction_logging_service.log_prediction_error()
    except Exception as exc:  # pragma: no cover - defense in depth only
        logger.error("Unexpected error while logging prediction failure: %s", exc)


@router.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        service = get_inference_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Model or label map is not loaded yet. Check /health/model.",
        )

    positions = np.array(request.sequence, dtype=np.float64)

    try:
        result = service.predict(positions)
    except ValueError as exc:
        # Shape or finiteness contract violation, raised by
        # feature_formatter and propagated unmodified through
        # InferenceService -- a client-side data problem, not a
        # server bug.
        _log_error_safely()
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        # Genuinely unexpected failure (e.g. a TensorFlow-level
        # error). Do not leak internals to the client.
        _log_error_safely()
        raise HTTPException(
            status_code=500, detail="Internal inference error."
        )

    try:
        prediction_logging_service.log_prediction(
            gesture_class_index=result["predicted_class_index"],
            confidence=result["confidence"],
            top_k=result["top_k"],
            inference_ms=result["inference_ms"],
        )
    except Exception as exc:  # pragma: no cover - defense in depth only;
        # log_prediction already catches its own errors, so reaching this
        # branch means a bug in the logging service itself, not a normal
        # database failure. Either way, the prediction response below must
        # still be returned unaffected.
        logger.error("Unexpected error while logging prediction: %s", exc)

    return result