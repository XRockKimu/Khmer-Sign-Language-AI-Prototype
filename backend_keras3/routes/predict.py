"""
/predict and /predict/{model_id} endpoints.

/predict (no model_id) is unchanged behaviorally from Milestone 5/6:
it continues to serve the LSTM model exactly as before, so the
existing LSTM frontend needs no changes and its request/response
contract is untouched -- verified in this milestone by direct
comparison against the pre-refactor response for identical input.

/predict/{model_id} is new in Milestone 9: the same handler logic,
generalized, added to support additional models (starting with GRU)
without duplicating InferenceService construction or inference logic.
Both routes share get_inference_service() and _run_prediction() --
there are two thin route functions, not two implementations.

Milestone 12: _run_prediction() is also the single place prediction
logging happens for every model this backend serves -- lstm, gru, bgru,
blstm all funnel through this one function, so there is exactly one
log_prediction()/log_prediction_error() call site here, not one per
model. A 422 (bad input) or 500 (inference failure) is logged as
status="error"; a 404 (unknown model_id) or 503 (model not loaded) is
not logged, since no prediction was ever attempted in those cases.
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

_inference_services: dict[str, InferenceService] = {}


def get_inference_service(model_id: str) -> InferenceService:
    """
    Lazily builds and caches one InferenceService per model_id, from
    the loaded model and shared label map singletons.

    Raises
    ------
    KeyError
        model_id isn't a model this backend knows about at all.
    RuntimeError
        model_id is known but hasn't been loaded yet.
    Both are mapped to HTTP responses by the route functions below,
    not here -- this function's job is only to build the service.
    """
    if model_id not in _inference_services:
        model = model_loader.get_model(model_id)
        labels = label_loader.get_labels()
        _inference_services[model_id] = InferenceService(model, labels)

    return _inference_services[model_id]


class PredictRequest(BaseModel):
    sequence: List[List[float]]


def _log_error_safely(model_id: str) -> None:
    """
    Defense-in-depth wrapper around log_prediction_error(), mirroring the
    try/except already around log_prediction() below: that function never
    raises either, so reaching this except branch means a bug in the
    logging service itself, not a normal database failure.
    """
    try:
        prediction_logging_service.log_prediction_error(model_id=model_id)
    except Exception as exc:  # pragma: no cover - defense in depth only
        logger.error(
            "Unexpected error while logging prediction failure for "
            "model_id=%s: %s",
            model_id,
            exc,
        )


def _run_prediction(model_id: str, request: PredictRequest) -> dict:
    try:
        service = get_inference_service(model_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_id!r}.",
        )
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=f"Model {model_id!r} is not loaded yet. Check /health/model.",
        )

    positions = np.array(request.sequence, dtype=np.float64)

    try:
        result = service.predict(positions)
    except ValueError as exc:
        # Shape or finiteness contract violation, raised by
        # feature_formatter and propagated unmodified through
        # InferenceService -- a client-side data problem, not a
        # server bug.
        _log_error_safely(model_id)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        # Genuinely unexpected failure (e.g. a TensorFlow-level
        # error). Do not leak internals to the client.
        _log_error_safely(model_id)
        raise HTTPException(
            status_code=500, detail="Internal inference error."
        )

    try:
        prediction_logging_service.log_prediction(
            model_id=model_id,
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


@router.post("/predict")
def predict(request: PredictRequest) -> dict:
    """Unchanged since Milestone 5/6: implicitly serves the LSTM model."""
    return _run_prediction("lstm", request)


@router.post("/predict/{model_id}")
def predict_model(model_id: str, request: PredictRequest) -> dict:
    """General form, added in Milestone 9 to support additional models."""
    return _run_prediction(model_id, request)
