"""
/health/model endpoint.

Reports whether every registered model (see ai_inference.model_registry)
and the shared label map are currently loaded, without loading them
itself. Milestone 9: generalized from a single model's flat status to
a per-model_id breakdown, since backend_keras3 now serves more than
one model. This is a response-shape change from Milestone 5/6's
version, but no frontend code calls /health/model -- it exists for
manual/ops verification only -- so nothing consumes the old shape.
"""

from fastapi import APIRouter, Response

from ai_inference import label_loader, model_loader
from ai_inference.model_registry import MODEL_FILENAMES

router = APIRouter()


@router.get("/health/model")
def health_model(response: Response) -> dict:
    models_status: dict = {}
    all_loaded = True

    for model_id in MODEL_FILENAMES:
        try:
            model = model_loader.get_model(model_id)
            models_status[model_id] = {
                "loaded": True,
                "input_shape": list(model.input_shape),
                "output_shape": list(model.output_shape),
            }
        except RuntimeError:
            models_status[model_id] = {"loaded": False}
            all_loaded = False

    try:
        labels = label_loader.get_labels()
        label_map_loaded = True
        num_classes = len(labels)
    except RuntimeError:
        label_map_loaded = False
        num_classes = None

    if not (all_loaded and label_map_loaded):
        response.status_code = 503

    return {
        "models": models_status,
        "label_map_loaded": label_map_loaded,
        "num_classes": num_classes,
    }
