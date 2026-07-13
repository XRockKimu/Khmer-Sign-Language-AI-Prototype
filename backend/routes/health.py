"""
/health/model endpoint.

Reports whether the model and label map are currently loaded, without
loading them itself. Uses model_loader.get_model() and
label_loader.get_labels() -- the read-only, fail-fast accessors -- so
calling this endpoint never has the side effect of triggering a load.
"""

from fastapi import APIRouter, Response

from ai_inference import label_loader, model_loader

router = APIRouter()


@router.get("/health/model")
def health_model(response: Response) -> dict:
    try:
        model_loader.get_model()
        model_loaded = True
    except RuntimeError:
        model_loaded = False

    try:
        labels = label_loader.get_labels()
        label_map_loaded = True
        num_classes = len(labels)
    except RuntimeError:
        label_map_loaded = False
        num_classes = None

    if not (model_loaded and label_map_loaded):
        response.status_code = 503

    return {
        "model_loaded": model_loaded,
        "label_map_loaded": label_map_loaded,
        "num_classes": num_classes,
    }
