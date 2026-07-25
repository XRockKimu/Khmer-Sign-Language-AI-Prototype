"""
FastAPI application entry point for backend_keras3.

Mirrors the existing backend's main.py: loads every registered model
(see ai_inference.model_registry) and the shared label map once at
startup via a lifespan context manager -- fail-fast, so if any .h5
file or the label JSON is missing, the app will not start -- then
mounts the /predict, /health/model, and /keypoints/extract routers.

Milestone 12: routes/predict.py logs every prediction (success or
error) via services.prediction_logging_service, but that wiring is
lazy and best-effort, not startup wiring -- see db/connection.py's
docstring -- so this lifespan function still has nothing to do for
the database, same as before.

Run with (from within backend_keras3/), on a different port than the
existing backend so both can run at once:

    uvicorn main:app --reload --port 8001
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

import routes.health as health_route
import routes.keypoints as keypoints_route
import routes.predict as predict_route
from ai_inference import label_loader, model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_all_models()
    label_loader.load_labels()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(predict_route.router)
app.include_router(health_route.router)
app.include_router(keypoints_route.router)
