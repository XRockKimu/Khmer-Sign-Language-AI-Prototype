"""
FastAPI application entry point.

Loads the trained model and label map once at startup via a lifespan
context manager -- fail-fast: if the .h5 file or label JSON is
missing, the app will not start -- then mounts the /predict and
/health/model routers.

Run with (from within backend/):

    uvicorn main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

import routes.health as health_route
import routes.predict as predict_route
from ai_inference import label_loader, model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_model()
    label_loader.load_labels()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(predict_route.router)
app.include_router(health_route.router)
