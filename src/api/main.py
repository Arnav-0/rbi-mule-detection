"""FastAPI application for RBI Mule Account Detection."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.utils.config import MODEL_PATH

logger = logging.getLogger(__name__)

_model = None


def get_model():
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        _model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
    else:
        logger.warning("No model found at %s — predictions will be unavailable", model_path)
    yield
    _model = None


app = FastAPI(
    title="RBI Mule Account Detection API",
    description="REST API for detecting mule accounts in banking transaction data.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router, prefix="/api/v1")
