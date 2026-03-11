"""FastAPI dependencies: model loading, DB connections."""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.db.init_db import init_database, get_connection
from src.utils.config import DB_PATH, MODEL_PATH, OUTPUTS_DIR
from src.features.registry import FEATURE_REGISTRY, get_all_feature_names

logger = logging.getLogger(__name__)


class ModelService:
    """Singleton service holding the loaded ML model."""

    def __init__(self):
        self.model = None
        self.feature_names: list[str] = get_all_feature_names()
        self.model_version: str = "v1"
        self.threshold: float = 0.5

    def load(self, path: Path = MODEL_PATH) -> bool:
        if path.exists():
            try:
                self.model = joblib.load(path)
                logger.info("Model loaded from %s", path)
                return True
            except Exception as e:
                logger.warning("Failed to load model from %s: %s", path, e)
        else:
            logger.warning("Model file not found at %s", path)
        return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        t = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(X)
        return (proba >= t).astype(int)


# Global instances
model_service = ModelService()
_db_initialized = False


def get_db() -> sqlite3.Connection:
    """Get a database connection, initializing if needed."""
    global _db_initialized
    if not _db_initialized:
        init_database()
        _db_initialized = True
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def get_model() -> ModelService:
    return model_service


def startup_load_model() -> None:
    """Called at FastAPI startup to load model and init DB."""
    global _db_initialized
    init_database()
    _db_initialized = True
    model_service.load()
    # Try loading threshold from reports
    threshold_path = OUTPUTS_DIR / "reports" / "threshold.txt"
    if threshold_path.exists():
        try:
            model_service.threshold = float(threshold_path.read_text().strip())
            logger.info("Loaded threshold: %.4f", model_service.threshold)
        except ValueError:
            pass
