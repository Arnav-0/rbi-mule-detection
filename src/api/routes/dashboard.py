"""Dashboard statistics endpoint."""

import sqlite3

from fastapi import APIRouter, Depends

from src.api.schemas import DashboardStatsResponse
from src.api.dependencies import get_db, get_model, ModelService
from src.db import crud

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/stats", response_model=DashboardStatsResponse)
def dashboard_stats(
    model: ModelService = Depends(get_model),
    db: sqlite3.Connection = Depends(get_db),
):
    stats = crud.get_prediction_stats(db)
    models = crud.get_all_models(db)
    active = crud.get_active_model(db)

    total = stats.get("total", 0) or 0
    flagged = stats.get("flagged", 0) or 0

    return DashboardStatsResponse(
        total_predictions=total,
        total_flagged=flagged,
        avg_probability=round(stats.get("avg_prob", 0.0) or 0.0, 4),
        models_registered=len(models),
        active_model=active["model_type"] if active else ("loaded" if model.is_loaded else "none"),
        flag_rate=round(flagged / total, 4) if total > 0 else 0.0,
    )
