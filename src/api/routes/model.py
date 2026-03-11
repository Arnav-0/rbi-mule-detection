"""Model info and feature endpoints."""

import sqlite3
from collections import Counter

from fastapi import APIRouter, Depends

from src.api.schemas import ModelInfoResponse, FeatureInfoResponse
from src.api.dependencies import get_db, get_model, ModelService
from src.db import crud
from src.features.registry import FEATURE_REGISTRY

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(
    model: ModelService = Depends(get_model),
    db: sqlite3.Connection = Depends(get_db),
):
    active = crud.get_active_model(db)
    if active:
        return ModelInfoResponse(
            model_id=active["model_id"],
            model_type=active["model_type"],
            version=active["version"],
            auc_roc=active.get("auc_roc"),
            auc_pr=active.get("auc_pr"),
            f1_score=active.get("f1_score"),
            threshold=active.get("threshold", 0.5),
            n_features=active.get("n_features"),
            is_active=True,
            metadata=active.get("metadata", {}),
        )
    return ModelInfoResponse(
        model_type="unknown",
        version=model.model_version,
        threshold=model.threshold,
        n_features=len(model.feature_names),
        is_active=model.is_loaded,
    )


@router.get("/model/features", response_model=FeatureInfoResponse)
def model_features():
    features = [
        {"name": name, **meta}
        for name, meta in FEATURE_REGISTRY.items()
    ]
    groups = Counter(meta["group"] for meta in FEATURE_REGISTRY.values())
    return FeatureInfoResponse(
        features=features,
        total=len(features),
        groups=dict(groups),
    )
