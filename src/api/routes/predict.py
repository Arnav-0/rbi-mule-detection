"""Prediction endpoints for mule account scoring."""
from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.features.registry import get_all_feature_names

logger = logging.getLogger(__name__)

router = APIRouter()

FEATURE_NAMES = get_all_feature_names()


class PredictionRequest(BaseModel):
    """Single account prediction request with feature values."""
    account_id: str = Field(..., description="Unique account identifier")
    features: dict[str, float] = Field(
        ...,
        description="Feature name → value mapping (57 features)",
    )


class PredictionResponse(BaseModel):
    account_id: str
    mule_probability: float
    risk_level: str
    threshold: float = 0.5


class BatchPredictionRequest(BaseModel):
    accounts: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


def _classify_risk(probability: float) -> str:
    if probability >= 0.8:
        return "critical"
    elif probability >= 0.5:
        return "high"
    elif probability >= 0.3:
        return "medium"
    return "low"


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Score a single account for mule probability."""
    from src.api.main import get_model

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_vector = np.array(
        [[request.features.get(f, 0.0) for f in FEATURE_NAMES]]
    )

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(feature_vector)[:, 1][0])
    else:
        prob = float(model.predict(feature_vector)[0])

    return PredictionResponse(
        account_id=request.account_id,
        mule_probability=round(prob, 6),
        risk_level=_classify_risk(prob),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Score multiple accounts in a single request."""
    from src.api.main import get_model

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_matrix = np.array(
        [[acct.features.get(f, 0.0) for f in FEATURE_NAMES] for acct in request.accounts]
    )

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_matrix)[:, 1]
    else:
        probs = model.predict(feature_matrix)

    predictions = [
        PredictionResponse(
            account_id=acct.account_id,
            mule_probability=round(float(p), 6),
            risk_level=_classify_risk(float(p)),
        )
        for acct, p in zip(request.accounts, probs)
    ]
    return BatchPredictionResponse(predictions=predictions)
