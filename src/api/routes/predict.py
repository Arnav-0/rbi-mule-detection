"""Prediction endpoints."""

import json
import logging
import sqlite3

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
)
from src.api.dependencies import get_db, get_model, ModelService
from src.db import crud

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predictions"])


def _build_predict_response(account_id: str, probability: float,
                            threshold: float, model_version: str,
                            explanation: dict = None) -> PredictResponse:
    prediction = int(probability >= threshold)
    label = "MULE" if prediction else "LEGITIMATE"
    top_features = []
    nl_text = ""
    if explanation:
        top_features = explanation.get("top_features", [])
        if isinstance(top_features, str):
            top_features = json.loads(top_features)
        nl_text = explanation.get("natural_language", "")
    return PredictResponse(
        account_id=account_id,
        probability=round(probability, 6),
        prediction=prediction,
        label=label,
        threshold=threshold,
        top_features=top_features if isinstance(top_features, list) else [],
        natural_language=nl_text or "",
        model_version=model_version,
    )


@router.post("/predict", response_model=PredictResponse)
def predict_single(
    req: PredictRequest,
    model: ModelService = Depends(get_model),
    db: sqlite3.Connection = Depends(get_db),
):
    # Check cache first
    cached = crud.get_prediction(db, req.account_id)
    if cached:
        explanation = crud.get_explanation(db, req.account_id)
        return _build_predict_response(
            req.account_id, cached["probability"], req.threshold,
            cached.get("model_version", "v1"), explanation
        )

    if not model.is_loaded:
        raise HTTPException(503, "Model not loaded. Run training pipeline first.")

    # Load features from DB
    feat_row = crud.get_features(db, req.account_id)
    if not feat_row:
        raise HTTPException(404, f"No features found for account {req.account_id}")

    features = feat_row["features"]
    X = np.array([[features.get(f, 0.0) for f in model.feature_names]])
    probability = float(model.predict_proba(X)[0])

    # Cache prediction
    prediction = int(probability >= req.threshold)
    crud.upsert_prediction(db, req.account_id, probability, prediction,
                           req.threshold, model.model_version)

    explanation = crud.get_explanation(db, req.account_id)
    return _build_predict_response(
        req.account_id, probability, req.threshold, model.model_version, explanation
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest,
    model: ModelService = Depends(get_model),
    db: sqlite3.Connection = Depends(get_db),
):
    results = []
    for aid in req.account_ids:
        try:
            single_req = PredictRequest(account_id=aid, threshold=req.threshold)
            resp = predict_single(single_req, model, db)
            results.append(resp)
        except HTTPException:
            results.append(PredictResponse(
                account_id=aid, probability=0.0, prediction=0,
                label="UNKNOWN", threshold=req.threshold,
                model_version=model.model_version
            ))

    flagged = sum(1 for r in results if r.prediction == 1)
    return BatchPredictResponse(predictions=results, total=len(results), flagged=flagged)
