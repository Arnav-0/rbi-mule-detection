"""Account lookup endpoint."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import AccountResponse
from src.api.dependencies import get_db
from src.db import crud

router = APIRouter(tags=["accounts"])


@router.get("/account/{account_id}", response_model=AccountResponse)
def get_account(account_id: str, db: sqlite3.Connection = Depends(get_db)):
    prediction = crud.get_prediction(db, account_id)
    features = crud.get_features(db, account_id)
    explanation = crud.get_explanation(db, account_id)

    if not prediction and not features:
        raise HTTPException(404, f"Account {account_id} not found")

    return AccountResponse(
        account_id=account_id,
        prediction=prediction,
        features=features.get("features") if features else None,
        explanation={
            "shap_values": explanation.get("shap_values") if explanation else None,
            "top_features": explanation.get("top_features") if explanation else None,
            "natural_language": explanation.get("natural_language") if explanation else None,
        } if explanation else None,
    )
