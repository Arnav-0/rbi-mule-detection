"""Fairness audit endpoint."""

import sqlite3

from fastapi import APIRouter, Depends

from src.api.schemas import FairnessResponse
from src.api.dependencies import get_db
from src.db import crud

router = APIRouter(tags=["fairness"])


@router.get("/fairness/report", response_model=FairnessResponse)
def fairness_report(
    model_id: str = None,
    db: sqlite3.Connection = Depends(get_db),
):
    reports = crud.get_fairness_reports(db, model_id)

    summary = {}
    if reports:
        passing = sum(1 for r in reports if r.get("pass_80_rule"))
        summary = {
            "total_audits": len(reports),
            "passing_80_rule": passing,
            "overall_pass": passing == len(reports),
        }

    return FairnessResponse(reports=reports, summary=summary)
