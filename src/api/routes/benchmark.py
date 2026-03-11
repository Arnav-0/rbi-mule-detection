"""Benchmark results endpoint."""

import sqlite3

from fastapi import APIRouter, Depends

from src.api.schemas import BenchmarkResponse
from src.api.dependencies import get_db
from src.db import crud

router = APIRouter(tags=["benchmark"])


@router.get("/benchmark/results", response_model=BenchmarkResponse)
def benchmark_results(db: sqlite3.Connection = Depends(get_db)):
    benchmarks = crud.get_benchmarks(db)

    best_model = ""
    if benchmarks:
        best = max(benchmarks, key=lambda b: b.get("metrics", {}).get("auc_roc", 0))
        best_model = best.get("model_type", "")

    return BenchmarkResponse(benchmarks=benchmarks, best_model=best_model)
