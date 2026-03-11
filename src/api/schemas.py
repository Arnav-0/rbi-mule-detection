"""Pydantic request/response models for the API."""

from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    account_id: str = Field(..., description="Account ID to predict")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")


class PredictResponse(BaseModel):
    account_id: str
    probability: float
    prediction: int
    label: str
    threshold: float
    top_features: list[dict] = []
    natural_language: str = ""
    model_version: str = ""


class BatchPredictRequest(BaseModel):
    account_ids: list[str] = Field(..., min_length=1, max_length=500)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total: int
    flagged: int


class ModelInfoResponse(BaseModel):
    model_id: str = ""
    model_type: str = ""
    version: str = ""
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    f1_score: Optional[float] = None
    threshold: float = 0.5
    n_features: Optional[int] = None
    is_active: bool = True
    metadata: dict = {}


class FeatureInfoResponse(BaseModel):
    features: list[dict]
    total: int
    groups: dict[str, int]


class AccountResponse(BaseModel):
    account_id: str
    prediction: Optional[dict] = None
    features: Optional[dict] = None
    explanation: Optional[dict] = None


class FairnessResponse(BaseModel):
    reports: list[dict]
    summary: dict = {}


class BenchmarkResponse(BaseModel):
    benchmarks: list[dict]
    best_model: str = ""


class DashboardStatsResponse(BaseModel):
    total_predictions: int = 0
    total_flagged: int = 0
    avg_probability: float = 0.0
    models_registered: int = 0
    active_model: str = ""
    flag_rate: float = 0.0


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    model_loaded: bool = False
    db_connected: bool = False
