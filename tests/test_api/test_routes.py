"""Tests for FastAPI routes."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "model_loaded" in data
    assert "db_connected" in data


def test_predict_returns_json(client):
    resp = client.post("/predict", json={"account_id": "NONEXISTENT_123", "threshold": 0.5})
    # Either 404 (no features) or 503 (no model) — both are valid JSON error responses
    assert resp.status_code in (404, 503)
    data = resp.json()
    assert "detail" in data


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_type" in data
    assert "version" in data
    assert "n_features" in data


def test_model_features(client):
    resp = client.get("/model/features")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 57
    assert len(data["features"]) == 57
    assert "velocity" in data["groups"]


def test_dashboard_stats(client):
    resp = client.get("/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "total_flagged" in data


def test_fairness_report(client):
    resp = client.get("/fairness/report")
    assert resp.status_code == 200
    data = resp.json()
    assert "reports" in data


def test_benchmark_results(client):
    resp = client.get("/benchmark/results")
    assert resp.status_code == 200
    data = resp.json()
    assert "benchmarks" in data


def test_account_not_found(client):
    resp = client.get("/account/NONEXISTENT_999")
    assert resp.status_code == 404
