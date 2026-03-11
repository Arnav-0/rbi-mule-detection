# API Reference

Base URL: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`

---

## GET /health

Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "db_connected": true
}
```

---

## POST /predict

Single account prediction with SHAP explanation.

**Request:**
```json
{
  "account_id": "ACCT_000003",
  "threshold": 0.5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| account_id | string | yes | Account ID to predict |
| threshold | float | no | Classification threshold (default: 0.5, range: 0-1) |

**Response (200):**
```json
{
  "account_id": "ACCT_000003",
  "probability": 0.8734,
  "prediction": 1,
  "label": "MULE",
  "threshold": 0.5,
  "top_features": [
    {"feature": "rapid_turnover_score", "shap_value": 0.1823},
    {"feature": "velocity_acceleration", "shap_value": 0.1456},
    {"feature": "burst_after_dormancy", "shap_value": 0.0987}
  ],
  "natural_language": "This account shows rapid pass-through behavior with high velocity acceleration following a dormancy period.",
  "model_version": "v1"
}
```

**Errors:** `404` (no features for account), `503` (model not loaded)

---

## POST /predict/batch

Batch prediction for up to 500 accounts.

**Request:**
```json
{
  "account_ids": ["ACCT_000003", "ACCT_000077", "ACCT_001234"],
  "threshold": 0.5
}
```

**Response (200):**
```json
{
  "predictions": [
    {
      "account_id": "ACCT_000003",
      "probability": 0.8734,
      "prediction": 1,
      "label": "MULE",
      "threshold": 0.5,
      "top_features": [],
      "natural_language": "",
      "model_version": "v1"
    }
  ],
  "total": 3,
  "flagged": 1
}
```

---

## GET /account/{account_id}

Full analysis for a single account from the database cache.

**Response (200):**
```json
{
  "account_id": "ACCT_000003",
  "prediction": {
    "account_id": "ACCT_000003",
    "probability": 0.8734,
    "prediction": 1,
    "threshold": 0.5,
    "suspicious_start": "2023-11-15T09:30:00",
    "suspicious_end": "2024-02-20T16:45:00",
    "model_version": "v1",
    "created_at": "2025-06-30T12:00:00"
  },
  "features": {
    "txn_count_7d": 45,
    "rapid_turnover_score": 0.82,
    "pagerank": 0.0012
  },
  "explanation": {
    "shap_values": {"rapid_turnover_score": 0.1823},
    "top_features": [{"feature": "rapid_turnover_score", "shap_value": 0.1823}],
    "natural_language": "High pass-through behavior detected."
  }
}
```

**Errors:** `404` (account not found in database)

---

## GET /model/info

Current active model metadata.

**Response (200):**
```json
{
  "model_id": "a1b2c3d4",
  "model_type": "catboost",
  "version": "v1",
  "auc_roc": 0.9109,
  "auc_pr": 0.4587,
  "f1_score": 0.4779,
  "threshold": 0.5,
  "n_features": 57,
  "is_active": true,
  "metadata": {}
}
```

---

## GET /model/features

Feature list with metadata from the 57-feature registry.

**Response (200):**
```json
{
  "features": [
    {
      "name": "txn_count_1d",
      "group": "velocity",
      "description": "Transaction count in last 1 day",
      "dtype": "int32",
      "power": "High"
    }
  ],
  "total": 57,
  "groups": {
    "velocity": 10,
    "amount_patterns": 8,
    "temporal": 8,
    "passthrough": 7,
    "graph_network": 10,
    "profile_mismatch": 5,
    "kyc_behavioral": 4,
    "interactions": 5
  }
}
```

---

## GET /dashboard/stats

Aggregate prediction statistics for the dashboard.

**Response (200):**
```json
{
  "total_predictions": 16015,
  "total_flagged": 174,
  "avg_probability": 0.0342,
  "models_registered": 6,
  "active_model": "catboost",
  "flag_rate": 0.0109
}
```

---

## GET /fairness/report

Fairness audit results across sensitive features.

**Query params:** `model_id` (optional) — filter by specific model.

**Response (200):**
```json
{
  "reports": [
    {
      "report_id": "f1a2b3c4",
      "model_id": "a1b2c3d4",
      "sensitive_feature": "rural_branch",
      "demographic_parity_diff": 0.032,
      "equalized_odds_diff": 0.018,
      "pass_80_rule": 1,
      "details": {
        "Y": {"tpr": 0.85, "fpr": 0.03},
        "N": {"tpr": 0.87, "fpr": 0.02}
      },
      "created_at": "2025-06-30T12:00:00"
    }
  ],
  "summary": {
    "total_audits": 3,
    "passing_80_rule": 3,
    "overall_pass": true
  }
}
```

---

## GET /benchmark/results

Model comparison benchmark results.

**Response (200):**
```json
{
  "benchmarks": [
    {
      "benchmark_id": "b1c2d3e4",
      "model_type": "catboost",
      "metrics": {
        "auc_roc": 0.9109,
        "auc_pr": 0.4587,
        "f1_score": 0.4779,
        "precision": 0.42,
        "recall": 0.55
      },
      "created_at": "2025-06-30T12:00:00"
    }
  ],
  "best_model": "catboost"
}
```

---

## Database Tables

The API uses SQLite (`outputs/db/mule_detection.db`) with 6 tables:

| Table | Primary Key | Purpose |
|-------|-------------|---------|
| predictions | account_id | Cached prediction results |
| features | account_id | Computed feature vectors (JSON) |
| explanations | account_id | SHAP values + natural language |
| model_registry | model_id | Trained model metadata |
| fairness_reports | report_id | Fairness audit results |
| benchmark_results | benchmark_id | Model comparison metrics |
