"""SQLite database schema definitions."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    account_id TEXT PRIMARY KEY,
    probability REAL NOT NULL,
    prediction INTEGER NOT NULL,
    threshold REAL NOT NULL DEFAULT 0.5,
    suspicious_start TEXT,
    suspicious_end TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS features (
    account_id TEXT PRIMARY KEY,
    feature_json TEXT NOT NULL,
    computed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS explanations (
    account_id TEXT PRIMARY KEY,
    shap_values_json TEXT,
    top_features_json TEXT,
    natural_language TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    auc_roc REAL,
    auc_pr REAL,
    f1_score REAL,
    threshold REAL DEFAULT 0.5,
    n_features INTEGER,
    model_path TEXT,
    is_active INTEGER DEFAULT 0,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fairness_reports (
    report_id TEXT PRIMARY KEY,
    model_id TEXT,
    sensitive_feature TEXT NOT NULL,
    demographic_parity_diff REAL,
    equalized_odds_diff REAL,
    pass_80_rule INTEGER,
    details_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

CREATE TABLE IF NOT EXISTS benchmark_results (
    benchmark_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    metrics_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_predictions_probability ON predictions(probability DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);
CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry(is_active);
CREATE INDEX IF NOT EXISTS idx_fairness_model ON fairness_reports(model_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_model ON benchmark_results(model_type);
"""


def get_schema() -> str:
    return SCHEMA
