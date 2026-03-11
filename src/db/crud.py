"""CRUD operations for all database tables."""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Optional


# --- Predictions ---

def upsert_prediction(conn: sqlite3.Connection, account_id: str, probability: float,
                      prediction: int, threshold: float, model_version: str = "v1",
                      suspicious_start: str = None, suspicious_end: str = None) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO predictions
           (account_id, probability, prediction, threshold, suspicious_start, suspicious_end, model_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (account_id, probability, prediction, threshold, suspicious_start, suspicious_end,
         model_version, datetime.utcnow().isoformat())
    )
    conn.commit()


def get_prediction(conn: sqlite3.Connection, account_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM predictions WHERE account_id = ?", (account_id,)).fetchone()
    return dict(row) if row else None


def get_all_predictions(conn: sqlite3.Connection, limit: int = 1000) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY probability DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_prediction_stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(prediction) as flagged,
               AVG(probability) as avg_prob,
               MIN(probability) as min_prob,
               MAX(probability) as max_prob
        FROM predictions
    """).fetchone()
    return dict(row) if row else {}


# --- Features ---

def upsert_features(conn: sqlite3.Connection, account_id: str, features: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO features (account_id, feature_json, computed_at)
           VALUES (?, ?, ?)""",
        (account_id, json.dumps(features), datetime.utcnow().isoformat())
    )
    conn.commit()


def get_features(conn: sqlite3.Connection, account_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM features WHERE account_id = ?", (account_id,)).fetchone()
    if row:
        result = dict(row)
        result["features"] = json.loads(result["feature_json"])
        return result
    return None


# --- Explanations ---

def upsert_explanation(conn: sqlite3.Connection, account_id: str,
                       shap_values: dict = None, top_features: list = None,
                       natural_language: str = None) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO explanations
           (account_id, shap_values_json, top_features_json, natural_language, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (account_id,
         json.dumps(shap_values) if shap_values else None,
         json.dumps(top_features) if top_features else None,
         natural_language,
         datetime.utcnow().isoformat())
    )
    conn.commit()


def get_explanation(conn: sqlite3.Connection, account_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM explanations WHERE account_id = ?", (account_id,)).fetchone()
    if row:
        result = dict(row)
        if result.get("shap_values_json"):
            result["shap_values"] = json.loads(result["shap_values_json"])
        if result.get("top_features_json"):
            result["top_features"] = json.loads(result["top_features_json"])
        return result
    return None


# --- Model Registry ---

def register_model(conn: sqlite3.Connection, model_type: str, version: str,
                   auc_roc: float = None, auc_pr: float = None, f1_score: float = None,
                   threshold: float = 0.5, n_features: int = None,
                   model_path: str = None, is_active: bool = False,
                   metadata: dict = None) -> str:
    model_id = str(uuid.uuid4())[:8]
    conn.execute(
        """INSERT INTO model_registry
           (model_id, model_type, version, auc_roc, auc_pr, f1_score, threshold,
            n_features, model_path, is_active, metadata_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (model_id, model_type, version, auc_roc, auc_pr, f1_score, threshold,
         n_features, model_path, int(is_active),
         json.dumps(metadata) if metadata else None,
         datetime.utcnow().isoformat())
    )
    conn.commit()
    return model_id


def get_active_model(conn: sqlite3.Connection) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM model_registry WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row:
        result = dict(row)
        if result.get("metadata_json"):
            result["metadata"] = json.loads(result["metadata_json"])
        return result
    return None


def get_all_models(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM model_registry ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


# --- Fairness Reports ---

def insert_fairness_report(conn: sqlite3.Connection, model_id: str,
                           sensitive_feature: str, demographic_parity_diff: float,
                           equalized_odds_diff: float, pass_80_rule: bool,
                           details: dict = None) -> str:
    report_id = str(uuid.uuid4())[:8]
    conn.execute(
        """INSERT INTO fairness_reports
           (report_id, model_id, sensitive_feature, demographic_parity_diff,
            equalized_odds_diff, pass_80_rule, details_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (report_id, model_id, sensitive_feature, demographic_parity_diff,
         equalized_odds_diff, int(pass_80_rule),
         json.dumps(details) if details else None,
         datetime.utcnow().isoformat())
    )
    conn.commit()
    return report_id


def get_fairness_reports(conn: sqlite3.Connection, model_id: str = None) -> list[dict]:
    if model_id:
        rows = conn.execute(
            "SELECT * FROM fairness_reports WHERE model_id = ? ORDER BY created_at DESC",
            (model_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM fairness_reports ORDER BY created_at DESC"
        ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("details_json"):
            d["details"] = json.loads(d["details_json"])
        results.append(d)
    return results


# --- Benchmark Results ---

def insert_benchmark(conn: sqlite3.Connection, model_type: str, metrics: dict) -> str:
    benchmark_id = str(uuid.uuid4())[:8]
    conn.execute(
        """INSERT INTO benchmark_results (benchmark_id, model_type, metrics_json, created_at)
           VALUES (?, ?, ?, ?)""",
        (benchmark_id, model_type, json.dumps(metrics), datetime.utcnow().isoformat())
    )
    conn.commit()
    return benchmark_id


def get_benchmarks(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM benchmark_results ORDER BY created_at DESC"
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("metrics_json"):
            d["metrics"] = json.loads(d["metrics_json"])
        results.append(d)
    return results
