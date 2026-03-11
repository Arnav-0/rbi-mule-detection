"""Initialize SQLite database with schema and seed from model artifacts."""

import argparse
import hashlib
import json
import logging
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.db.models import get_schema
from src.utils.config import DB_PATH

logger = logging.getLogger(__name__)


def init_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(get_schema())
        conn.commit()
        logger.info("Database initialized at %s", db_path)
    finally:
        conn.close()


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def seed_database(db_path: Path = DB_PATH) -> None:
    """Seed database from trained model artifacts."""
    init_database(db_path)
    conn = sqlite3.connect(str(db_path))

    try:
        # --- 1. Model registry ---
        model_path = Path("outputs/models/best_model.joblib")
        if model_path.exists():
            model = joblib.load(model_path)
            model_type = type(model).__name__.lower()
            model_id = hashlib.md5(str(model_path).encode()).hexdigest()[:8]

            # Try to get metrics from submission
            features = pd.read_parquet("data/processed/features_matrix.parquet")
            n_features = features.shape[1]

            conn.execute(
                """INSERT OR REPLACE INTO model_registry
                   (model_id, model_type, version, auc_roc, auc_pr, f1_score,
                    threshold, n_features, model_path, is_active, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_id, model_type, "v1", None, None, None,
                 0.5, n_features, str(model_path), 1, "{}"),
            )
            logger.info("Registered model %s (%s)", model_id, model_type)

            # --- 2. Predictions ---
            sub_path = Path("outputs/predictions/submission.csv")
            if sub_path.exists():
                sub = pd.read_csv(sub_path)
                rows = []
                for _, r in sub.iterrows():
                    rows.append((
                        r["account_id"],
                        float(r["is_mule"]),
                        int(r["is_mule"] >= 0.5),
                        0.5,
                        r.get("suspicious_start") if pd.notna(r.get("suspicious_start")) else None,
                        r.get("suspicious_end") if pd.notna(r.get("suspicious_end")) else None,
                        "v1",
                    ))
                conn.executemany(
                    """INSERT OR REPLACE INTO predictions
                       (account_id, probability, prediction, threshold,
                        suspicious_start, suspicious_end, model_version)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                logger.info("Seeded %d predictions", len(rows))

            # --- 3. Features ---
            rows = []
            for acct_id in features.index:
                feat_dict = features.loc[acct_id].to_dict()
                # Convert numpy types to Python types for JSON serialization
                feat_json = json.dumps({k: float(v) for k, v in feat_dict.items()})
                rows.append((acct_id, feat_json))
            conn.executemany(
                """INSERT OR REPLACE INTO features (account_id, feature_json)
                   VALUES (?, ?)""",
                rows,
            )
            logger.info("Seeded %d feature vectors", len(rows))

            # --- 4. Explanations (from SHAP if available) ---
            expl_path = Path("outputs/shap_values/explanations.json")
            if expl_path.exists():
                with open(expl_path) as f:
                    explanations = json.load(f)
                rows = []
                for acct_id, expl in explanations.items():
                    rows.append((
                        acct_id,
                        json.dumps(expl.get("top_features", {})),
                        json.dumps(list(expl.get("top_features", {}).keys())[:5]),
                        expl.get("natural_language", ""),
                    ))
                conn.executemany(
                    """INSERT OR REPLACE INTO explanations
                       (account_id, shap_values_json, top_features_json, natural_language)
                       VALUES (?, ?, ?, ?)""",
                    rows,
                )
                logger.info("Seeded %d explanations", len(rows))

            # --- 5. Benchmark results ---
            # Insert the best model's info as a benchmark entry
            bench_id = hashlib.md5(f"bench_{model_type}".encode()).hexdigest()[:8]
            conn.execute(
                """INSERT OR REPLACE INTO benchmark_results
                   (benchmark_id, model_type, metrics_json)
                   VALUES (?, ?, ?)""",
                (bench_id, model_type, json.dumps({"model": model_type, "version": "v1"})),
            )
            logger.info("Seeded benchmark entry for %s", model_type)

        conn.commit()
        logger.info("Database seeding complete at %s", db_path)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="store_true", help="Seed with model artifacts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.seed:
        seed_database()
    else:
        init_database()
    print(f"Database ready at {DB_PATH}")
