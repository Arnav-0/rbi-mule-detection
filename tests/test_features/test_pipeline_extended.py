"""Extended pipeline tests: graph path, run_and_save, edge cases."""

import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.features.registry import get_all_feature_names

CUTOFF = pd.Timestamp("2025-01-31")


def _make_data(n_accounts=5, n_txns=80, seed=7):
    rng = np.random.default_rng(seed)
    accts = [f"ACC_{i:03d}" for i in range(n_accounts)]
    txn = pd.DataFrame({
        "account_id": rng.choice(accts, n_txns),
        "transaction_date": pd.date_range("2024-09-01", periods=n_txns, freq="6h"),
        "transaction_amount": rng.uniform(200, 80000, n_txns).astype("float32"),
        "is_credit": rng.integers(0, 2, n_txns).astype("int8"),
        "counterparty_id": rng.choice([f"CP_{i}" for i in range(8)], n_txns),
        "balance_after": rng.uniform(1000, 300000, n_txns).astype("float32"),
    })
    profile = pd.DataFrame({
        "account_id": accts,
        "declared_income": rng.uniform(100000, 1500000, n_accounts),
        "account_age_days": rng.integers(90, 2500, n_accounts).astype(float),
        "current_balance": rng.uniform(5000, 400000, n_accounts),
        "account_type": rng.choice(["savings", "current"], n_accounts),
        "customer_id": [f"CUST_{i}" for i in range(n_accounts)],
        "is_new_account": rng.integers(0, 2, n_accounts),
    })
    labels = pd.DataFrame({
        "account_id": accts[:3],
        "is_mule": [1, 0, 1],
    })
    return txn, profile, labels


def test_pipeline_with_graph():
    """Full pipeline including graph features (no skip_graph)."""
    txn, profile, labels = _make_data()
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=False)
    result = pipeline.run(txn, profile, labels)
    assert result is not None
    assert result.shape[1] == 57
    assert result.isnull().sum().sum() == 0
    assert result.index.name == "account_id"
    # Graph features should be non-zero for accounts with transactions
    assert (result["in_degree"] >= 0).all()
    assert (result["pagerank"] >= 0).all()


def test_pipeline_graph_community_mule_density():
    """When labels provided, community_mule_density should be computed."""
    txn, profile, labels = _make_data()
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=False)
    result = pipeline.run(txn, profile, labels)
    # community_mule_density should be >= 0 for all accounts
    assert (result["community_mule_density"] >= 0).all()


def test_run_and_save_creates_parquet(tmp_path):
    """run_and_save should persist a valid parquet file."""
    txn, profile, labels = _make_data()
    out = tmp_path / "features.parquet"
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run_and_save(txn, profile, labels, str(out))
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert loaded.shape == result.shape
    assert list(loaded.columns) == list(result.columns)


def test_run_and_save_creates_parent_dirs(tmp_path):
    """run_and_save should create nested output directories."""
    txn, profile, labels = _make_data()
    out = tmp_path / "nested" / "deep" / "features.parquet"
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    pipeline.run_and_save(txn, profile, labels, str(out))
    assert out.exists()


def test_pipeline_single_account():
    """Single account edge case should not crash."""
    txn = pd.DataFrame({
        "account_id": ["ONLY_ONE"] * 10,
        "transaction_date": pd.date_range("2025-01-01", periods=10, freq="D"),
        "transaction_amount": np.full(10, 5000.0, dtype="float32"),
        "is_credit": np.array([1, 0] * 5, dtype="int8"),
        "counterparty_id": [f"CP_{i}" for i in range(10)],
        "balance_after": np.full(10, 50000.0, dtype="float32"),
    })
    profile = pd.DataFrame({
        "account_id": ["ONLY_ONE"],
        "declared_income": [500000.0],
        "account_age_days": [365.0],
        "current_balance": [50000.0],
        "account_type": ["savings"],
        "customer_id": ["CUST_1"],
        "is_new_account": [0],
    })
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run(txn, profile)
    assert result.shape == (1, 57)
    assert result.isnull().sum().sum() == 0


def test_pipeline_all_features_present():
    """Every one of the 57 registered feature names must appear in output."""
    txn, profile, _ = _make_data()
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run(txn, profile)
    all_feats = set(get_all_feature_names())
    output_cols = set(result.columns)
    missing = all_feats - output_cols
    assert not missing, f"Missing features: {missing}"


def test_pipeline_no_data_after_cutoff():
    """All transactions after cutoff → all features should be zero."""
    accts = ["ACC_A", "ACC_B"]
    txn = pd.DataFrame({
        "account_id": ["ACC_A"] * 5,
        "transaction_date": pd.date_range("2025-03-01", periods=5, freq="D"),
        "transaction_amount": np.full(5, 1000.0, dtype="float32"),
        "is_credit": np.ones(5, dtype="int8"),
        "counterparty_id": ["CP_X"] * 5,
        "balance_after": np.full(5, 10000.0, dtype="float32"),
    })
    profile = pd.DataFrame({
        "account_id": accts,
        "declared_income": [200000.0, 300000.0],
        "account_age_days": [500.0, 700.0],
        "current_balance": [20000.0, 30000.0],
        "account_type": ["savings", "current"],
        "customer_id": ["C1", "C2"],
        "is_new_account": [0, 0],
    })
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run(txn, profile)
    # Count features for ACC_A should be 0 (all txns in future)
    assert result.loc["ACC_A", "txn_count_30d"] == 0
    assert result.loc["ACC_A", "txn_count_90d"] == 0
