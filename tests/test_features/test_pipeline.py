import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline

CUTOFF = pd.Timestamp('2025-06-30')


def _make_synthetic_data():
    rng = np.random.default_rng(42)
    accounts = [f'ACC{i:03d}' for i in range(5)]
    n_txns = 50
    txn = pd.DataFrame({
        'account_id': rng.choice(accounts, n_txns),
        'transaction_date': pd.date_range('2025-05-01', periods=n_txns, freq='12h'),
        'transaction_amount': rng.uniform(100, 100000, n_txns).astype('float32'),
        'is_credit': rng.choice([0, 1], n_txns),
        'counterparty_id': rng.choice([f'CP{i}' for i in range(10)], n_txns),
        'balance_after': rng.uniform(10000, 500000, n_txns).astype('float32'),
    })
    profile = pd.DataFrame({
        'account_id': accounts,
        'declared_income': [50000.0] * 5,
        'account_age_days': [365, 30, 200, 500, 90],
        'current_balance': [100000.0] * 5,
        'account_type': ['savings', 'current', 'savings', 'current', 'savings'],
    })
    return txn, profile


class TestFeaturePipeline:
    def test_pipeline_runs_without_error(self):
        txn, profile = _make_synthetic_data()
        pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
        result = pipeline.run(txn, profile)
        assert len(result) == 5

    def test_pipeline_output_column_count(self):
        txn, profile = _make_synthetic_data()
        pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
        result = pipeline.run(txn, profile)
        # Total: 10+8+8+7+5+4+10+5 = 57
        assert result.shape[1] == 57, f"Expected 57 columns, got {result.shape[1]}: {sorted(result.columns.tolist())}"

    def test_pipeline_no_nulls(self):
        txn, profile = _make_synthetic_data()
        pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
        result = pipeline.run(txn, profile)
        assert result.isna().sum().sum() == 0

    def test_pipeline_no_inf(self):
        txn, profile = _make_synthetic_data()
        pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
        result = pipeline.run(txn, profile)
        assert not np.isinf(result.values).any()
