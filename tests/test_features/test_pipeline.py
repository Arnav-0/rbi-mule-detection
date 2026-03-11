"""Tests for the master feature pipeline."""
import numpy as np
import pandas as pd
from src.features.pipeline import FeaturePipeline
from src.features.velocity import VelocityFeatureGenerator
from src.features.amount_patterns import AmountPatternFeatureGenerator
from src.features.temporal import TemporalFeatureGenerator
from src.features.passthrough import PassThroughFeatureGenerator
from src.features.graph_network import GraphNetworkFeatureGenerator
from src.features.profile_mismatch import ProfileMismatchFeatureGenerator
from src.features.kyc_behavioral import KYCBehavioralFeatureGenerator
from src.features.interactions import InteractionFeatureGenerator

CUTOFF = pd.Timestamp('2025-01-31')

ALL_FEATURE_NAMES = (
    VelocityFeatureGenerator().get_feature_names()
    + AmountPatternFeatureGenerator().get_feature_names()
    + TemporalFeatureGenerator().get_feature_names()
    + PassThroughFeatureGenerator().get_feature_names()
    + ProfileMismatchFeatureGenerator().get_feature_names()
    + KYCBehavioralFeatureGenerator().get_feature_names()
    + GraphNetworkFeatureGenerator().get_feature_names()
    + InteractionFeatureGenerator().get_feature_names()
)


def _make_synthetic_data(n_accounts=5, n_txns=50, seed=42):
    rng = np.random.default_rng(seed)
    accounts = [f'ACC_{i:03d}' for i in range(n_accounts)]

    txn_accounts = rng.choice(accounts, size=n_txns)
    dates = pd.date_range(start='2024-12-01', end='2025-01-30', periods=n_txns)
    amounts = rng.uniform(500, 100000, size=n_txns)
    is_credit = rng.integers(0, 2, size=n_txns)
    counterparties = rng.choice([f'CP_{i}' for i in range(10)], size=n_txns)

    txn = pd.DataFrame({
        'account_id': txn_accounts,
        'transaction_date': dates,
        'transaction_amount': amounts,
        'is_credit': is_credit,
        'counterparty_id': counterparties,
        'balance_after': rng.uniform(1000, 200000, size=n_txns),
    })

    profile = pd.DataFrame({
        'account_id': accounts,
        'declared_income': rng.uniform(50000, 500000, size=n_accounts),
        'account_age_days': rng.integers(30, 3000, size=n_accounts).astype(float),
        'current_balance': rng.uniform(1000, 200000, size=n_accounts),
        'account_type': rng.choice(['savings', 'current'], size=n_accounts),
        'customer_id': [f'CUST_{i}' for i in range(n_accounts)],
    })

    return txn, profile


def test_pipeline_runs_without_error():
    """Pipeline runs end-to-end with skip_graph=True."""
    txn, profile = _make_synthetic_data()
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run(txn, profile)
    assert result is not None
    assert len(result) > 0


def test_pipeline_output_has_correct_columns():
    """Pipeline output has all expected feature columns."""
    txn, profile = _make_synthetic_data()
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run(txn, profile)
    for feat in ALL_FEATURE_NAMES:
        assert feat in result.columns, f"Missing feature: {feat}"


def test_pipeline_no_nulls():
    """Pipeline output contains no NaN values."""
    txn, profile = _make_synthetic_data()
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run(txn, profile)
    null_counts = result.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert len(cols_with_nulls) == 0, f"Columns with nulls: {cols_with_nulls.to_dict()}"


def test_pipeline_no_inf():
    """Pipeline output contains no infinite values."""
    txn, profile = _make_synthetic_data()
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run(txn, profile)
    inf_counts = np.isinf(result.select_dtypes(include=[np.number])).sum()
    cols_with_inf = inf_counts[inf_counts > 0]
    assert len(cols_with_inf) == 0, f"Columns with inf: {cols_with_inf.to_dict()}"


def test_pipeline_index_name():
    """Pipeline output index is named 'account_id'."""
    txn, profile = _make_synthetic_data()
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run(txn, profile)
    assert result.index.name == 'account_id'


def test_pipeline_no_future_leakage():
    """Transactions after cutoff excluded from all features."""
    txn, profile = _make_synthetic_data()
    # Add future txns with extreme amounts
    future = pd.DataFrame({
        'account_id': ['ACC_000'] * 10,
        'transaction_date': pd.date_range('2025-02-01', periods=10),
        'transaction_amount': [9999999.0] * 10,
        'is_credit': [1] * 10,
        'counterparty_id': ['FUTURE_CP'] * 10,
        'balance_after': [9999999.0] * 10,
    })
    txn_with_future = pd.concat([txn, future], ignore_index=True)
    pipeline = FeaturePipeline(cutoff_date=CUTOFF, skip_graph=True)
    result = pipeline.run(txn_with_future, profile)
    # The extreme future amounts should not appear in sum_30d
    assert result.loc['ACC_000', 'txn_amount_sum_30d'] < 9999999.0
