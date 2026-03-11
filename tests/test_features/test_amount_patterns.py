"""Tests for amount pattern feature generator."""
import pytest
import numpy as np
import pandas as pd
from src.features.amount_patterns import AmountPatternFeatureGenerator

CUTOFF = pd.Timestamp('2025-01-31')


def _make_txn(account_id, amounts, date='2025-01-15'):
    return pd.DataFrame({
        'account_id': [account_id] * len(amounts),
        'transaction_date': pd.to_datetime([date] * len(amounts)),
        'transaction_amount': amounts,
        'is_credit': [1] * len(amounts),
        'counterparty_id': ['CP'] * len(amounts),
    })


def _compute(amounts, acc='ACC1'):
    txn = _make_txn(acc, amounts)
    profile = pd.DataFrame({'account_id': [acc]})
    gen = AmountPatternFeatureGenerator()
    return gen.compute(txn, profile, cutoff_date=CUTOFF).loc[acc]


def test_all_round_amounts():
    """amounts=[1000, 5000, 50000] → round_amount_ratio=1.0."""
    row = _compute([1000.0, 5000.0, 50000.0])
    assert row['round_amount_ratio'] == pytest.approx(1.0)


def test_not_all_round():
    """1234 is not round → ratio < 1."""
    row = _compute([1000.0, 1234.0])
    assert row['round_amount_ratio'] == pytest.approx(0.5)


def test_structuring_detection():
    """[48000, 49000, 49500, 1000] → 3/4 = 0.75 in [45000,49999]."""
    row = _compute([48000.0, 49000.0, 49500.0, 1000.0])
    assert row['structuring_score'] == pytest.approx(0.75)


def test_no_structuring():
    """amounts=[1000, 100000] → structuring_score=0."""
    row = _compute([1000.0, 100000.0])
    assert row['structuring_score'] == pytest.approx(0.0)


def test_structuring_broad():
    """40000 is in broad range [40000,49999]."""
    row = _compute([40000.0, 100.0])
    assert row['structuring_score_broad'] == pytest.approx(0.5)


def test_entropy_uniform_high():
    """Uniform random amounts → entropy > 2."""
    rng = np.random.default_rng(42)
    amounts = rng.uniform(100, 100000, size=200).tolist()
    row = _compute(amounts)
    assert row['amount_entropy'] > 2.0


def test_gini_equal():
    """All same amount → Gini coefficient ≈ 0."""
    row = _compute([5000.0, 5000.0, 5000.0, 5000.0, 5000.0])
    assert row['amount_concentration'] == pytest.approx(0.0, abs=1e-6)


def test_pct_above_10k():
    """2/4 amounts > 10000 → pct = 0.5."""
    row = _compute([5000.0, 9999.0, 10001.0, 50000.0])
    assert row['pct_above_10k'] == pytest.approx(0.5)


def test_single_txn_no_dist_features():
    """Single transaction → skewness and kurtosis = 0 (not enough data)."""
    row = _compute([50000.0])
    assert row['amount_skewness'] == 0.0
    assert row['amount_kurtosis'] == 0.0


def test_feature_count():
    assert len(AmountPatternFeatureGenerator().get_feature_names()) == 8
