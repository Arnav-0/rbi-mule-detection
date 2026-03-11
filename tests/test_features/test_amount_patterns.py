import numpy as np
import pandas as pd
import pytest

from src.features.amount_patterns import AmountPatternFeatureGenerator

CUTOFF = pd.Timestamp('2025-06-30')


def _make_txns(account_id, amounts):
    n = len(amounts)
    return pd.DataFrame({
        'account_id': [account_id] * n,
        'transaction_date': pd.date_range('2025-06-01', periods=n, freq='D'),
        'transaction_amount': amounts,
    })


class TestAmountPatternFeatures:
    def test_all_round_amounts(self):
        txn = _make_txns('A1', [1000, 5000, 50000])
        gen = AmountPatternFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'round_amount_ratio'] == 1.0

    def test_structuring_detection(self):
        txn = _make_txns('A1', [48000, 49000, 49500, 1000])
        gen = AmountPatternFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'structuring_score'] == 0.75

    def test_no_structuring(self):
        txn = _make_txns('A1', [1000, 100000])
        gen = AmountPatternFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'structuring_score'] == 0.0

    def test_entropy_uniform_high(self):
        rng = np.random.default_rng(42)
        amounts = rng.uniform(100, 100000, size=100).tolist()
        txn = _make_txns('A1', amounts)
        gen = AmountPatternFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'amount_entropy'] > 2.0

    def test_gini_equal(self):
        txn = _make_txns('A1', [5000] * 10)
        gen = AmountPatternFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert abs(result.at['A1', 'amount_concentration']) < 0.01
