import pandas as pd
import pytest

from src.features.velocity import VelocityFeatureGenerator


def _make_txns(account_id, dates, amounts=None):
    n = len(dates)
    if amounts is None:
        amounts = [1000.0] * n
    return pd.DataFrame({
        'account_id': [account_id] * n,
        'transaction_date': pd.to_datetime(dates),
        'transaction_amount': amounts,
    })


CUTOFF = pd.Timestamp('2025-06-30')


class TestVelocityFeatures:
    def test_count_windows(self):
        # 5 txns in last 7 days
        dates = pd.date_range('2025-06-24', periods=5, freq='D')
        txn = _make_txns('A1', dates)
        gen = VelocityFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'txn_count_7d'] == 5

    def test_velocity_acceleration(self):
        # Burst: 20 txns in 7d, 20 txns in 30d => acceleration = 20 / (20/4) = 4.0
        dates_recent = pd.date_range('2025-06-24', periods=20, freq='6h')
        dates_old = pd.date_range('2025-06-05', periods=0, freq='D')  # none outside 7d
        txn = _make_txns('A1', dates_recent.tolist())
        gen = VelocityFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'velocity_acceleration'] > 2.0

    def test_empty_account(self):
        txn = pd.DataFrame({
            'account_id': ['A1'],
            'transaction_date': [pd.Timestamp('2024-01-01')],
            'transaction_amount': [100.0],
        })
        profile = pd.DataFrame({'account_id': ['A1', 'A2']})
        gen = VelocityFeatureGenerator()
        result = gen.compute(txn, profile=profile, cutoff_date=CUTOFF)
        # A2 has no transactions
        assert result.at['A2', 'txn_count_7d'] == 0
        assert result.at['A2', 'txn_count_30d'] == 0

    def test_no_future_leakage(self):
        dates = [pd.Timestamp('2025-06-29'), pd.Timestamp('2025-07-05')]
        txn = _make_txns('A1', dates)
        gen = VelocityFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        # Only 1 txn before cutoff
        assert result.at['A1', 'txn_count_7d'] == 1
