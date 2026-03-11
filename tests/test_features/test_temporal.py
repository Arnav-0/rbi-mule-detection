import pandas as pd
import pytest

from src.features.temporal import TemporalFeatureGenerator

CUTOFF = pd.Timestamp('2025-06-30')


def _make_txns(account_id, dates, hours=None):
    n = len(dates)
    dates = pd.to_datetime(dates)
    if hours is not None:
        dates = [d.replace(hour=h) for d, h in zip(dates, hours)]
    return pd.DataFrame({
        'account_id': [account_id] * n,
        'transaction_date': dates,
        'transaction_amount': [1000.0] * n,
    })


class TestTemporalFeatures:
    def test_dormancy_gap(self):
        # Txn on Jan 1, then Apr 11 (100-day gap)
        dates = ['2025-01-01', '2025-04-11', '2025-04-12']
        txn = _make_txns('A1', dates)
        gen = TemporalFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'dormancy_days'] == 100
        assert result.at['A1', 'max_gap_days'] == 100

    def test_no_dormancy(self):
        dates = pd.date_range('2025-06-01', periods=30, freq='D')
        txn = _make_txns('A1', dates)
        gen = TemporalFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'dormancy_days'] == 0

    def test_burst_detection(self):
        # Big gap then 20 txns in last 30 days
        dates = ['2025-01-01'] + pd.date_range('2025-06-05', periods=20, freq='D').tolist()
        txn = _make_txns('A1', dates)
        gen = TemporalFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'burst_after_dormancy'] == 1

    def test_unusual_hours(self):
        dates = pd.date_range('2025-06-01', periods=5, freq='D')
        hours = [2, 2, 2, 2, 2]
        txn = _make_txns('A1', dates, hours=hours)
        gen = TemporalFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'unusual_hour_ratio'] == 1.0

    def test_weekend_ratio(self):
        # Create 10 txns, 3 on weekends
        # 2025-06-02=Mon, 06-07=Sat, 06-08=Sun, 06-14=Sat
        dates = ['2025-06-02', '2025-06-03', '2025-06-04', '2025-06-05',
                 '2025-06-06', '2025-06-09', '2025-06-10',
                 '2025-06-07', '2025-06-08', '2025-06-14']
        txn = _make_txns('A1', dates)
        gen = TemporalFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert abs(result.at['A1', 'weekend_ratio'] - 0.3) < 0.01
