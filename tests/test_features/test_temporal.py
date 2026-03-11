"""Tests for temporal feature generator."""
import pytest
import pandas as pd
from src.features.temporal import TemporalFeatureGenerator

CUTOFF = pd.Timestamp('2025-01-31')


def _make_txn(account_id, datetimes):
    return pd.DataFrame({
        'account_id': [account_id] * len(datetimes),
        'transaction_date': pd.to_datetime(datetimes),
        'transaction_amount': [1000.0] * len(datetimes),
        'is_credit': [1] * len(datetimes),
        'counterparty_id': ['CP'] * len(datetimes),
    })


def _compute(account_id, datetimes, profile=None):
    txn = _make_txn(account_id, datetimes)
    if profile is None:
        profile = pd.DataFrame({'account_id': [account_id]})
    gen = TemporalFeatureGenerator()
    return gen.compute(txn, profile, cutoff_date=CUTOFF).loc[account_id]


def test_dormancy_gap():
    """100-day gap → dormancy_days=100, max_gap_days=100."""
    # Two txns: one 200 days before cutoff, next 100 days before cutoff
    d1 = (CUTOFF - pd.Timedelta(days=200)).strftime('%Y-%m-%d %H:%M')
    d2 = (CUTOFF - pd.Timedelta(days=100)).strftime('%Y-%m-%d %H:%M')
    d3 = (CUTOFF - pd.Timedelta(days=10)).strftime('%Y-%m-%d %H:%M')
    row = _compute('ACC1', [d1, d2, d3])
    assert row['max_gap_days'] == pytest.approx(100.0, abs=1.0)
    assert row['dormancy_days'] == pytest.approx(100.0, abs=1.0)


def test_no_dormancy():
    """Daily txns → max gap ≤ 1 → dormancy_days=0."""
    dates = [(CUTOFF - pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
    row = _compute('ACC1', dates)
    assert row['dormancy_days'] == 0.0
    assert row['max_gap_days'] <= 1.0


def test_burst_detection():
    """Large gap + many recent txns → burst_after_dormancy=1."""
    # Gap > 90 days: one txn 200 days ago
    old = [(CUTOFF - pd.Timedelta(days=200)).strftime('%Y-%m-%d')]
    # 20 recent txns in last 30 days
    recent = [(CUTOFF - pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 21)]
    row = _compute('ACC1', old + recent)
    assert row['dormancy_days'] > 90
    assert row['burst_after_dormancy'] == 1.0


def test_no_burst_without_dormancy():
    """No dormancy → burst=0 even with many recent txns."""
    dates = [(CUTOFF - pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 25)]
    row = _compute('ACC1', dates)
    assert row['burst_after_dormancy'] == 0.0


def test_unusual_hours():
    """All txns at 2AM → unusual_hour_ratio=1.0."""
    dates = [(CUTOFF - pd.Timedelta(days=i)).strftime('%Y-%m-%d 02:00:00') for i in range(1, 6)]
    row = _compute('ACC1', dates)
    assert row['unusual_hour_ratio'] == pytest.approx(1.0)


def test_daytime_hours():
    """All txns at 10AM → ratio=0."""
    dates = [(CUTOFF - pd.Timedelta(days=i)).strftime('%Y-%m-%d 10:00:00') for i in range(1, 6)]
    row = _compute('ACC1', dates)
    assert row['unusual_hour_ratio'] == pytest.approx(0.0)


def test_weekend_ratio():
    """Find exactly 3 weekends in 10 txns → ratio ≈ 0.3."""
    # Manually pick dates: 3 Saturday/Sunday and 7 weekday
    # 2025-01-04 is Saturday, 2025-01-05 Sunday
    dates = [
        '2025-01-04', '2025-01-05', '2025-01-11',  # 3 weekends
        '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',  # 5 weekdays
        '2025-01-13', '2025-01-14',  # 2 more weekdays
    ]
    row = _compute('ACC1', dates)
    assert row['weekend_ratio'] == pytest.approx(0.3, abs=0.05)


def test_feature_count():
    assert len(TemporalFeatureGenerator().get_feature_names()) == 8
