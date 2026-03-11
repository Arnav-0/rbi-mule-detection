"""Tests for velocity feature generator."""
import pandas as pd
from src.features.velocity import VelocityFeatureGenerator


CUTOFF = pd.Timestamp('2025-01-31')


def _make_txn(account_ids, dates, amounts=None):
    n = len(account_ids)
    if amounts is None:
        amounts = [1000.0] * n
    return pd.DataFrame({
        'account_id': account_ids,
        'transaction_date': pd.to_datetime(dates, format='mixed'),
        'transaction_amount': amounts,
        'is_credit': [1] * n,
        'counterparty_id': ['CP001'] * n,
    })


def _make_profile(account_ids):
    return pd.DataFrame({'account_id': account_ids})


def test_count_windows():
    """5 txns in last 7 days → txn_count_7d = 5."""
    cutoff = pd.Timestamp('2025-01-31')
    # 5 txns within last 7 days
    dates_7d = [(cutoff - pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 6)]
    # 3 older txns (> 7 days but <= 30 days)
    dates_old = [(cutoff - pd.Timedelta(days=10 + i)).strftime('%Y-%m-%d') for i in range(3)]
    all_dates = dates_7d + dates_old
    txn = _make_txn(['ACC1'] * 8, all_dates)
    profile = _make_profile(['ACC1'])
    gen = VelocityFeatureGenerator()
    result = gen.compute(txn, profile, cutoff_date=cutoff)
    assert result.loc['ACC1', 'txn_count_7d'] == 5
    assert result.loc['ACC1', 'txn_count_30d'] == 8


def test_velocity_acceleration():
    """Burst of activity → acceleration > 2.0."""
    cutoff = pd.Timestamp('2025-01-31')
    # Use Timestamps directly to avoid mixed format issues
    # 12 txns within last 3 days (clearly inside 7d window)
    burst = [cutoff - pd.Timedelta(days=d, hours=h) for d in range(1, 4) for h in range(4)]
    # 2 older txns outside 7d but inside 30d
    old_30 = [cutoff - pd.Timedelta(days=15 + i) for i in range(2)]
    all_dates = burst + old_30
    txn = _make_txn(['ACC1'] * len(all_dates), all_dates)
    profile = _make_profile(['ACC1'])
    gen = VelocityFeatureGenerator()
    result = gen.compute(txn, profile, cutoff_date=cutoff)
    # count_7d=12, count_30d=14 → accel = 12 / max(14/4, 1) = 12/3.5 ≈ 3.4 > 2
    assert result.loc['ACC1', 'velocity_acceleration'] > 2.0


def test_empty_account():
    """Account with no transactions → all zeros."""
    cutoff = pd.Timestamp('2025-01-31')
    txn = _make_txn(['OTHER'] * 3, ['2025-01-10', '2025-01-11', '2025-01-12'])
    profile = _make_profile(['ACC_EMPTY', 'OTHER'])
    gen = VelocityFeatureGenerator()
    result = gen.compute(txn, profile, cutoff_date=cutoff)
    row = result.loc['ACC_EMPTY']
    assert row['txn_count_1d'] == 0
    assert row['txn_count_7d'] == 0
    assert row['txn_count_30d'] == 0
    assert row['txn_amount_sum_30d'] == 0


def test_no_future_leakage():
    """Transactions after cutoff must be excluded."""
    cutoff = pd.Timestamp('2025-01-31')
    # Txns after cutoff
    future_dates = ['2025-02-01', '2025-02-10', '2025-03-01']
    # Txns before cutoff
    past_dates = ['2025-01-25', '2025-01-20']
    all_dates = future_dates + past_dates
    txn = _make_txn(['ACC1'] * len(all_dates), all_dates)
    profile = _make_profile(['ACC1'])
    gen = VelocityFeatureGenerator()
    result = gen.compute(txn, profile, cutoff_date=cutoff)
    # Only 2 past txns should count (both within 30d of cutoff)
    assert result.loc['ACC1', 'txn_count_30d'] == 2
    assert result.loc['ACC1', 'txn_count_90d'] == 2


def test_feature_names():
    assert len(VelocityFeatureGenerator().get_feature_names()) == 10
