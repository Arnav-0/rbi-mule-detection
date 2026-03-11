"""Tests for pass-through feature generator."""
import pytest
import pandas as pd
from src.features.passthrough import PassThroughFeatureGenerator

CUTOFF = pd.Timestamp('2025-01-31')


def _make_txn(records):
    """records: list of (account_id, datetime_str, amount, is_credit)."""
    return pd.DataFrame({
        'account_id': [r[0] for r in records],
        'transaction_date': pd.to_datetime([r[1] for r in records]),
        'transaction_amount': [float(r[2]) for r in records],
        'is_credit': [int(r[3]) for r in records],
        'counterparty_id': ['CP'] * len(records),
    })


def _compute(records, acc='ACC1'):
    txn = _make_txn(records)
    profile = pd.DataFrame({'account_id': [acc]})
    gen = PassThroughFeatureGenerator()
    return gen.compute(txn, profile, cutoff_date=CUTOFF).loc[acc]


def test_rapid_passthrough():
    """Credit at T, debit at T+30min same amount → delta≈0.5h, matched_ratio=1.0."""
    t0 = '2025-01-15 10:00:00'
    t1 = '2025-01-15 10:30:00'
    records = [
        ('ACC1', t0, 50000, 1),  # credit
        ('ACC1', t1, 50000, 0),  # debit 30 min later
    ]
    row = _compute(records)
    assert row['credit_debit_time_delta_min'] == pytest.approx(0.5, abs=0.1)
    assert row['matched_amount_ratio'] == pytest.approx(1.0)
    assert row['rapid_turnover_score'] == pytest.approx(1.0)


def test_no_passthrough_only_credits():
    """Only credits → delta=999, matched_ratio=0."""
    records = [('ACC1', '2025-01-10', 10000, 1), ('ACC1', '2025-01-11', 20000, 1)]
    row = _compute(records)
    assert row['credit_debit_time_delta_median'] == pytest.approx(999.0)
    assert row['credit_debit_time_delta_min'] == pytest.approx(999.0)
    assert row['matched_amount_ratio'] == pytest.approx(0.0)


def test_net_flow_balanced():
    """Equal credits and debits → net_flow_ratio ≈ 1.0."""
    records = [
        ('ACC1', '2025-01-10 09:00', 50000, 1),
        ('ACC1', '2025-01-10 12:00', 50000, 0),
    ]
    row = _compute(records)
    assert row['net_flow_ratio'] == pytest.approx(1.0, abs=0.01)


def test_symmetry_perfect():
    """5 credits + 5 debits → credit_debit_symmetry=1.0."""
    records = []
    for i in range(5):
        records.append(('ACC1', f'2025-01-{10+i} 09:00', 10000, 1))
        records.append(('ACC1', f'2025-01-{10+i} 18:00', 10000, 0))
    row = _compute(records)
    assert row['credit_debit_symmetry'] == pytest.approx(1.0)


def test_symmetry_unbalanced():
    """5 credits, 0 debits → symmetry=0."""
    records = [('ACC1', f'2025-01-{10+i}', 10000, 1) for i in range(5)]
    row = _compute(records)
    assert row['credit_debit_symmetry'] == pytest.approx(0.0)


def test_max_single_day_volume():
    """Single day with 3 txns of 10k each → max_day_vol=30k."""
    records = [
        ('ACC1', '2025-01-15 09:00', 10000, 1),
        ('ACC1', '2025-01-15 12:00', 10000, 0),
        ('ACC1', '2025-01-15 18:00', 10000, 1),
        ('ACC1', '2025-01-16 10:00', 5000, 1),
    ]
    row = _compute(records)
    assert row['max_single_day_volume'] == pytest.approx(30000.0)


def test_feature_count():
    assert len(PassThroughFeatureGenerator().get_feature_names()) == 7
