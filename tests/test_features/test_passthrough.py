import pandas as pd
import pytest

from src.features.passthrough import PassThroughFeatureGenerator

CUTOFF = pd.Timestamp('2025-06-30')


class TestPassThroughFeatures:
    def test_rapid_passthrough(self):
        txn = pd.DataFrame({
            'account_id': ['A1', 'A1'],
            'transaction_date': [
                pd.Timestamp('2025-06-15 10:00:00'),
                pd.Timestamp('2025-06-15 10:30:00'),
            ],
            'transaction_amount': [50000.0, 50000.0],
            'is_credit': [1, 0],
        })
        gen = PassThroughFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert abs(result.at['A1', 'credit_debit_time_delta_median'] - 0.5) < 0.01
        assert result.at['A1', 'matched_amount_ratio'] == 1.0

    def test_no_passthrough(self):
        txn = pd.DataFrame({
            'account_id': ['A1', 'A1'],
            'transaction_date': [
                pd.Timestamp('2025-06-15 10:00:00'),
                pd.Timestamp('2025-06-16 10:00:00'),
            ],
            'transaction_amount': [50000.0, 50000.0],
            'is_credit': [1, 1],
        })
        gen = PassThroughFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'credit_debit_time_delta_median'] == 999
        assert result.at['A1', 'matched_amount_ratio'] == 0

    def test_net_flow_balanced(self):
        txn = pd.DataFrame({
            'account_id': ['A1'] * 4,
            'transaction_date': pd.date_range('2025-06-10', periods=4, freq='D'),
            'transaction_amount': [10000.0, 10000.0, 10000.0, 10000.0],
            'is_credit': [1, 0, 1, 0],
        })
        gen = PassThroughFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert abs(result.at['A1', 'net_flow_ratio'] - 1.0) < 0.01

    def test_symmetry_perfect(self):
        txn = pd.DataFrame({
            'account_id': ['A1'] * 10,
            'transaction_date': pd.date_range('2025-06-01', periods=10, freq='D'),
            'transaction_amount': [5000.0] * 10,
            'is_credit': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        })
        gen = PassThroughFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'credit_debit_symmetry'] == 1.0
