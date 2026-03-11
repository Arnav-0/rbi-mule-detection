import pandas as pd
import pytest

from src.features.profile_mismatch import ProfileMismatchFeatureGenerator
from src.features.velocity import VelocityFeatureGenerator

CUTOFF = pd.Timestamp('2025-06-30')


class TestProfileMismatchFeatures:
    def _make_data(self):
        txn = pd.DataFrame({
            'account_id': ['A1'] * 10,
            'transaction_date': pd.date_range('2025-06-01', periods=10, freq='D'),
            'transaction_amount': [60000.0] * 10,
        })
        profile = pd.DataFrame({
            'account_id': ['A1'],
            'avg_balance': [50000.0],
            'account_opening_date': ['2024-06-30'],
            'product_family': ['S'],
            'daily_avg_balance': [120000.0],
            'monthly_avg_balance': [100000.0],
        })
        return txn, profile

    def test_txn_volume_vs_income(self):
        txn, profile = self._make_data()
        vel = VelocityFeatureGenerator().compute(txn, profile, CUTOFF)
        gen = ProfileMismatchFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF, velocity_features=vel)
        # sum_30d = 600000, avg_balance = 50000 => ratio = 12
        assert result.at['A1', 'txn_volume_vs_income'] == 12.0

    def test_product_txn_mismatch(self):
        txn, profile = self._make_data()
        vel = VelocityFeatureGenerator().compute(txn, profile, CUTOFF)
        gen = ProfileMismatchFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF, velocity_features=vel)
        # product_family='S' + mean > 50000 => mismatch = 1
        assert result.at['A1', 'product_txn_mismatch'] == 1

    def test_balance_volatility(self):
        txn, profile = self._make_data()
        vel = VelocityFeatureGenerator().compute(txn, profile, CUTOFF)
        gen = ProfileMismatchFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF, velocity_features=vel)
        # daily=120000, monthly=100000 => volatility = 20000/100000 = 0.2
        assert result.at['A1', 'balance_volatility'] > 0

    def test_no_profile(self):
        txn = pd.DataFrame({
            'account_id': ['A1'] * 3,
            'transaction_date': pd.date_range('2025-06-01', periods=3, freq='D'),
            'transaction_amount': [1000.0] * 3,
        })
        gen = ProfileMismatchFeatureGenerator()
        result = gen.compute(txn, cutoff_date=CUTOFF)
        assert result.at['A1', 'txn_volume_vs_income'] == 0
