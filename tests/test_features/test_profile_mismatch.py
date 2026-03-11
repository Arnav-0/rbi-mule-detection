"""Tests for profile mismatch and KYC behavioral feature generators."""
import pytest
import pandas as pd
from src.features.profile_mismatch import ProfileMismatchFeatureGenerator
from src.features.kyc_behavioral import KYCBehavioralFeatureGenerator

CUTOFF = pd.Timestamp('2025-01-31')


def _make_txn(account_id, amounts, dates=None, is_credits=None):
    n = len(amounts)
    if dates is None:
        dates = ['2025-01-15'] * n
    if is_credits is None:
        is_credits = [1] * n
    return pd.DataFrame({
        'account_id': [account_id] * n,
        'transaction_date': pd.to_datetime(dates),
        'transaction_amount': [float(a) for a in amounts],
        'is_credit': is_credits,
        'counterparty_id': ['CP'] * n,
        'balance_after': [float(a) * 1.1 for a in amounts],
    })


def _make_velocity(acc_id, sum_30=0, cnt_30=0, mean_30=0):
    return pd.DataFrame({
        'txn_amount_sum_30d': [sum_30],
        'txn_count_30d': [cnt_30],
        'txn_amount_mean_30d': [mean_30],
    }, index=pd.Index([acc_id], name='account_id'))


class TestProfileMismatch:
    def test_high_volume_vs_income(self):
        """High txn volume vs low income → large ratio."""
        txn = _make_txn('ACC1', [100000.0] * 5)
        profile = pd.DataFrame({
            'account_id': ['ACC1'],
            'declared_income': [10000.0],
            'account_age_days': [365.0],
            'current_balance': [5000.0],
            'account_type': ['savings'],
        })
        vel = _make_velocity('ACC1', sum_30=500000, cnt_30=5, mean_30=100000)
        gen = ProfileMismatchFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF, velocity_features=vel)
        assert result.loc['ACC1', 'txn_volume_vs_income'] == pytest.approx(50.0)

    def test_product_mismatch_savings_high_value(self):
        """Savings account with mean txn > 50k → mismatch=1."""
        txn = _make_txn('ACC1', [60000.0] * 3)
        profile = pd.DataFrame({
            'account_id': ['ACC1'],
            'declared_income': [100000.0],
            'account_age_days': [365.0],
            'current_balance': [50000.0],
            'account_type': ['savings'],
        })
        vel = _make_velocity('ACC1', sum_30=180000, cnt_30=3, mean_30=60000)
        gen = ProfileMismatchFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF, velocity_features=vel)
        assert result.loc['ACC1', 'product_txn_mismatch'] == 1.0

    def test_product_no_mismatch_current_account(self):
        """Current account with high txn → no mismatch."""
        txn = _make_txn('ACC1', [60000.0] * 3)
        profile = pd.DataFrame({
            'account_id': ['ACC1'],
            'declared_income': [200000.0],
            'account_age_days': [365.0],
            'current_balance': [100000.0],
            'account_type': ['current'],
        })
        vel = _make_velocity('ACC1', sum_30=180000, cnt_30=3, mean_30=60000)
        gen = ProfileMismatchFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF, velocity_features=vel)
        assert result.loc['ACC1', 'product_txn_mismatch'] == 0.0

    def test_balance_volatility(self):
        """Varying balance_after → balance_volatility > 0."""
        amounts = [1000, 50000, 500, 80000, 2000]
        txn = _make_txn('ACC1', amounts,
                        dates=['2025-01-10', '2025-01-11', '2025-01-12', '2025-01-13', '2025-01-14'])
        txn['balance_after'] = [1000.0, 51000.0, 1500.0, 81500.0, 3500.0]
        profile = pd.DataFrame({
            'account_id': ['ACC1'],
            'declared_income': [100000.0],
            'account_age_days': [365.0],
            'current_balance': [3500.0],
            'account_type': ['savings'],
        })
        vel = _make_velocity('ACC1', sum_30=133500, cnt_30=5, mean_30=26700)
        gen = ProfileMismatchFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF, velocity_features=vel)
        assert result.loc['ACC1', 'balance_volatility'] > 0.0

    def test_no_velocity_features_defaults_zero(self):
        """Without velocity features → ratios are 0."""
        txn = _make_txn('ACC1', [1000.0])
        profile = pd.DataFrame({
            'account_id': ['ACC1'],
            'declared_income': [50000.0],
            'account_age_days': [200.0],
            'current_balance': [10000.0],
            'account_type': ['savings'],
        })
        gen = ProfileMismatchFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF, velocity_features=None)
        assert result.loc['ACC1', 'txn_volume_vs_income'] == pytest.approx(0.0)

    def test_feature_count(self):
        assert len(ProfileMismatchFeatureGenerator().get_feature_names()) == 5


class TestKYCBehavioral:
    def test_kyc_completeness_all_present(self):
        """All KYC fields present → completeness = 1.0."""
        profile = pd.DataFrame({
            'account_id': ['ACC1'],
            'pan_number': ['ABCDE1234F'],
            'aadhaar_number': ['123456789012'],
            'address': ['Some Street'],
            'email': ['test@example.com'],
            'mobile_number': ['9876543210'],
            'dob': ['1990-01-01'],
            'photo_id': ['DL123'],
        })
        txn = _make_txn('ACC1', [1000.0])
        gen = KYCBehavioralFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF)
        assert result.loc['ACC1', 'kyc_completeness'] == pytest.approx(1.0)

    def test_kyc_completeness_none(self):
        """No KYC fields → completeness = 0."""
        profile = pd.DataFrame({'account_id': ['ACC1']})
        txn = _make_txn('ACC1', [1000.0])
        gen = KYCBehavioralFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF)
        assert result.loc['ACC1', 'kyc_completeness'] == pytest.approx(0.0)

    def test_linked_account_count(self):
        """2 accounts under same customer → linked_count=2."""
        profile = pd.DataFrame({
            'account_id': ['ACC1', 'ACC2'],
            'customer_id': ['CUST1', 'CUST1'],
        })
        txn = _make_txn('ACC1', [1000.0])
        gen = KYCBehavioralFeatureGenerator()
        result = gen.compute(txn, profile, CUTOFF)
        assert result.loc['ACC1', 'linked_account_count'] == 2.0

    def test_feature_count(self):
        assert len(KYCBehavioralFeatureGenerator().get_feature_names()) == 4
