import numpy as np
import pandas as pd

from src.features.base import BaseFeatureGenerator


class ProfileMismatchFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('profile_mismatch', 'profile_mismatch')

    def get_feature_names(self) -> list[str]:
        return [
            'txn_volume_vs_income', 'account_age_vs_activity',
            'avg_txn_vs_balance', 'product_txn_mismatch', 'balance_volatility',
        ]

    def compute(self, txn, profile=None, cutoff_date=None, **kwargs):
        velocity_features = kwargs.get('velocity_features')
        if cutoff_date is None:
            cutoff_date = pd.Timestamp('2025-06-30')
        cutoff_date = pd.Timestamp(cutoff_date)

        txn = txn.copy()
        txn['transaction_date'] = pd.to_datetime(txn['transaction_date'])
        txn_valid = txn[txn['transaction_date'] <= cutoff_date]

        all_accounts = (
            profile['account_id'].unique() if profile is not None and 'account_id' in profile.columns
            else txn_valid['account_id'].unique()
        )
        result = pd.DataFrame(0.0, index=pd.Index(all_accounts, name='account_id'),
                              columns=self.get_feature_names())

        if profile is not None and 'account_id' in profile.columns:
            prof = profile.set_index('account_id')
        else:
            prof = pd.DataFrame(index=pd.Index(all_accounts, name='account_id'))

        # Velocity features (from prior stage)
        vel = velocity_features if velocity_features is not None else pd.DataFrame(
            0, index=pd.Index(all_accounts, name='account_id'),
            columns=['txn_amount_sum_30d', 'txn_count_30d', 'txn_amount_mean_30d']
        )

        sum_30d = vel['txn_amount_sum_30d'].reindex(all_accounts, fill_value=0)
        count_30d = vel['txn_count_30d'].reindex(all_accounts, fill_value=0)
        mean_30d = vel['txn_amount_mean_30d'].reindex(all_accounts, fill_value=0)

        # Balance (use avg_balance)
        balance = pd.to_numeric(prof['avg_balance'], errors='coerce').reindex(all_accounts, fill_value=0) if 'avg_balance' in prof.columns else pd.Series(0, index=all_accounts)

        # Account age
        if 'account_opening_date' in prof.columns:
            open_dates = pd.to_datetime(prof['account_opening_date'], errors='coerce')
            age_days = (cutoff_date - open_dates).dt.days.fillna(0).reindex(all_accounts, fill_value=0)
        else:
            age_days = pd.Series(0, index=all_accounts)

        # Product family
        product_family = prof['product_family'].reindex(all_accounts, fill_value='') if 'product_family' in prof.columns else pd.Series('', index=all_accounts)

        result['txn_volume_vs_income'] = sum_30d / balance.abs().clip(lower=1)
        result['account_age_vs_activity'] = count_30d / age_days.clip(lower=1)
        result['avg_txn_vs_balance'] = mean_30d / balance.abs().clip(lower=1)
        result['product_txn_mismatch'] = ((product_family.str.upper() == 'S') & (mean_30d > 50000)).astype(float)

        # Balance volatility
        if 'daily_avg_balance' in prof.columns and 'monthly_avg_balance' in prof.columns:
            daily = pd.to_numeric(prof['daily_avg_balance'], errors='coerce').fillna(0).reindex(all_accounts, fill_value=0)
            monthly = pd.to_numeric(prof['monthly_avg_balance'], errors='coerce').fillna(0).reindex(all_accounts, fill_value=0)
            result['balance_volatility'] = (daily - monthly).abs() / monthly.abs().clip(lower=1)

        result = result.fillna(0).replace([np.inf, -np.inf], 0)
        self.validate_output(result)
        return result
