import numpy as np
import pandas as pd

from src.features.base import BaseFeatureGenerator


class TemporalFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('temporal', 'temporal')

    def get_feature_names(self) -> list[str]:
        return [
            'dormancy_days', 'max_gap_days', 'burst_after_dormancy',
            'unusual_hour_ratio', 'weekend_ratio', 'night_weekend_combo',
            'monthly_txn_cv', 'days_to_first_txn',
        ]

    def compute(self, txn, profile=None, cutoff_date=None, **kwargs):
        if cutoff_date is None:
            cutoff_date = pd.Timestamp('2025-06-30')
        cutoff_date = pd.Timestamp(cutoff_date)

        txn = txn.copy()
        txn['transaction_date'] = pd.to_datetime(txn['transaction_date'])
        txn_valid = txn[txn['transaction_date'] <= cutoff_date].copy()

        if 'transaction_hour' not in txn_valid.columns:
            txn_valid['transaction_hour'] = txn_valid['transaction_date'].dt.hour
        if 'is_weekend' not in txn_valid.columns:
            txn_valid['is_weekend'] = txn_valid['transaction_date'].dt.dayofweek.isin([5, 6]).astype(int)
        if 'is_night' not in txn_valid.columns:
            txn_valid['is_night'] = txn_valid['transaction_hour'].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)

        all_accounts = (
            profile['account_id'].unique() if profile is not None and 'account_id' in profile.columns
            else txn_valid['account_id'].unique()
        )
        result = pd.DataFrame(0.0, index=pd.Index(all_accounts, name='account_id'),
                              columns=self.get_feature_names())

        grp = txn_valid.groupby('account_id')

        # Vectorized: unusual_hour_ratio, weekend_ratio, night_weekend_combo
        result['unusual_hour_ratio'] = grp['is_night'].mean().reindex(all_accounts, fill_value=0)
        result['weekend_ratio'] = grp['is_weekend'].mean().reindex(all_accounts, fill_value=0)
        result['night_weekend_combo'] = result['unusual_hour_ratio'] * result['weekend_ratio']

        # Max gap and dormancy — need sorted dates per account
        txn_sorted = txn_valid[['account_id', 'transaction_date']].sort_values(['account_id', 'transaction_date'])
        txn_sorted['prev_date'] = txn_sorted.groupby('account_id')['transaction_date'].shift(1)
        txn_sorted['gap_days'] = (txn_sorted['transaction_date'] - txn_sorted['prev_date']).dt.days

        max_gaps = txn_sorted.groupby('account_id')['gap_days'].max().reindex(all_accounts, fill_value=0)
        result['max_gap_days'] = max_gaps
        result['dormancy_days'] = max_gaps.where(max_gaps > 90, 0)

        # Burst after dormancy: dormancy > 0 and >10 txns in last 30d
        window_30d_start = cutoff_date - pd.Timedelta(days=30)
        recent_txn = txn_valid[txn_valid['transaction_date'] > window_30d_start]
        recent_counts = recent_txn.groupby('account_id').size().reindex(all_accounts, fill_value=0)
        result['burst_after_dormancy'] = ((result['dormancy_days'] > 0) & (recent_counts > 10)).astype(float)

        # Monthly CV
        txn_valid['year_month'] = txn_valid['transaction_date'].dt.to_period('M')
        monthly_counts = txn_valid.groupby(['account_id', 'year_month']).size().unstack(fill_value=0)
        monthly_mean = monthly_counts.mean(axis=1)
        monthly_std = monthly_counts.std(axis=1)
        cv = (monthly_std / monthly_mean.clip(lower=0.01)).reindex(all_accounts, fill_value=0)
        result['monthly_txn_cv'] = cv

        # Days to first txn from account opening
        open_col = None
        if profile is not None:
            for col_name in ['account_opening_date', 'account_open_date']:
                if col_name in profile.columns:
                    open_col = col_name
                    break
        if open_col:
            open_dates = profile.set_index('account_id')[open_col]
            open_dates = pd.to_datetime(open_dates, errors='coerce')
            first_txn = grp['transaction_date'].min().reindex(all_accounts)
            days_diff = (first_txn - open_dates.reindex(all_accounts)).dt.days
            result['days_to_first_txn'] = days_diff.fillna(0)

        result = result.fillna(0).replace([np.inf, -np.inf], 0)
        self.validate_output(result)
        return result
