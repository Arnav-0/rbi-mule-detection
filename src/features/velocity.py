import pandas as pd

from src.features.base import BaseFeatureGenerator


class VelocityFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('velocity', 'velocity')

    def get_feature_names(self) -> list[str]:
        return [
            'txn_count_1d', 'txn_count_7d', 'txn_count_30d', 'txn_count_90d',
            'txn_amount_mean_30d', 'txn_amount_max_30d', 'txn_amount_std_30d',
            'txn_amount_sum_30d', 'velocity_acceleration', 'frequency_change_ratio',
        ]

    def compute(self, txn, profile=None, cutoff_date=None, **kwargs):
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
        result = pd.DataFrame(index=pd.Index(all_accounts, name='account_id'))

        for window in [1, 7, 30, 90]:
            window_start = cutoff_date - pd.Timedelta(days=window)
            window_txn = txn_valid[txn_valid['transaction_date'] > window_start]
            grp = window_txn.groupby('account_id')

            result[f'txn_count_{window}d'] = grp.size().reindex(all_accounts, fill_value=0)

            if window == 30:
                result['txn_amount_mean_30d'] = grp['transaction_amount'].mean().reindex(all_accounts, fill_value=0)
                result['txn_amount_max_30d'] = grp['transaction_amount'].max().reindex(all_accounts, fill_value=0)
                result['txn_amount_std_30d'] = grp['transaction_amount'].std().reindex(all_accounts, fill_value=0)
                result['txn_amount_sum_30d'] = grp['transaction_amount'].sum().reindex(all_accounts, fill_value=0)

        result['velocity_acceleration'] = (
            result['txn_count_7d'] / (result['txn_count_30d'] / 4).clip(lower=1)
        )
        result['frequency_change_ratio'] = (
            result['txn_count_30d'] / (result['txn_count_90d'] / 3).clip(lower=1)
        )

        result = result.fillna(0)
        self.validate_output(result)
        return result
