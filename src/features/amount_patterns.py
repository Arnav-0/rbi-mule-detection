import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.features.base import BaseFeatureGenerator


class AmountPatternFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('amount_patterns', 'amount_patterns')

    def get_feature_names(self) -> list[str]:
        return [
            'round_amount_ratio', 'structuring_score', 'structuring_score_broad',
            'amount_entropy', 'amount_skewness', 'amount_kurtosis',
            'pct_above_10k', 'amount_concentration',
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
        result = pd.DataFrame(0.0, index=pd.Index(all_accounts, name='account_id'),
                              columns=self.get_feature_names())

        amt = txn_valid[['account_id', 'transaction_amount']].copy()
        amt['is_round'] = ((amt['transaction_amount'] % 1000 == 0) |
                           (amt['transaction_amount'] % 5000 == 0) |
                           (amt['transaction_amount'] % 10000 == 0))
        amt['in_struct'] = (amt['transaction_amount'] >= 45000) & (amt['transaction_amount'] <= 49999)
        amt['in_struct_broad'] = (amt['transaction_amount'] >= 40000) & (amt['transaction_amount'] <= 49999)
        amt['above_10k'] = amt['transaction_amount'] > 10000

        grp = amt.groupby('account_id')
        result['round_amount_ratio'] = grp['is_round'].mean().reindex(all_accounts, fill_value=0)
        result['structuring_score'] = grp['in_struct'].mean().reindex(all_accounts, fill_value=0)
        result['structuring_score_broad'] = grp['in_struct_broad'].mean().reindex(all_accounts, fill_value=0)
        result['pct_above_10k'] = grp['above_10k'].mean().reindex(all_accounts, fill_value=0)

        # Vectorized skewness and kurtosis
        skew_vals = grp['transaction_amount'].skew().reindex(all_accounts, fill_value=0)
        kurt_vals = grp['transaction_amount'].apply(lambda x: x.kurtosis() if len(x) >= 3 else 0).reindex(all_accounts, fill_value=0)
        result['amount_skewness'] = skew_vals
        result['amount_kurtosis'] = kurt_vals

        # Entropy and concentration (Gini) need per-group compute but use apply
        def _entropy(x):
            if len(x) < 2:
                return 0.0
            hist, _ = np.histogram(x, bins=20)
            return scipy_stats.entropy(hist + 1e-10)

        def _gini(x):
            arr = np.sort(np.asarray(x, dtype=float))
            n = len(arr)
            if n < 2 or arr.sum() == 0:
                return 0.0
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n

        result['amount_entropy'] = grp['transaction_amount'].apply(_entropy).reindex(all_accounts, fill_value=0)
        result['amount_concentration'] = grp['transaction_amount'].apply(_gini).reindex(all_accounts, fill_value=0)

        result = result.fillna(0)
        self.validate_output(result)
        return result
