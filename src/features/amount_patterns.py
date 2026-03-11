"""Group 2: Amount pattern features (8 features) — structuring and round-amount detection."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from src.features.base import BaseFeatureGenerator


def _gini(arr: np.ndarray) -> float:
    if arr.sum() == 0 or len(arr) == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_arr)) / (n * np.sum(sorted_arr)) - (n + 1) / n


def _entropy(amounts: np.ndarray) -> float:
    counts, _ = np.histogram(amounts, bins=20)
    return float(stats.entropy(counts + 1e-10))


class AmountPatternFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('amount_patterns', 'amount_patterns')

    def get_feature_names(self) -> list[str]:
        return [
            'round_amount_ratio', 'structuring_score', 'structuring_score_broad',
            'amount_entropy', 'amount_skewness', 'amount_kurtosis',
            'pct_above_10k', 'amount_concentration',
        ]

    def compute(self, txn: pd.DataFrame, profile: pd.DataFrame, cutoff_date=None, **kwargs) -> pd.DataFrame:
        if cutoff_date is None:
            cutoff_date = pd.Timestamp('2025-06-30')

        txn = txn.copy()
        txn['transaction_date'] = pd.to_datetime(txn['transaction_date'])
        txn_valid = txn[txn['transaction_date'] <= cutoff_date].copy()

        if profile is not None and len(profile) > 0:
            all_accounts = profile['account_id'].unique()
        else:
            all_accounts = txn_valid['account_id'].unique()

        grouped = txn_valid.groupby('account_id')

        def _compute_row(acc_id, amounts):
            n = len(amounts)
            if n == 0:
                return {feat: 0.0 for feat in self.get_feature_names()}

            amt = amounts.values.astype(float)

            round_mask = (amt % 1000 == 0) | (amt % 5000 == 0) | (amt % 10000 == 0)
            round_ratio = round_mask.mean()

            struct_mask = (amt >= 45000) & (amt <= 49999)
            struct_score = struct_mask.mean()

            struct_broad_mask = (amt >= 40000) & (amt <= 49999)
            struct_broad = struct_broad_mask.mean()

            entropy = _entropy(amt)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                skewness = float(stats.skew(amt)) if n >= 3 else 0.0
                kurtosis = float(stats.kurtosis(amt)) if n >= 3 else 0.0
            skewness = 0.0 if np.isnan(skewness) else skewness
            kurtosis = 0.0 if np.isnan(kurtosis) else kurtosis
            pct_above = (amt > 10000).mean()
            concentration = _gini(amt)

            return {
                'round_amount_ratio': round_ratio,
                'structuring_score': struct_score,
                'structuring_score_broad': struct_broad,
                'amount_entropy': entropy,
                'amount_skewness': skewness,
                'amount_kurtosis': kurtosis,
                'pct_above_10k': pct_above,
                'amount_concentration': concentration,
            }

        account_data = {}
        for acc_id, grp in grouped:
            account_data[acc_id] = _compute_row(acc_id, grp['transaction_amount'])

        zero_row = {feat: 0.0 for feat in self.get_feature_names()}
        records = [account_data.get(acc, zero_row) for acc in all_accounts]
        result = pd.DataFrame(records, index=all_accounts)
        result.index.name = 'account_id'

        self.validate_output(result)
        return result
