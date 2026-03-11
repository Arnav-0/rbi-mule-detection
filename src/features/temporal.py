"""Group 3: Temporal features (8 features) — dormancy, bursts, unusual timing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.features.base import BaseFeatureGenerator

NIGHT_HOURS = {23, 0, 1, 2, 3, 4, 5}


class TemporalFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('temporal', 'temporal')

    def get_feature_names(self) -> list[str]:
        return [
            'dormancy_days', 'max_gap_days', 'burst_after_dormancy',
            'unusual_hour_ratio', 'weekend_ratio', 'night_weekend_combo',
            'monthly_txn_cv', 'days_to_first_txn',
        ]

    def compute(self, txn: pd.DataFrame, profile: pd.DataFrame, cutoff_date=None, **kwargs) -> pd.DataFrame:
        if cutoff_date is None:
            cutoff_date = pd.Timestamp('2025-06-30')

        txn = txn.copy()
        txn['transaction_date'] = pd.to_datetime(txn['transaction_date'])
        txn_valid = txn[txn['transaction_date'] <= cutoff_date].copy()

        # Derive hour/weekend if not present
        if 'transaction_hour' not in txn_valid.columns:
            txn_valid['transaction_hour'] = txn_valid['transaction_date'].dt.hour
        if 'is_weekend' not in txn_valid.columns:
            txn_valid['is_weekend'] = txn_valid['transaction_date'].dt.dayofweek >= 5

        if profile is not None and len(profile) > 0:
            all_accounts = profile['account_id'].unique()
        else:
            all_accounts = txn_valid['account_id'].unique()

        cutoff_30d = cutoff_date - pd.Timedelta(days=30)

        # Pre-build open_date lookup to avoid O(n*m) profile scans in the loop
        open_date_map: dict = {}
        if profile is not None and 'account_open_date' in profile.columns:
            for _, prow in profile.iterrows():
                if pd.notna(prow['account_open_date']):
                    open_date_map[prow['account_id']] = pd.to_datetime(prow['account_open_date'])

        rows = {}
        for acc_id, grp in txn_valid.groupby('account_id'):
            grp = grp.sort_values('transaction_date')
            dates = grp['transaction_date']

            gaps = dates.diff().dt.days.dropna()
            max_gap = float(gaps.max()) if len(gaps) > 0 else 0.0
            dormancy = max_gap if max_gap > 90 else 0.0

            recent_count = int((dates > cutoff_30d).sum())
            burst = 1.0 if (dormancy > 0 and recent_count > 10) else 0.0

            hours = grp['transaction_hour'].values
            unusual_ratio = float(np.isin(hours, list(NIGHT_HOURS)).mean()) if len(hours) > 0 else 0.0

            weekend_ratio = float(grp['is_weekend'].mean()) if len(grp) > 0 else 0.0
            night_weekend = unusual_ratio * weekend_ratio

            # Monthly CV
            monthly_counts = grp.groupby(grp['transaction_date'].dt.to_period('M')).size()
            if len(monthly_counts) > 1 and monthly_counts.mean() > 0:
                cv = float(monthly_counts.std() / monthly_counts.mean())
            else:
                cv = 0.0

            # Days to first txn from account open date (O(1) lookup)
            days_to_first = 0.0
            if acc_id in open_date_map:
                first_txn = dates.min()
                days_to_first = float((first_txn - open_date_map[acc_id]).days)

            rows[acc_id] = {
                'dormancy_days': dormancy,
                'max_gap_days': max_gap,
                'burst_after_dormancy': burst,
                'unusual_hour_ratio': unusual_ratio,
                'weekend_ratio': weekend_ratio,
                'night_weekend_combo': night_weekend,
                'monthly_txn_cv': cv,
                'days_to_first_txn': days_to_first,
            }

        zero_row = {feat: 0.0 for feat in self.get_feature_names()}
        records = [rows.get(acc, zero_row) for acc in all_accounts]
        result = pd.DataFrame(records, index=all_accounts)
        result.index.name = 'account_id'

        self.validate_output(result)
        return result
