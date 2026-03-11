"""Group 6: Profile mismatch features (5 features) — transaction vs declared profile."""
from __future__ import annotations

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

    def compute(self, txn: pd.DataFrame, profile: pd.DataFrame, cutoff_date=None,
                velocity_features: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        if cutoff_date is None:
            cutoff_date = pd.Timestamp('2025-06-30')

        txn = txn.copy()
        txn['transaction_date'] = pd.to_datetime(txn['transaction_date'])
        txn_valid = txn[txn['transaction_date'] <= cutoff_date].copy()

        if profile is not None and len(profile) > 0:
            all_accounts = profile['account_id'].unique()
        else:
            all_accounts = txn_valid['account_id'].unique()

        # Build profile lookup indexed by account_id
        if profile is not None and len(profile) > 0:
            prof = profile.set_index('account_id')
        else:
            prof = pd.DataFrame(index=all_accounts)

        # Use velocity features if provided
        vel = velocity_features if velocity_features is not None else pd.DataFrame(index=all_accounts)

        rows = {}
        # Balance volatility from txn balance_after column
        if 'balance_after' in txn_valid.columns:
            txn_valid['_date'] = pd.to_datetime(txn_valid['transaction_date']).dt.date
            daily_bal = (
                txn_valid.groupby(['account_id', '_date'])['balance_after']
                .last()
            )
            bal_std = daily_bal.groupby('account_id').std().fillna(0)
            bal_mean = daily_bal.groupby('account_id').mean().fillna(0)
            bal_vol = (bal_std / bal_mean.clip(lower=1)).fillna(0)
        else:
            bal_vol = pd.Series(0.0, index=all_accounts)

        for acc_id in all_accounts:
            p = prof.loc[acc_id] if acc_id in prof.index else {}

            income = float(p.get('declared_income', 0) or 0)
            age_days = float(p.get('account_age_days', 0) or 0)
            balance = float(p.get('current_balance', 0) or 0)
            acc_type = str(p.get('account_type', '')) if p is not None else ''

            txn_sum_30 = float(vel['txn_amount_sum_30d'].get(acc_id, 0) if 'txn_amount_sum_30d' in vel.columns else 0)
            txn_cnt_30 = float(vel['txn_count_30d'].get(acc_id, 0) if 'txn_count_30d' in vel.columns else 0)
            txn_mean_30 = float(vel['txn_amount_mean_30d'].get(acc_id, 0) if 'txn_amount_mean_30d' in vel.columns else 0)

            vol_vs_income = txn_sum_30 / max(income, 1.0)
            age_vs_activity = txn_cnt_30 / max(age_days, 1.0)
            avg_vs_balance = txn_mean_30 / max(balance, 1.0)
            mismatch = 1.0 if (acc_type.lower() == 'savings' and txn_mean_30 > 50000) else 0.0
            bvol = float(bal_vol.get(acc_id, 0.0))

            rows[acc_id] = {
                'txn_volume_vs_income': vol_vs_income,
                'account_age_vs_activity': age_vs_activity,
                'avg_txn_vs_balance': avg_vs_balance,
                'product_txn_mismatch': mismatch,
                'balance_volatility': bvol,
            }

        zero_row = {feat: 0.0 for feat in self.get_feature_names()}
        records = [rows.get(acc, zero_row) for acc in all_accounts]
        result = pd.DataFrame(records, index=all_accounts)
        result.index.name = 'account_id'

        self.validate_output(result)
        return result
