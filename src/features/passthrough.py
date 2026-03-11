"""Group 4: Pass-through features (7 features) — rapid money-in/money-out detection."""
from __future__ import annotations

import pandas as pd
from src.features.base import BaseFeatureGenerator


class PassThroughFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('passthrough', 'passthrough')

    def get_feature_names(self) -> list[str]:
        return [
            'credit_debit_time_delta_median', 'credit_debit_time_delta_min',
            'matched_amount_ratio', 'net_flow_ratio', 'rapid_turnover_score',
            'credit_debit_symmetry', 'max_single_day_volume',
        ]

    def compute(self, txn: pd.DataFrame, profile: pd.DataFrame, cutoff_date=None, **kwargs) -> pd.DataFrame:
        if cutoff_date is None:
            cutoff_date = pd.Timestamp('2025-06-30')

        txn_valid = txn.copy()
        txn_valid['transaction_date'] = pd.to_datetime(txn_valid['transaction_date'])
        txn_valid = txn_valid[txn_valid['transaction_date'] <= cutoff_date].copy()

        if profile is not None and len(profile) > 0:
            all_accounts = profile['account_id'].unique()
        else:
            all_accounts = txn_valid['account_id'].unique()

        # ------------------------------------------------------------------
        # Vectorized aggregates (no per-account loop needed)
        # ------------------------------------------------------------------

        # Max single-day volume
        txn_valid['_date'] = txn_valid['transaction_date'].dt.date
        daily_vol = txn_valid.groupby(['account_id', '_date'])['transaction_amount'].sum()
        max_daily = daily_vol.groupby('account_id').max()

        credit_mask = txn_valid['is_credit'] == 1
        debit_mask = txn_valid['is_credit'] == 0

        credit_sums = txn_valid.loc[credit_mask].groupby('account_id')['transaction_amount'].sum()
        debit_sums = txn_valid.loc[debit_mask].groupby('account_id')['transaction_amount'].sum()
        n_credits = txn_valid.loc[credit_mask].groupby('account_id').size()
        n_debits = txn_valid.loc[debit_mask].groupby('account_id').size()

        # ------------------------------------------------------------------
        # Global merge_asof: for each credit find the NEXT debit (same account)
        # ------------------------------------------------------------------
        # merge_asof requires global sort on the ON key (credit_date / debit_date).
        # The `by='account_id'` parameter restricts matching to the same account.
        credits = (
            txn_valid.loc[credit_mask, ['account_id', 'transaction_date', 'transaction_amount']]
            .rename(columns={'transaction_date': 'credit_date', 'transaction_amount': 'credit_amt'})
            .sort_values('credit_date')
            .reset_index(drop=True)
        )
        debits = (
            txn_valid.loc[debit_mask, ['account_id', 'transaction_date', 'transaction_amount']]
            .rename(columns={'transaction_date': 'debit_date', 'transaction_amount': 'debit_amt'})
            .sort_values('debit_date')
            .reset_index(drop=True)
        )

        if len(credits) > 0 and len(debits) > 0:
            merged = pd.merge_asof(
                credits,
                debits,
                left_on='credit_date',
                right_on='debit_date',
                by='account_id',
                direction='forward',
            )
            merged = merged.dropna(subset=['debit_date'])
            merged['delta_h'] = (
                (merged['debit_date'] - merged['credit_date']).dt.total_seconds() / 3600.0
            )
            # Matched: within 5% amount AND < 24h
            merged['_amount_close'] = (
                (merged['credit_amt'] - merged['debit_amt']).abs()
                / merged['credit_amt'].clip(lower=1)
            ) < 0.05
            merged['_matched'] = merged['_amount_close'] & (merged['delta_h'] < 24.0)
            merged['_rapid'] = merged['delta_h'] < 48.0

            # Per-account stats from merged
            grp = merged.groupby('account_id')
            delta_median = grp['delta_h'].median()
            delta_min = grp['delta_h'].min()
            matched_count = grp['_matched'].sum()
            rapid_credit_sum = merged.loc[merged['_rapid']].groupby('account_id')['credit_amt'].sum()
        else:
            delta_median = pd.Series(dtype=float)
            delta_min = pd.Series(dtype=float)
            matched_count = pd.Series(dtype=float)
            rapid_credit_sum = pd.Series(dtype=float)

        # ------------------------------------------------------------------
        # Assemble result for all accounts
        # ------------------------------------------------------------------
        rows = {}
        for acc_id in all_accounts:
            nc = int(n_credits.get(acc_id, 0))
            nd = int(n_debits.get(acc_id, 0))
            total_credits = float(credit_sums.get(acc_id, 0.0))
            total_debits = float(debit_sums.get(acc_id, 0.0))

            net_flow = total_credits / max(total_debits, 1.0)
            symmetry = 1.0 - abs(nc - nd) / max(nc + nd, 1)
            max_day_vol = float(max_daily.get(acc_id, 0.0))

            has_pairs = acc_id in delta_median.index
            if not has_pairs:
                rows[acc_id] = {
                    'credit_debit_time_delta_median': 999.0,
                    'credit_debit_time_delta_min': 999.0,
                    'matched_amount_ratio': 0.0,
                    'net_flow_ratio': net_flow,
                    'rapid_turnover_score': 0.0,
                    'credit_debit_symmetry': symmetry,
                    'max_single_day_volume': max_day_vol,
                }
            else:
                mc = float(matched_count.get(acc_id, 0.0))
                rcs = float(rapid_credit_sum.get(acc_id, 0.0))
                rows[acc_id] = {
                    'credit_debit_time_delta_median': float(delta_median[acc_id]),
                    'credit_debit_time_delta_min': float(delta_min[acc_id]),
                    'matched_amount_ratio': mc / max(nc, 1),
                    'net_flow_ratio': net_flow,
                    'rapid_turnover_score': rcs / max(total_credits, 1.0),
                    'credit_debit_symmetry': symmetry,
                    'max_single_day_volume': max_day_vol,
                }

        zero_row = {
            'credit_debit_time_delta_median': 999.0,
            'credit_debit_time_delta_min': 999.0,
            'matched_amount_ratio': 0.0,
            'net_flow_ratio': 0.0,
            'rapid_turnover_score': 0.0,
            'credit_debit_symmetry': 0.0,
            'max_single_day_volume': 0.0,
        }
        records = [rows.get(acc, zero_row) for acc in all_accounts]
        result = pd.DataFrame(records, index=all_accounts)
        result.index.name = 'account_id'

        self.validate_output(result)
        return result
