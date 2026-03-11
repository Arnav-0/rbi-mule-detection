import numpy as np
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
        result = pd.DataFrame(index=pd.Index(all_accounts, name='account_id'),
                              columns=self.get_feature_names(), dtype=float)
        result[:] = 0.0
        result['credit_debit_time_delta_median'] = 999.0
        result['credit_debit_time_delta_min'] = 999.0

        grp = txn_valid.groupby('account_id')

        # Max single day volume (vectorized)
        daily_vol = txn_valid.groupby(
            ['account_id', txn_valid['transaction_date'].dt.date]
        )['transaction_amount'].sum()
        result['max_single_day_volume'] = daily_vol.groupby(level=0).max().reindex(all_accounts, fill_value=0)

        # Credit/debit counts and sums
        credit_mask = txn_valid['is_credit'] == 1
        credits_df = txn_valid[credit_mask]
        debits_df = txn_valid[~credit_mask]

        n_credits = credits_df.groupby('account_id').size().reindex(all_accounts, fill_value=0)
        n_debits = debits_df.groupby('account_id').size().reindex(all_accounts, fill_value=0)
        sum_credits = credits_df.groupby('account_id')['transaction_amount'].sum().reindex(all_accounts, fill_value=0)
        sum_debits = debits_df.groupby('account_id')['transaction_amount'].sum().reindex(all_accounts, fill_value=0)

        result['net_flow_ratio'] = (sum_credits / sum_debits.clip(lower=1)).clip(upper=100)
        total = n_credits + n_debits
        result['credit_debit_symmetry'] = 1 - (n_credits - n_debits).abs() / total.clip(lower=1)

        # Pass-through matching: use merge_asof per account in batches
        # Only process accounts that have both credits and debits
        has_both = all_accounts[(n_credits > 0) & (n_debits > 0)]
        print(f"  Passthrough: matching {len(has_both)} accounts with credits+debits...")

        credits_sorted = credits_df[['account_id', 'transaction_date', 'transaction_amount']].sort_values(
            ['account_id', 'transaction_date'])
        debits_sorted = debits_df[['account_id', 'transaction_date', 'transaction_amount']].sort_values(
            ['account_id', 'transaction_date'])

        # Process in chunks to avoid memory issues
        chunk_size = 5000
        for i in range(0, len(has_both), chunk_size):
            chunk_accounts = has_both[i:i + chunk_size]
            c_chunk = credits_sorted[credits_sorted['account_id'].isin(chunk_accounts)]
            d_chunk = debits_sorted[debits_sorted['account_id'].isin(chunk_accounts)]

            if len(c_chunk) == 0 or len(d_chunk) == 0:
                continue

            # merge_asof by account
            c_prep = c_chunk.rename(columns={'transaction_amount': 'credit_amount'})
            d_prep = d_chunk.rename(columns={
                'transaction_date': 'debit_date',
                'transaction_amount': 'debit_amount'
            })

            merged = pd.merge_asof(
                c_prep.sort_values('transaction_date'),
                d_prep.sort_values('debit_date'),
                by='account_id',
                left_on='transaction_date',
                right_on='debit_date',
                direction='forward',
            ).dropna(subset=['debit_date'])

            if len(merged) == 0:
                continue

            merged['delta_hours'] = (merged['debit_date'] - merged['transaction_date']).dt.total_seconds() / 3600
            merged['amt_diff_ratio'] = (merged['credit_amount'] - merged['debit_amount']).abs() / merged['credit_amount'].clip(lower=1)
            merged['matched_24h'] = (merged['amt_diff_ratio'] < 0.05) & (merged['delta_hours'] < 24)
            merged['rapid_48h'] = (merged['amt_diff_ratio'] < 0.05) & (merged['delta_hours'] < 48)

            for acct, mg in merged.groupby('account_id'):
                if acct not in result.index:
                    continue
                result.at[acct, 'credit_debit_time_delta_median'] = mg['delta_hours'].median()
                result.at[acct, 'credit_debit_time_delta_min'] = mg['delta_hours'].min()
                nc = n_credits[acct]
                result.at[acct, 'matched_amount_ratio'] = mg['matched_24h'].sum() / max(nc, 1)
                rapid_amount = mg.loc[mg['rapid_48h'], 'credit_amount'].sum()
                result.at[acct, 'rapid_turnover_score'] = rapid_amount / max(sum_credits[acct], 1)

        result = result.fillna(0)
        self.validate_output(result)
        return result
