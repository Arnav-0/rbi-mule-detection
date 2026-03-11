import pandas as pd
import numpy as np

from src.features.base import BaseFeatureGenerator


class KYCBehavioralFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('kyc_behavioral', 'kyc_behavioral')

    def get_feature_names(self) -> list[str]:
        return [
            'mobile_change_flag', 'activity_change_post_mobile',
            'kyc_completeness', 'linked_account_count',
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

        if profile is None:
            self.validate_output(result)
            return result

        prof = profile.set_index('account_id') if 'account_id' in profile.columns else profile

        # Mobile change flag
        mobile_col = 'last_mobile_update_date' if 'last_mobile_update_date' in prof.columns else 'mobile_change_date'
        if mobile_col in prof.columns:
            mobile_dates = pd.to_datetime(prof[mobile_col], format='mixed', errors='coerce')
            has_mobile = mobile_dates.notna()
            result['mobile_change_flag'] = has_mobile.reindex(all_accounts, fill_value=False).astype(float)

            # Activity change post mobile — vectorized approach
            # For accounts with mobile change, compute txn count 30d before vs 30d after
            accounts_with_mobile = has_mobile[has_mobile].index.intersection(all_accounts)
            if len(accounts_with_mobile) > 0:
                # Build a lookup of mobile dates for relevant accounts
                mc_dates = mobile_dates.reindex(accounts_with_mobile).dropna()
                # Join mobile date to transactions
                txn_mc = txn_valid[txn_valid['account_id'].isin(mc_dates.index)].copy()
                txn_mc = txn_mc.merge(
                    mc_dates.rename('mc_date').reset_index(),
                    on='account_id', how='inner'
                )
                txn_mc['days_from_mc'] = (txn_mc['transaction_date'] - txn_mc['mc_date']).dt.days

                before = txn_mc[(txn_mc['days_from_mc'] >= -30) & (txn_mc['days_from_mc'] < 0)].groupby('account_id').size()
                after = txn_mc[(txn_mc['days_from_mc'] >= 0) & (txn_mc['days_from_mc'] <= 30)].groupby('account_id').size()

                ratio = after.reindex(accounts_with_mobile, fill_value=0) / before.reindex(accounts_with_mobile, fill_value=0).clip(lower=1)
                result.loc[ratio.index.intersection(result.index), 'activity_change_post_mobile'] = ratio

        # KYC completeness
        kyc_fields = [c for c in prof.columns if 'kyc' in c.lower() or c in [
            'pan_available', 'aadhaar_available', 'passport_available',
            'nomination_flag',
        ]]
        if kyc_fields:
            # Convert Y/N to boolean for completeness
            kyc_data = prof[kyc_fields].copy()
            for col in kyc_fields:
                if kyc_data[col].dtype == object:
                    kyc_data[col] = (kyc_data[col].str.upper() == 'Y').astype(float)
            completeness = kyc_data.mean(axis=1)
            result['kyc_completeness'] = completeness.reindex(all_accounts, fill_value=0)

        # Linked account count
        if 'customer_id' in prof.columns:
            cust_counts = prof.groupby('customer_id').size()
            cust_map = prof['customer_id']
            linked = cust_map.map(cust_counts).reindex(all_accounts, fill_value=1)
            result['linked_account_count'] = linked

        result = result.fillna(0)
        self.validate_output(result)
        return result
