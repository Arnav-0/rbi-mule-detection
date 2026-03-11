"""Group 7: KYC behavioral features (4 features)."""
from __future__ import annotations

import pandas as pd
from src.features.base import BaseFeatureGenerator

KYC_FIELDS = ['pan_number', 'aadhaar_number', 'address', 'email', 'mobile_number', 'dob', 'photo_id']


class KYCBehavioralFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('kyc_behavioral', 'kyc_behavioral')

    def get_feature_names(self) -> list[str]:
        return [
            'mobile_change_flag', 'activity_change_post_mobile',
            'kyc_completeness', 'linked_account_count',
        ]

    def compute(self, txn: pd.DataFrame, profile: pd.DataFrame, cutoff_date=None, **kwargs) -> pd.DataFrame:
        if cutoff_date is None:
            cutoff_date = pd.Timestamp('2025-06-30')

        txn = txn.copy()
        txn['transaction_date'] = pd.to_datetime(txn['transaction_date'])
        txn_valid = txn[txn['transaction_date'] <= cutoff_date].copy()

        if profile is not None and len(profile) > 0:
            all_accounts = profile['account_id'].unique()
            prof = profile.set_index('account_id')
        else:
            all_accounts = txn_valid['account_id'].unique()
            prof = pd.DataFrame(index=all_accounts)

        # Linked account count: accounts per customer_id
        linked_counts: dict = {}
        if 'customer_id' in prof.columns:
            cust_to_accounts = prof.groupby('customer_id').size()
            for acc_id in all_accounts:
                if acc_id in prof.index:
                    cust_id = prof.loc[acc_id, 'customer_id']
                    linked_counts[acc_id] = float(cust_to_accounts.get(cust_id, 1))
                else:
                    linked_counts[acc_id] = 1.0

        rows = {}
        for acc_id in all_accounts:
            p = prof.loc[acc_id] if acc_id in prof.index else {}

            # Mobile change flag
            mobile_flag = 0.0
            mobile_change_date = None
            for col in ['mobile_change_date', 'phone_change_date', 'contact_change_date']:
                if col in (p.keys() if hasattr(p, 'keys') else []):
                    val = p.get(col)
                    if val is not None and str(val).strip() not in ('', 'nan', 'None', 'NaT'):
                        mobile_flag = 1.0
                        try:
                            mobile_change_date = pd.to_datetime(val)
                        except Exception:
                            pass
                        break

            # Activity change post mobile change
            activity_change = 0.0
            if mobile_flag == 1.0 and mobile_change_date is not None:
                acc_txn = txn_valid[txn_valid['account_id'] == acc_id]
                before_cnt = int((acc_txn['transaction_date'] >= (mobile_change_date - pd.Timedelta(days=30)))
                                 & (acc_txn['transaction_date'] < mobile_change_date)).sum()
                after_cnt = int((acc_txn['transaction_date'] >= mobile_change_date)
                                & (acc_txn['transaction_date'] < mobile_change_date + pd.Timedelta(days=30))).sum()
                activity_change = float(after_cnt - before_cnt)

            # KYC completeness
            present_kyc = [f for f in KYC_FIELDS if f in (p.keys() if hasattr(p, 'keys') else [])
                           and p.get(f) is not None
                           and str(p.get(f)).strip() not in ('', 'nan', 'None')]
            completeness = len(present_kyc) / len(KYC_FIELDS)

            linked = linked_counts.get(acc_id, 1.0)

            rows[acc_id] = {
                'mobile_change_flag': mobile_flag,
                'activity_change_post_mobile': activity_change,
                'kyc_completeness': completeness,
                'linked_account_count': linked,
            }

        zero_row = {feat: 0.0 for feat in self.get_feature_names()}
        records = [rows.get(acc, zero_row) for acc in all_accounts]
        result = pd.DataFrame(records, index=all_accounts)
        result.index.name = 'account_id'

        self.validate_output(result)
        return result
