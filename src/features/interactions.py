import pandas as pd

from src.features.base import BaseFeatureGenerator


class InteractionFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('interactions', 'interactions')

    def get_feature_names(self) -> list[str]:
        return [
            'dormancy_x_burst', 'round_x_structuring',
            'fanin_x_passthrough_speed', 'new_account_x_high_value',
            'velocity_x_centrality',
        ]

    def _safe_col(self, df, col):
        return df[col] if col in df.columns else 0

    def compute_from_features(self, all_features, profile=None):
        result = pd.DataFrame(index=all_features.index)
        result.index.name = 'account_id'

        result['dormancy_x_burst'] = (
            self._safe_col(all_features, 'dormancy_days') *
            self._safe_col(all_features, 'txn_count_7d')
        )
        result['round_x_structuring'] = (
            self._safe_col(all_features, 'round_amount_ratio') *
            self._safe_col(all_features, 'structuring_score')
        )

        median_delta = self._safe_col(all_features, 'credit_debit_time_delta_median')
        if isinstance(median_delta, (int, float)):
            inv_speed = 1 / max(median_delta, 0.1)
        else:
            inv_speed = 1 / median_delta.clip(lower=0.1)
        result['fanin_x_passthrough_speed'] = (
            self._safe_col(all_features, 'fan_in_ratio') * inv_speed
        )

        # is_new_account: account opened within 90 days of cutoff
        is_new = 0
        if profile is not None and 'account_id' in profile.columns:
            if 'account_opening_date' in profile.columns:
                prof_temp = profile.set_index('account_id')
                open_dates = pd.to_datetime(prof_temp['account_opening_date'], errors='coerce')
                age_days = (pd.Timestamp('2025-06-30') - open_dates).dt.days.fillna(365)
                is_new = (age_days.reindex(all_features.index, fill_value=365) < 90).astype(float)
            elif 'account_age_days' in profile.columns:
                age_map = profile.set_index('account_id')['account_age_days']
                is_new = (age_map.reindex(all_features.index, fill_value=365) < 90).astype(float)
        result['new_account_x_high_value'] = (
            is_new * self._safe_col(all_features, 'pct_above_10k')
        )

        result['velocity_x_centrality'] = (
            self._safe_col(all_features, 'velocity_acceleration') *
            self._safe_col(all_features, 'betweenness_centrality')
        )

        result = result.fillna(0)
        self.validate_output(result)
        return result

    def compute(self, txn=None, profile=None, cutoff_date=None, **kwargs):
        raise NotImplementedError("Use compute_from_features() instead")
