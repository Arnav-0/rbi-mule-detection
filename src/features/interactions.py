"""Group 8: Interaction features (5 features) — compound cross-group signals."""
from __future__ import annotations

import pandas as pd
from src.features.base import BaseFeatureGenerator


class InteractionFeatureGenerator(BaseFeatureGenerator):
    def __init__(self):
        super().__init__('interactions', 'interactions')

    def get_feature_names(self) -> list[str]:
        return [
            'dormancy_x_burst',
            'round_x_structuring',
            'fanin_x_passthrough_speed',
            'new_account_x_high_value',
            'velocity_x_centrality',
        ]

    def compute(self, txn: pd.DataFrame, profile: pd.DataFrame, cutoff_date=None, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Use compute_from_features() for interaction features.")

    def compute_from_features(self, all_features: pd.DataFrame, profile: pd.DataFrame = None) -> pd.DataFrame:
        """Compute interaction features from the complete feature matrix."""
        def _col(name: str) -> pd.Series:
            if name in all_features.columns:
                return all_features[name].fillna(0)
            return pd.Series(0.0, index=all_features.index)

        result = pd.DataFrame(index=all_features.index)
        result.index.name = 'account_id'

        result['dormancy_x_burst'] = _col('dormancy_days') * _col('txn_count_7d')

        result['round_x_structuring'] = _col('round_amount_ratio') * _col('structuring_score')

        result['fanin_x_passthrough_speed'] = (
            _col('fan_in_ratio')
            * (1.0 / _col('credit_debit_time_delta_median').clip(lower=0.1))
        )

        # is_new_account: derive from account_age_days if available, else 0
        if 'account_age_days' in (profile.columns if profile is not None else []):
            prof_indexed = profile.set_index('account_id') if 'account_id' in profile.columns else profile
            age = prof_indexed['account_age_days'].reindex(all_features.index, fill_value=365)
            is_new = (age <= 180).astype(float)
        else:
            is_new = pd.Series(0.0, index=all_features.index)

        result['new_account_x_high_value'] = is_new * _col('pct_above_10k')

        result['velocity_x_centrality'] = _col('velocity_acceleration') * _col('betweenness_centrality')

        return result
