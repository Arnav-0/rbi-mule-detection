import numpy as np


class NaturalLanguageExplainer:
    FEATURE_DESCRIPTIONS = {
        'txn_count': 'total number of transactions',
        'credit_count': 'number of incoming credits',
        'debit_count': 'number of outgoing debits',
        'credit_amount_sum': 'total amount of credits received',
        'debit_amount_sum': 'total amount of debits sent',
        'credit_amount_mean': 'average credit amount',
        'debit_amount_mean': 'average debit amount',
        'credit_amount_std': 'variability in credit amounts',
        'debit_amount_std': 'variability in debit amounts',
        'credit_amount_max': 'largest single credit received',
        'debit_amount_max': 'largest single debit sent',
        'net_flow': 'net money flow (credits minus debits)',
        'credit_debit_ratio': 'ratio of credits to debits',
        'credit_debit_amount_ratio': 'ratio of credit amounts to debit amounts',
        'avg_txn_amount': 'average transaction amount',
        'txn_amount_cv': 'coefficient of variation in transaction amounts',
        'rapid_movement_count': 'number of rapid fund movements (credit followed by quick debit)',
        'rapid_movement_ratio': 'proportion of transactions involving rapid fund movement',
        'matched_amount_ratio': 'credits that were followed by matching debits within 24 hours',
        'avg_time_between_credit_debit': 'average time gap between receiving and sending funds',
        'min_time_between_credit_debit': 'shortest time between receiving and sending funds',
        'round_amount_ratio': 'proportion of transactions with round amounts',
        'just_below_threshold_count': 'transactions just below reporting thresholds',
        'just_below_threshold_ratio': 'proportion of transactions just below reporting thresholds',
        'unique_counterparties': 'number of unique counterparties',
        'counterparty_concentration': 'concentration of transactions among few counterparties',
        'unique_credit_sources': 'number of unique funding sources',
        'unique_debit_destinations': 'number of unique destinations for funds',
        'source_dest_overlap': 'overlap between funding sources and destinations',
        'fan_in_count': 'number of accounts sending money in (fan-in pattern)',
        'fan_out_count': 'number of accounts receiving money out (fan-out pattern)',
        'fan_in_out_ratio': 'ratio of fan-in to fan-out activity',
        'night_txn_ratio': 'proportion of transactions at night (11PM-5AM)',
        'weekend_txn_ratio': 'proportion of transactions on weekends',
        'peak_hour_concentration': 'concentration of transactions in peak hours',
        'txn_hour_entropy': 'randomness in transaction timing across hours',
        'daily_txn_velocity': 'average daily transaction rate',
        'weekly_txn_velocity': 'average weekly transaction rate',
        'max_daily_txn_count': 'maximum transactions in a single day',
        'max_daily_txn_amount': 'maximum total amount transacted in a single day',
        'burst_score': 'burstiness of transaction activity',
        'dormancy_ratio': 'proportion of days with no activity',
        'active_days': 'number of days with at least one transaction',
        'account_age_days': 'age of the account in days',
        'days_since_first_txn': 'days since first transaction',
        'days_since_last_txn': 'days since most recent transaction',
        'early_activity_ratio': 'proportion of activity in first 30 days',
        'structuring_score': 'likelihood of deliberate transaction structuring',
        'layering_depth': 'depth of layering pattern detected',
        'velocity_change_ratio': 'change in transaction velocity over time',
        'amount_trend_slope': 'trend in transaction amounts over time',
        'pattern_regularity_score': 'regularity/repetitiveness of transaction patterns',
        'counterparty_risk_score': 'aggregate risk from counterparty relationships',
        'geographic_dispersion': 'geographic spread of transaction counterparties',
        'channel_diversity': 'diversity of transaction channels used',
        'balance_volatility': 'volatility of account balance',
        'avg_balance': 'average account balance',
        'min_balance': 'minimum account balance observed',
        'balance_utilization': 'how much of available balance is typically used',
    }

    def explain(self, shap_values, feature_values, feature_names, top_n=5) -> str:
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[::-1][:top_n]

        explanation = "This account was flagged because:\n"
        for rank, idx in enumerate(top_indices, 1):
            fname = feature_names[idx]
            fval = feature_values[idx]
            shap_val = shap_values[idx]
            direction = "increased" if shap_val > 0 else "decreased"
            desc = self.FEATURE_DESCRIPTIONS.get(fname, fname)
            explanation += (
                f"  {rank}. The {desc} (value: {fval:.4f}) "
                f"{direction} the risk score by {abs(shap_val):.4f}\n"
            )

        return explanation
