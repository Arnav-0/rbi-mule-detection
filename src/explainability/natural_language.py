"""Natural language explanations for mule account flagging decisions."""
from __future__ import annotations

import numpy as np

FEATURE_DESCRIPTIONS = {
    # Velocity features
    "txn_count_30d": "high transaction volume in the last 30 days",
    "txn_count_7d": "high transaction volume in the last 7 days",
    "txn_count_90d": "high transaction volume in the last 90 days",
    "credit_count_30d": "unusually high number of incoming credits",
    "debit_count_30d": "unusually high number of outgoing debits",
    "unique_counterparties_30d": "transactions with many distinct counterparties",
    "credit_velocity_7d": "rapid incoming credit activity in the past week",
    "debit_velocity_7d": "rapid outgoing debit activity in the past week",
    "txn_velocity_ratio": "sudden acceleration in transaction pace compared to history",
    # Amount pattern features
    "matched_amount_ratio": "credits were followed by matching debits within 24 hours",
    "round_amount_ratio": "high proportion of round-number transactions (structuring indicator)",
    "avg_credit_amount": "unusually large average credit amounts",
    "avg_debit_amount": "unusually large average debit amounts",
    "amount_std": "high variability in transaction amounts",
    "max_single_txn": "very large individual transaction detected",
    "credit_debit_ratio": "nearly equal credits and debits (pass-through pattern)",
    "small_txn_ratio": "high proportion of small transactions (smurfing indicator)",
    "large_txn_ratio": "high proportion of large transactions",
    "amount_entropy": "irregular distribution of transaction amounts",
    "structuring_score": "transaction amounts just below reporting thresholds",
    # Temporal features
    "night_txn_ratio": "high proportion of transactions during night hours (10pm–6am)",
    "weekend_txn_ratio": "high proportion of transactions on weekends",
    "txn_hour_entropy": "transactions spread across many different hours",
    "txn_day_entropy": "transactions spread across many different days",
    "burst_score": "transactions clustered into short bursts of activity",
    "dormancy_break": "account reactivated after a long period of inactivity",
    "active_days_ratio": "account active on most days (potential automation)",
    "peak_hour_concentration": "most transactions concentrated in a single hour",
    # Profile mismatch features
    "age_income_mismatch": "income level inconsistent with account holder's age",
    "occupation_txn_mismatch": "transaction patterns inconsistent with stated occupation",
    "geography_mismatch": "transactions from locations far from registered address",
    "account_age_days": "newly opened account (higher risk)",
    "kyc_completeness": "incomplete KYC documentation on file",
    "address_change_count": "multiple address changes registered recently",
    "contact_change_count": "multiple contact detail changes registered recently",
    "profile_update_frequency": "unusually frequent profile updates",
    # KYC / behavioral features
    "kyc_flag_count": "multiple KYC flags raised by compliance team",
    "pep_exposure": "account holder has politically exposed person connections",
    "adverse_media_score": "adverse media mentions associated with account",
    "sanction_proximity": "proximity to sanctioned entities in transaction network",
    "high_risk_country_txn": "transactions involving high-risk jurisdictions",
    "cash_equivalent_ratio": "high proportion of cash-equivalent transactions",
    "nominee_account_flag": "account shows signs of being operated by a nominee",
    # Graph / network features
    "in_degree": "receives funds from many different accounts",
    "out_degree": "sends funds to many different accounts",
    "betweenness_centrality": "acts as a central hub in the transaction network",
    "clustering_coefficient": "embedded in a tightly-knit group of accounts",
    "connected_mule_ratio": "directly connected to accounts flagged as mules",
    "network_risk_score": "high aggregate risk score from network neighbors",
    "community_mule_density": "belongs to a community with many mule accounts",
    # Interaction / derived features
    "velocity_amount_interaction": "combination of high velocity and large amounts",
    "night_round_interaction": "round-amount transactions during night hours",
    "network_behavioral_score": "combined network and behavioral risk signal",
    "composite_risk_score": "overall composite risk score across all dimensions",
    "mule_probability_prior": "prior mule probability from rule-based system",
    "passthrough_score": "funds flow through account rapidly without accumulation",
    "layering_score": "multiple layers of fund movement detected",
}


class NaturalLanguageExplainer:
    def __init__(self):
        self.feature_descriptions = FEATURE_DESCRIPTIONS

    def explain(
        self,
        shap_values: np.ndarray,
        feature_values,
        feature_names: list[str],
        top_n: int = 5,
    ) -> str:
        """Generate a plain-English explanation for a flagged account.

        Returns a formatted string with the top N reasons.
        """
        vals = np.asarray(shap_values)
        top_idx = np.argsort(np.abs(vals))[::-1][:top_n]

        reasons = []
        for rank, idx in enumerate(top_idx, start=1):
            fname = feature_names[idx]
            fval = feature_values[idx] if hasattr(feature_values, "__getitem__") else getattr(feature_values, fname, "N/A")
            shap_val = vals[idx]
            direction = "increased" if shap_val > 0 else "decreased"

            description = self.feature_descriptions.get(fname, fname.replace("_", " "))
            if isinstance(fval, float):
                fval_str = f"{fval:.3f}"
            else:
                fval_str = str(fval)

            reasons.append(
                f"  {rank}. {description.capitalize()} "
                f"(value: {fval_str}, {direction} risk by {abs(shap_val):.3f})"
            )

        header = "This account was flagged because:\n"
        return header + "\n".join(reasons)

    def explain_batch(
        self,
        shap_values: np.ndarray,
        feature_matrix,
        feature_names: list[str],
        account_ids: list[str],
        top_n: int = 5,
    ) -> dict[str, str]:
        """Generate explanations for multiple accounts."""
        explanations = {}
        for i, account_id in enumerate(account_ids):
            fvals = feature_matrix[i] if hasattr(feature_matrix, "__getitem__") else feature_matrix.iloc[i].values
            explanations[account_id] = self.explain(shap_values[i], fvals, feature_names, top_n)
        return explanations
