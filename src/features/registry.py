"""Complete registry of all 57 features used in the mule detection system."""

FEATURE_REGISTRY: dict[str, dict] = {
    # VELOCITY (10)
    "txn_count_1d": {"group": "velocity", "description": "Transaction count in last 1 day", "dtype": "float32", "power": "Medium"},
    "txn_count_7d": {"group": "velocity", "description": "Transaction count in last 7 days", "dtype": "float32", "power": "High"},
    "txn_count_30d": {"group": "velocity", "description": "Transaction count in last 30 days", "dtype": "float32", "power": "High"},
    "txn_count_90d": {"group": "velocity", "description": "Transaction count in last 90 days", "dtype": "float32", "power": "Medium"},
    "txn_amount_mean_30d": {"group": "velocity", "description": "Mean transaction amount in last 30 days", "dtype": "float32", "power": "High"},
    "txn_amount_max_30d": {"group": "velocity", "description": "Max transaction amount in last 30 days", "dtype": "float32", "power": "Medium"},
    "txn_amount_std_30d": {"group": "velocity", "description": "Std of transaction amounts in last 30 days", "dtype": "float32", "power": "Medium"},
    "txn_amount_sum_30d": {"group": "velocity", "description": "Total transaction volume in last 30 days", "dtype": "float32", "power": "High"},
    "velocity_acceleration": {"group": "velocity", "description": "7d count vs 30d/4 ratio — burst detection", "dtype": "float32", "power": "High"},
    "frequency_change_ratio": {"group": "velocity", "description": "30d count vs 90d/3 ratio — trend detection", "dtype": "float32", "power": "High"},
    # AMOUNT_PATTERNS (8)
    "round_amount_ratio": {"group": "amount_patterns", "description": "% of transactions with round amounts (1k/5k/10k)", "dtype": "float32", "power": "High"},
    "structuring_score": {"group": "amount_patterns", "description": "% of transactions in [45000, 49999] — below reporting threshold", "dtype": "float32", "power": "High"},
    "structuring_score_broad": {"group": "amount_patterns", "description": "% of transactions in [40000, 49999]", "dtype": "float32", "power": "High"},
    "amount_entropy": {"group": "amount_patterns", "description": "Entropy of amount distribution (low=patterned)", "dtype": "float32", "power": "Medium"},
    "amount_skewness": {"group": "amount_patterns", "description": "Skewness of amount distribution", "dtype": "float32", "power": "Medium"},
    "amount_kurtosis": {"group": "amount_patterns", "description": "Kurtosis of amount distribution", "dtype": "float32", "power": "Medium"},
    "pct_above_10k": {"group": "amount_patterns", "description": "% of transactions above INR 10,000", "dtype": "float32", "power": "High"},
    "amount_concentration": {"group": "amount_patterns", "description": "Gini coefficient of amount distribution", "dtype": "float32", "power": "Medium"},
    # TEMPORAL (8)
    "dormancy_days": {"group": "temporal", "description": "Max gap between transactions if >90 days", "dtype": "float32", "power": "High"},
    "max_gap_days": {"group": "temporal", "description": "Maximum gap (days) between any two consecutive transactions", "dtype": "float32", "power": "High"},
    "burst_after_dormancy": {"group": "temporal", "description": "1 if dormancy then >10 txns in 30d", "dtype": "float32", "power": "High"},
    "unusual_hour_ratio": {"group": "temporal", "description": "% of transactions at night hours (23:00–05:00)", "dtype": "float32", "power": "High"},
    "weekend_ratio": {"group": "temporal", "description": "% of transactions on weekends", "dtype": "float32", "power": "Medium"},
    "night_weekend_combo": {"group": "temporal", "description": "unusual_hour_ratio × weekend_ratio", "dtype": "float32", "power": "Medium"},
    "monthly_txn_cv": {"group": "temporal", "description": "Coefficient of variation of monthly transaction counts", "dtype": "float32", "power": "Medium"},
    "days_to_first_txn": {"group": "temporal", "description": "Days from account open to first transaction", "dtype": "float32", "power": "Medium"},
    # PASSTHROUGH (7)
    "credit_debit_time_delta_median": {"group": "passthrough", "description": "Median hours between credit and next debit", "dtype": "float32", "power": "High"},
    "credit_debit_time_delta_min": {"group": "passthrough", "description": "Minimum hours between credit and next debit", "dtype": "float32", "power": "High"},
    "matched_amount_ratio": {"group": "passthrough", "description": "% of credits matched by similar-amount debit within 24h", "dtype": "float32", "power": "High"},
    "net_flow_ratio": {"group": "passthrough", "description": "Total credits / total debits", "dtype": "float32", "power": "Medium"},
    "rapid_turnover_score": {"group": "passthrough", "description": "Credits quickly followed by debits / total credits", "dtype": "float32", "power": "High"},
    "credit_debit_symmetry": {"group": "passthrough", "description": "1 - |n_credits - n_debits| / (n_credits + n_debits)", "dtype": "float32", "power": "Medium"},
    "max_single_day_volume": {"group": "passthrough", "description": "Maximum total transaction volume in a single day", "dtype": "float32", "power": "Medium"},
    # GRAPH_NETWORK (10)
    "in_degree": {"group": "graph_network", "description": "Number of unique senders to this account", "dtype": "float32", "power": "High"},
    "out_degree": {"group": "graph_network", "description": "Number of unique receivers from this account", "dtype": "float32", "power": "High"},
    "fan_in_ratio": {"group": "graph_network", "description": "in_degree / max(out_degree, 1)", "dtype": "float32", "power": "High"},
    "fan_out_ratio": {"group": "graph_network", "description": "out_degree / max(in_degree, 1)", "dtype": "float32", "power": "High"},
    "betweenness_centrality": {"group": "graph_network", "description": "Betweenness centrality in transaction graph", "dtype": "float32", "power": "High"},
    "pagerank": {"group": "graph_network", "description": "PageRank score in weighted transaction graph", "dtype": "float32", "power": "High"},
    "community_id": {"group": "graph_network", "description": "Louvain community assignment", "dtype": "float32", "power": "Medium"},
    "community_mule_density": {"group": "graph_network", "description": "% of labeled mules in account's community", "dtype": "float32", "power": "High"},
    "clustering_coefficient": {"group": "graph_network", "description": "Local clustering coefficient", "dtype": "float32", "power": "Medium"},
    "total_counterparties": {"group": "graph_network", "description": "Total unique counterparties (in + out degree)", "dtype": "float32", "power": "Medium"},
    # PROFILE_MISMATCH (5)
    "txn_volume_vs_income": {"group": "profile_mismatch", "description": "30d txn volume / declared income", "dtype": "float32", "power": "High"},
    "account_age_vs_activity": {"group": "profile_mismatch", "description": "30d txn count / account age days", "dtype": "float32", "power": "Medium"},
    "avg_txn_vs_balance": {"group": "profile_mismatch", "description": "Mean 30d txn amount / current balance", "dtype": "float32", "power": "High"},
    "product_txn_mismatch": {"group": "profile_mismatch", "description": "1 if savings account with high-value transactions", "dtype": "float32", "power": "High"},
    "balance_volatility": {"group": "profile_mismatch", "description": "Std/mean of daily balance", "dtype": "float32", "power": "Medium"},
    # KYC_BEHAVIORAL (4)
    "mobile_change_flag": {"group": "kyc_behavioral", "description": "1 if mobile number changed", "dtype": "float32", "power": "High"},
    "activity_change_post_mobile": {"group": "kyc_behavioral", "description": "Txn count ratio 30d before vs after mobile change", "dtype": "float32", "power": "High"},
    "kyc_completeness": {"group": "kyc_behavioral", "description": "% of KYC fields filled", "dtype": "float32", "power": "Medium"},
    "linked_account_count": {"group": "kyc_behavioral", "description": "Number of accounts linked to the same customer", "dtype": "float32", "power": "Medium"},
    # INTERACTIONS (5)
    "dormancy_x_burst": {"group": "interactions", "description": "dormancy_days × txn_count_7d", "dtype": "float32", "power": "High"},
    "round_x_structuring": {"group": "interactions", "description": "round_amount_ratio × structuring_score", "dtype": "float32", "power": "High"},
    "fanin_x_passthrough_speed": {"group": "interactions", "description": "fan_in_ratio × (1 / max(credit_debit_time_delta_median, 0.1))", "dtype": "float32", "power": "High"},
    "new_account_x_high_value": {"group": "interactions", "description": "is_new_account × pct_above_10k", "dtype": "float32", "power": "High"},
    "velocity_x_centrality": {"group": "interactions", "description": "velocity_acceleration × betweenness_centrality", "dtype": "float32", "power": "High"},
}


def get_features_by_group(group: str) -> list[str]:
    return [name for name, meta in FEATURE_REGISTRY.items() if meta["group"] == group]


def get_all_feature_names() -> list[str]:
    return list(FEATURE_REGISTRY.keys())


def get_high_power_features() -> list[str]:
    return [name for name, meta in FEATURE_REGISTRY.items() if meta["power"] == "High"]


def print_feature_summary() -> None:
    from collections import Counter
    group_counts = Counter(meta["group"] for meta in FEATURE_REGISTRY.values())
    power_counts = Counter(meta["power"] for meta in FEATURE_REGISTRY.values())
    print(f"Total features: {len(FEATURE_REGISTRY)}")
    print("\nBy group:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count}")
    print("\nBy power:")
    for power, count in sorted(power_counts.items()):
        print(f"  {power}: {count}")
