FEATURE_REGISTRY = {
    # VELOCITY (10)
    "txn_count_1d": {"group": "velocity", "description": "Transaction count in last 1 day", "dtype": "int32", "power": "High"},
    "txn_count_7d": {"group": "velocity", "description": "Transaction count in last 7 days", "dtype": "int32", "power": "High"},
    "txn_count_30d": {"group": "velocity", "description": "Transaction count in last 30 days", "dtype": "int32", "power": "High"},
    "txn_count_90d": {"group": "velocity", "description": "Transaction count in last 90 days", "dtype": "int32", "power": "Medium"},
    "txn_amount_mean_30d": {"group": "velocity", "description": "Mean transaction amount in last 30 days", "dtype": "float32", "power": "High"},
    "txn_amount_max_30d": {"group": "velocity", "description": "Max transaction amount in last 30 days", "dtype": "float32", "power": "Medium"},
    "txn_amount_std_30d": {"group": "velocity", "description": "Std dev of transaction amount in last 30 days", "dtype": "float32", "power": "Medium"},
    "txn_amount_sum_30d": {"group": "velocity", "description": "Sum of transaction amounts in last 30 days", "dtype": "float32", "power": "High"},
    "velocity_acceleration": {"group": "velocity", "description": "Ratio of 7d to 30d transaction rate", "dtype": "float32", "power": "High"},
    "frequency_change_ratio": {"group": "velocity", "description": "Ratio of recent to historical frequency", "dtype": "float32", "power": "High"},
    # AMOUNT_PATTERNS (8)
    "round_amount_ratio": {"group": "amount_patterns", "description": "Fraction of transactions with round amounts", "dtype": "float32", "power": "High"},
    "structuring_score": {"group": "amount_patterns", "description": "Score for amounts just below reporting threshold", "dtype": "float32", "power": "High"},
    "structuring_score_broad": {"group": "amount_patterns", "description": "Broader structuring detection score", "dtype": "float32", "power": "Medium"},
    "amount_entropy": {"group": "amount_patterns", "description": "Entropy of binned transaction amounts", "dtype": "float32", "power": "Medium"},
    "amount_skewness": {"group": "amount_patterns", "description": "Skewness of transaction amounts", "dtype": "float32", "power": "Medium"},
    "amount_kurtosis": {"group": "amount_patterns", "description": "Kurtosis of transaction amounts", "dtype": "float32", "power": "Medium"},
    "pct_above_10k": {"group": "amount_patterns", "description": "Percent of transactions above 10,000", "dtype": "float32", "power": "Medium"},
    "amount_concentration": {"group": "amount_patterns", "description": "HHI concentration of transaction amounts", "dtype": "float32", "power": "Medium"},
    # TEMPORAL (8)
    "dormancy_days": {"group": "temporal", "description": "Max consecutive days without transactions", "dtype": "int32", "power": "High"},
    "max_gap_days": {"group": "temporal", "description": "Maximum gap between consecutive transactions", "dtype": "float32", "power": "High"},
    "burst_after_dormancy": {"group": "temporal", "description": "Transaction burst score after dormancy period", "dtype": "float32", "power": "High"},
    "unusual_hour_ratio": {"group": "temporal", "description": "Fraction of transactions at unusual hours", "dtype": "float32", "power": "Medium"},
    "weekend_ratio": {"group": "temporal", "description": "Fraction of transactions on weekends", "dtype": "float32", "power": "Medium"},
    "night_weekend_combo": {"group": "temporal", "description": "Fraction of transactions at night on weekends", "dtype": "float32", "power": "Medium"},
    "monthly_txn_cv": {"group": "temporal", "description": "CV of monthly transaction counts", "dtype": "float32", "power": "Medium"},
    "days_to_first_txn": {"group": "temporal", "description": "Days from account open to first transaction", "dtype": "int32", "power": "Medium"},
    # PASSTHROUGH (7)
    "credit_debit_time_delta_median": {"group": "passthrough", "description": "Median time between credit and following debit", "dtype": "float32", "power": "High"},
    "credit_debit_time_delta_min": {"group": "passthrough", "description": "Minimum time between credit and following debit", "dtype": "float32", "power": "High"},
    "matched_amount_ratio": {"group": "passthrough", "description": "Ratio of credits with matching debit amounts", "dtype": "float32", "power": "High"},
    "net_flow_ratio": {"group": "passthrough", "description": "Net flow as ratio of total volume", "dtype": "float32", "power": "High"},
    "rapid_turnover_score": {"group": "passthrough", "description": "Score for rapid credit-to-debit turnover", "dtype": "float32", "power": "High"},
    "credit_debit_symmetry": {"group": "passthrough", "description": "Symmetry between credit and debit volumes", "dtype": "float32", "power": "Medium"},
    "max_single_day_volume": {"group": "passthrough", "description": "Maximum total volume in a single day", "dtype": "float32", "power": "Medium"},
    # GRAPH_NETWORK (10)
    "in_degree": {"group": "graph_network", "description": "Number of unique incoming counterparties", "dtype": "int32", "power": "High"},
    "out_degree": {"group": "graph_network", "description": "Number of unique outgoing counterparties", "dtype": "int32", "power": "High"},
    "fan_in_ratio": {"group": "graph_network", "description": "In-degree relative to total degree", "dtype": "float32", "power": "High"},
    "fan_out_ratio": {"group": "graph_network", "description": "Out-degree relative to total degree", "dtype": "float32", "power": "High"},
    "betweenness_centrality": {"group": "graph_network", "description": "Betweenness centrality in transaction graph", "dtype": "float32", "power": "High"},
    "pagerank": {"group": "graph_network", "description": "PageRank score in transaction graph", "dtype": "float32", "power": "High"},
    "community_id": {"group": "graph_network", "description": "Community ID from Louvain clustering", "dtype": "int32", "power": "Medium"},
    "community_mule_density": {"group": "graph_network", "description": "Fraction of mules in same community", "dtype": "float32", "power": "High"},
    "clustering_coefficient": {"group": "graph_network", "description": "Local clustering coefficient", "dtype": "float32", "power": "Medium"},
    "total_counterparties": {"group": "graph_network", "description": "Total unique counterparties", "dtype": "int32", "power": "Medium"},
    # PROFILE_MISMATCH (5)
    "txn_volume_vs_income": {"group": "profile_mismatch", "description": "Transaction volume relative to declared income", "dtype": "float32", "power": "High"},
    "account_age_vs_activity": {"group": "profile_mismatch", "description": "Activity level relative to account age", "dtype": "float32", "power": "Medium"},
    "avg_txn_vs_balance": {"group": "profile_mismatch", "description": "Average transaction size relative to balance", "dtype": "float32", "power": "Medium"},
    "product_txn_mismatch": {"group": "profile_mismatch", "description": "Mismatch between product type and transaction pattern", "dtype": "float32", "power": "Medium"},
    "balance_volatility": {"group": "profile_mismatch", "description": "Volatility of account balance over time", "dtype": "float32", "power": "Medium"},
    # KYC_BEHAVIORAL (4)
    "mobile_change_flag": {"group": "kyc_behavioral", "description": "Whether mobile number was recently changed", "dtype": "int8", "power": "Medium"},
    "activity_change_post_mobile": {"group": "kyc_behavioral", "description": "Activity change after mobile number update", "dtype": "float32", "power": "Medium"},
    "kyc_completeness": {"group": "kyc_behavioral", "description": "Completeness score of KYC information", "dtype": "float32", "power": "Medium"},
    "linked_account_count": {"group": "kyc_behavioral", "description": "Number of linked accounts for same customer", "dtype": "int32", "power": "Medium"},
    # INTERACTIONS (5)
    "dormancy_x_burst": {"group": "interactions", "description": "Interaction: dormancy days * burst score", "dtype": "float32", "power": "High"},
    "round_x_structuring": {"group": "interactions", "description": "Interaction: round amount ratio * structuring score", "dtype": "float32", "power": "High"},
    "fanin_x_passthrough_speed": {"group": "interactions", "description": "Interaction: fan-in ratio * rapid turnover", "dtype": "float32", "power": "High"},
    "new_account_x_high_value": {"group": "interactions", "description": "Interaction: new account flag * high value ratio", "dtype": "float32", "power": "High"},
    "velocity_x_centrality": {"group": "interactions", "description": "Interaction: velocity * betweenness centrality", "dtype": "float32", "power": "High"},
}


def get_features_by_group(group: str) -> list[str]:
    return [name for name, meta in FEATURE_REGISTRY.items() if meta["group"] == group]


def get_all_feature_names() -> list[str]:
    return list(FEATURE_REGISTRY.keys())


def get_high_power_features() -> list[str]:
    return [name for name, meta in FEATURE_REGISTRY.items() if meta["power"] == "High"]


def print_feature_summary():
    from collections import Counter
    groups = Counter(meta["group"] for meta in FEATURE_REGISTRY.values())
    print(f"Total features: {len(FEATURE_REGISTRY)}")
    for group, count in sorted(groups.items()):
        print(f"  {group}: {count}")
    high = len(get_high_power_features())
    print(f"High power: {high}, Medium power: {len(FEATURE_REGISTRY) - high}")
