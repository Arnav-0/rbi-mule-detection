"""Real-time feature computation for a single account from raw transaction data.

Computes all 57 features on the fly so predictions can run on accounts
outside the pre-built feature matrix.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_features_realtime(
    txn_df: pd.DataFrame,
    account_id: str = "NEW_ACCOUNT",
    account_opening_date: str = None,
    avg_balance: float = 0.0,
    cutoff_date=None,
) -> pd.DataFrame:
    """Compute all 57 features for a single account from raw transactions.

    Parameters
    ----------
    txn_df : DataFrame
        Raw transactions with columns: transaction_date (or transaction_timestamp),
        transaction_amount (or amount), transaction_type (or txn_type, 'C'/'D'),
        and optionally counterparty_id.
    account_id : str
        Account identifier to use.
    account_opening_date : str or None
        Account opening date string (for days_to_first_txn, account_age features).
    avg_balance : float
        Average account balance (for profile mismatch features).
    cutoff_date : Timestamp or None
        Cutoff date for feature computation. Defaults to max transaction date.

    Returns
    -------
    DataFrame with 1 row (account_id index) and 57 feature columns.
    """
    txn = txn_df.copy()

    # --- Normalize column names ---
    rename = {
        "transaction_timestamp": "transaction_date",
        "amount": "transaction_amount",
        "txn_type": "transaction_type",
    }
    txn.rename(columns={k: v for k, v in rename.items() if k in txn.columns}, inplace=True)

    # Ensure required columns exist
    if "transaction_date" not in txn.columns:
        raise ValueError("Need 'transaction_date' or 'transaction_timestamp' column")
    if "transaction_amount" not in txn.columns:
        raise ValueError("Need 'transaction_amount' or 'amount' column")

    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"], errors="coerce")
    txn["transaction_amount"] = pd.to_numeric(txn["transaction_amount"], errors="coerce").fillna(0)
    txn = txn.dropna(subset=["transaction_date"])

    if len(txn) == 0:
        raise ValueError("No valid transactions after parsing dates")

    # Derive is_credit
    if "is_credit" not in txn.columns:
        if "transaction_type" in txn.columns:
            txn["is_credit"] = (txn["transaction_type"].str.upper() == "C").astype(int)
        else:
            txn["is_credit"] = 0

    txn["account_id"] = account_id

    if cutoff_date is None:
        cutoff_date = txn["transaction_date"].max()
    cutoff_date = pd.Timestamp(cutoff_date)

    txn = txn[txn["transaction_date"] <= cutoff_date].copy()
    if len(txn) == 0:
        raise ValueError("No transactions before cutoff date")

    # Derived time columns
    txn["transaction_hour"] = txn["transaction_date"].dt.hour
    txn["is_weekend"] = txn["transaction_date"].dt.dayofweek.isin([5, 6]).astype(int)
    txn["is_night"] = txn["transaction_hour"].isin([23, 0, 1, 2, 3, 4, 5]).astype(int)

    result = {}

    # ═══════════════════════════════════════
    # VELOCITY (10 features)
    # ═══════════════════════════════════════
    for window in [1, 7, 30, 90]:
        start = cutoff_date - pd.Timedelta(days=window)
        w_txn = txn[txn["transaction_date"] > start]
        result[f"txn_count_{window}d"] = len(w_txn)
        if window == 30:
            amounts_30d = w_txn["transaction_amount"]
            result["txn_amount_mean_30d"] = amounts_30d.mean() if len(amounts_30d) > 0 else 0
            result["txn_amount_max_30d"] = amounts_30d.max() if len(amounts_30d) > 0 else 0
            result["txn_amount_std_30d"] = amounts_30d.std() if len(amounts_30d) > 1 else 0
            result["txn_amount_sum_30d"] = amounts_30d.sum()

    c30 = result["txn_count_30d"]
    c7 = result["txn_count_7d"]
    c90 = result["txn_count_90d"]
    result["velocity_acceleration"] = c7 / max(c30 / 4, 1)
    result["frequency_change_ratio"] = c30 / max(c90 / 3, 1)

    # ═══════════════════════════════════════
    # AMOUNT PATTERNS (8 features)
    # ═══════════════════════════════════════
    amounts = txn["transaction_amount"]
    n = len(amounts)

    is_round = ((amounts % 1000 == 0) | (amounts % 5000 == 0) | (amounts % 10000 == 0))
    result["round_amount_ratio"] = is_round.mean() if n > 0 else 0

    in_struct = (amounts >= 45000) & (amounts <= 49999)
    result["structuring_score"] = in_struct.mean() if n > 0 else 0

    in_struct_broad = (amounts >= 40000) & (amounts <= 49999)
    result["structuring_score_broad"] = in_struct_broad.mean() if n > 0 else 0

    result["pct_above_10k"] = (amounts > 10000).mean() if n > 0 else 0
    result["amount_skewness"] = amounts.skew() if n >= 3 else 0
    result["amount_kurtosis"] = amounts.kurtosis() if n >= 3 else 0

    if n >= 2:
        hist, _ = np.histogram(amounts, bins=20)
        result["amount_entropy"] = scipy_stats.entropy(hist + 1e-10)
    else:
        result["amount_entropy"] = 0

    arr = np.sort(amounts.values.astype(float))
    if n >= 2 and arr.sum() > 0:
        idx = np.arange(1, n + 1)
        result["amount_concentration"] = (2 * np.sum(idx * arr)) / (n * np.sum(arr)) - (n + 1) / n
    else:
        result["amount_concentration"] = 0

    # ═══════════════════════════════════════
    # TEMPORAL (8 features)
    # ═══════════════════════════════════════
    result["unusual_hour_ratio"] = txn["is_night"].mean() if n > 0 else 0
    result["weekend_ratio"] = txn["is_weekend"].mean() if n > 0 else 0
    result["night_weekend_combo"] = result["unusual_hour_ratio"] * result["weekend_ratio"]

    dates_sorted = txn["transaction_date"].sort_values()
    gaps = dates_sorted.diff().dt.days.dropna()
    result["max_gap_days"] = gaps.max() if len(gaps) > 0 else 0
    result["dormancy_days"] = result["max_gap_days"] if result["max_gap_days"] > 90 else 0

    # Burst after dormancy
    window_30d_start = cutoff_date - pd.Timedelta(days=30)
    recent_count = len(txn[txn["transaction_date"] > window_30d_start])
    result["burst_after_dormancy"] = float(result["dormancy_days"] > 0 and recent_count > 10)

    # Monthly CV
    monthly_counts = txn.set_index("transaction_date").resample("M").size()
    if len(monthly_counts) >= 2:
        result["monthly_txn_cv"] = monthly_counts.std() / max(monthly_counts.mean(), 0.01)
    else:
        result["monthly_txn_cv"] = 0

    # Days to first txn
    if account_opening_date:
        open_dt = pd.to_datetime(account_opening_date, errors="coerce")
        first_txn = dates_sorted.iloc[0] if len(dates_sorted) > 0 else cutoff_date
        result["days_to_first_txn"] = max((first_txn - open_dt).days, 0) if pd.notna(open_dt) else 0
    else:
        result["days_to_first_txn"] = 0

    # ═══════════════════════════════════════
    # PASSTHROUGH (7 features)
    # ═══════════════════════════════════════
    credits = txn[txn["is_credit"] == 1].sort_values("transaction_date")
    debits = txn[txn["is_credit"] == 0].sort_values("transaction_date")

    n_credits = len(credits)
    n_debits = len(debits)
    sum_credits = credits["transaction_amount"].sum()
    sum_debits = debits["transaction_amount"].sum()

    # Daily volume
    daily_vol = txn.groupby(txn["transaction_date"].dt.date)["transaction_amount"].sum()
    result["max_single_day_volume"] = daily_vol.max() if len(daily_vol) > 0 else 0

    result["net_flow_ratio"] = min(sum_credits / max(sum_debits, 1), 100)
    result["credit_debit_symmetry"] = 1 - abs(n_credits - n_debits) / max(n_credits + n_debits, 1)

    # Credit-debit matching
    result["credit_debit_time_delta_median"] = 999.0
    result["credit_debit_time_delta_min"] = 999.0
    result["matched_amount_ratio"] = 0.0
    result["rapid_turnover_score"] = 0.0

    if n_credits > 0 and n_debits > 0:
        merged = pd.merge_asof(
            credits[["transaction_date", "transaction_amount"]].rename(
                columns={"transaction_amount": "credit_amount"}
            ),
            debits[["transaction_date", "transaction_amount"]].rename(
                columns={"transaction_date": "debit_date", "transaction_amount": "debit_amount"}
            ),
            left_on="transaction_date",
            right_on="debit_date",
            direction="forward",
        ).dropna(subset=["debit_date"])

        if len(merged) > 0:
            merged["delta_hours"] = (merged["debit_date"] - merged["transaction_date"]).dt.total_seconds() / 3600
            merged["amt_diff_ratio"] = (merged["credit_amount"] - merged["debit_amount"]).abs() / merged["credit_amount"].clip(lower=1)
            merged["matched_24h"] = (merged["amt_diff_ratio"] < 0.05) & (merged["delta_hours"] < 24)
            merged["rapid_48h"] = (merged["amt_diff_ratio"] < 0.05) & (merged["delta_hours"] < 48)

            result["credit_debit_time_delta_median"] = merged["delta_hours"].median()
            result["credit_debit_time_delta_min"] = merged["delta_hours"].min()
            result["matched_amount_ratio"] = merged["matched_24h"].sum() / max(n_credits, 1)
            rapid_amount = merged.loc[merged["rapid_48h"], "credit_amount"].sum()
            result["rapid_turnover_score"] = rapid_amount / max(sum_credits, 1)

    # ═══════════════════════════════════════
    # GRAPH NETWORK (10 features) — simplified for single account
    # ═══════════════════════════════════════
    if "counterparty_id" in txn.columns:
        cp = txn.dropna(subset=["counterparty_id"])
        in_cp = cp[cp["is_credit"] == 1]["counterparty_id"].nunique()
        out_cp = cp[cp["is_credit"] == 0]["counterparty_id"].nunique()
    else:
        in_cp = 0
        out_cp = 0

    result["in_degree"] = in_cp
    result["out_degree"] = out_cp
    result["fan_in_ratio"] = in_cp / max(out_cp, 1)
    result["fan_out_ratio"] = out_cp / max(in_cp, 1)
    result["total_counterparties"] = in_cp + out_cp
    # Cannot compute graph-global metrics for a single isolated account
    result["betweenness_centrality"] = 0.0
    result["pagerank"] = 0.0
    result["community_id"] = 0
    result["community_mule_density"] = 0.0
    result["clustering_coefficient"] = 0.0

    # ═══════════════════════════════════════
    # PROFILE MISMATCH (5 features)
    # ═══════════════════════════════════════
    sum_30d = result["txn_amount_sum_30d"]
    count_30d = result["txn_count_30d"]
    mean_30d = result["txn_amount_mean_30d"]
    bal = max(abs(avg_balance), 1)

    result["txn_volume_vs_income"] = sum_30d / bal
    if account_opening_date:
        open_dt = pd.to_datetime(account_opening_date, errors="coerce")
        age_days = max((cutoff_date - open_dt).days, 1) if pd.notna(open_dt) else 1
    else:
        age_days = 1
    result["account_age_vs_activity"] = count_30d / age_days
    result["avg_txn_vs_balance"] = mean_30d / bal
    result["product_txn_mismatch"] = 0.0  # No product info for external accounts
    result["balance_volatility"] = 0.0  # No daily/monthly balance for external accounts

    # ═══════════════════════════════════════
    # KYC BEHAVIORAL (4 features)
    # ═══════════════════════════════════════
    result["mobile_change_flag"] = 0.0
    result["activity_change_post_mobile"] = 0.0
    result["kyc_completeness"] = 0.0
    result["linked_account_count"] = 1.0

    # ═══════════════════════════════════════
    # INTERACTIONS (5 features)
    # ═══════════════════════════════════════
    result["dormancy_x_burst"] = result["dormancy_days"] * result["txn_count_7d"]
    result["round_x_structuring"] = result["round_amount_ratio"] * result["structuring_score"]
    inv_speed = 1 / max(result["credit_debit_time_delta_median"], 0.1)
    result["fanin_x_passthrough_speed"] = result["fan_in_ratio"] * inv_speed
    is_new = float(age_days < 90) if account_opening_date else 0.0
    result["new_account_x_high_value"] = is_new * result["pct_above_10k"]
    result["velocity_x_centrality"] = result["velocity_acceleration"] * result["betweenness_centrality"]

    # Build final DataFrame
    feature_df = pd.DataFrame([result], index=pd.Index([account_id], name="account_id"))

    # Ensure correct column order (match the trained model)
    from src.features.registry import get_all_feature_names
    expected_cols = get_all_feature_names()
    for col in expected_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
    feature_df = feature_df[expected_cols]

    feature_df = feature_df.fillna(0).replace([np.inf, -np.inf], 0)
    return feature_df
