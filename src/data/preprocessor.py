import numpy as np
import pandas as pd


def preprocess_transactions(txn: pd.DataFrame) -> pd.DataFrame:
    df = txn.copy()

    if "transaction_amount" in df.columns:
        df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce").fillna(0)

    if "counterparty_id" in df.columns:
        df["counterparty_id"] = df["counterparty_id"].fillna("UNKNOWN")

    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    if "transaction_date" in df.columns:
        df["transaction_hour"] = df["transaction_date"].dt.hour
        df["transaction_dow"] = df["transaction_date"].dt.dayofweek  # 0=Mon, 6=Sun
        df["is_weekend"] = df["transaction_dow"].isin([5, 6]).astype("int8")
        df["is_night"] = df["transaction_hour"].isin([23, 0, 1, 2, 3, 4, 5]).astype("int8")

    if "transaction_amount" in df.columns:
        df["amount_log"] = np.log1p(df["transaction_amount"])

    return df


def preprocess_profile(profile: pd.DataFrame) -> pd.DataFrame:
    df = profile.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN")

    return df
