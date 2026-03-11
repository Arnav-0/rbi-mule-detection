import numpy as np
import pandas as pd
from src.data.loader import load_transactions, load_static_tables


def preprocess_transactions(txn: pd.DataFrame) -> pd.DataFrame:
    df = txn.copy()

    df["transaction_amount"] = df["transaction_amount"].fillna(0)
    if "counterparty_id" in df.columns:
        df["counterparty_id"] = df["counterparty_id"].fillna("UNKNOWN")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    df["transaction_hour"] = df["transaction_date"].dt.hour
    df["transaction_dow"] = df["transaction_date"].dt.dayofweek
    df["is_weekend"] = df["transaction_dow"].isin([5, 6]).astype("int8")
    df["is_night"] = df["transaction_hour"].isin([23, 0, 1, 2, 3, 4, 5]).astype("int8")
    df["amount_log"] = np.log1p(df["transaction_amount"].abs())

    # Ensure is_credit exists (derived in loader, but be safe)
    if "is_credit" not in df.columns and "transaction_type" in df.columns:
        df["is_credit"] = (df["transaction_type"] == "C").astype("int8")

    return df


def preprocess_profile(profile: pd.DataFrame) -> pd.DataFrame:
    df = profile.copy()

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN")

    return df


class Preprocessor:
    def load_and_preprocess(self, data_dir=None):
        txn = load_transactions(data_dir)
        static = load_static_tables(data_dir)

        txn = preprocess_transactions(txn)

        # Build profile: accounts + linkage + customers + products
        accounts = static["accounts"]
        linkage = static["linkage"]
        customers = static["customers"]
        products = static["products"]
        labels = static["labels"]

        profile = accounts.merge(linkage, on="account_id", how="left")
        profile = profile.merge(customers, on="customer_id", how="left")
        profile = profile.merge(products, on="customer_id", how="left")
        profile = preprocess_profile(profile)

        return txn, profile, labels
