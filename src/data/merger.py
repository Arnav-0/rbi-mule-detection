import logging

import pandas as pd

from src.utils.config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

CUTOFF_DATE = pd.Timestamp("2025-06-30")


def build_account_profile(static_tables: dict) -> pd.DataFrame:
    accounts = static_tables.get("accounts")
    linkage = static_tables.get("linkage")
    customers = static_tables.get("customers")
    products = static_tables.get("products")

    if accounts is None:
        raise ValueError("accounts table is required")

    profile = accounts.copy()
    logger.info(f"Starting with accounts: {len(profile)} rows")

    # Join linkage (account_id → customer_id)
    if linkage is not None and "account_id" in linkage.columns:
        before = len(profile)
        profile = profile.merge(linkage, on="account_id", how="left", suffixes=("", "_linkage"))
        logger.info(f"After linkage join: {len(profile)} rows (was {before})")

    # Join customers (customer_id → demographics)
    if customers is not None and "customer_id" in profile.columns and "customer_id" in customers.columns:
        before = len(profile)
        profile = profile.merge(customers, on="customer_id", how="left", suffixes=("", "_customer"))
        logger.info(f"After customer join: {len(profile)} rows (was {before})")

    # Join products (customer_id → product holdings)
    if products is not None and "customer_id" in profile.columns and "customer_id" in products.columns:
        before = len(profile)
        profile = profile.merge(products, on="customer_id", how="left", suffixes=("", "_product"))
        logger.info(f"After products join: {len(profile)} rows (was {before})")

    # Derive account age
    for col in ["account_open_date", "account_opening_date", "open_date"]:
        if col in profile.columns:
            profile[col] = pd.to_datetime(profile[col], errors="coerce")
            profile["account_age_days"] = (CUTOFF_DATE - profile[col]).dt.days.clip(lower=0)
            break
    if "account_age_days" not in profile.columns:
        profile["account_age_days"] = 0

    profile["is_new_account"] = (profile["account_age_days"] < 180).astype(int)

    logger.info(f"Final profile shape: {profile.shape}")
    return profile


def add_labels(profile: pd.DataFrame, labels: pd.DataFrame, test_ids: pd.DataFrame) -> pd.DataFrame:
    result = profile.copy()

    label_ids = set()
    if labels is not None and "account_id" in labels.columns:
        result = result.merge(labels[["account_id", "is_mule"]], on="account_id", how="left")
        label_ids = set(labels["account_id"])
    else:
        result["is_mule"] = None

    test_account_ids = set()
    if test_ids is not None and "account_id" in test_ids.columns:
        test_account_ids = set(test_ids["account_id"])

    def assign_dataset(row):
        if row["account_id"] in label_ids:
            return "train"
        elif row["account_id"] in test_account_ids:
            return "test"
        return "unlabeled"

    result["dataset"] = result.apply(assign_dataset, axis=1)
    logger.info(f"Dataset distribution:\n{result['dataset'].value_counts()}")
    return result


if __name__ == "__main__":
    import logging
    from src.data.loader import load_all
    logging.basicConfig(level=logging.INFO)

    txn, static = load_all()
    profile = build_account_profile(static)
    profile = add_labels(profile, static.get("labels"), static.get("test_ids"))

    print(f"\nMerged profile: {profile.shape}")
    print(profile["dataset"].value_counts())

    out = DATA_PROCESSED_DIR / "merged_accounts.parquet"
    profile.to_parquet(out, index=False)
    print(f"Saved to {out}")
