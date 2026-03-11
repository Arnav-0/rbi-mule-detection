import logging
import pandas as pd
from src.utils.config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)


def build_account_profile(static_tables: dict) -> pd.DataFrame:
    accounts = static_tables.get("accounts")
    linkage = static_tables.get("linkage")
    customers = static_tables.get("customers")
    products = static_tables.get("products")

    if accounts is None:
        raise ValueError("accounts table is required")

    profile = accounts.copy()
    logger.info("Starting with accounts: %d rows", len(profile))

    if linkage is not None:
        link_cols = [c for c in linkage.columns if c != "account_id" or c == "account_id"]
        profile = profile.merge(linkage, on="account_id", how="left", suffixes=("", "_linkage"))
        logger.info("After linkage join: %d rows", len(profile))

    if customers is not None and "customer_id" in profile.columns:
        profile = profile.merge(customers, on="customer_id", how="left", suffixes=("", "_cust"))
        logger.info("After customers join: %d rows", len(profile))

    if products is not None and "customer_id" in profile.columns:
        profile = profile.merge(products, on="customer_id", how="left", suffixes=("", "_prod"))
        logger.info("After products join: %d rows", len(profile))

    # Handle both possible column names from different data sources
    open_date_col = None
    for col in ["account_opening_date", "account_open_date"]:
        if col in profile.columns:
            open_date_col = col
            break

    if open_date_col is not None:
        profile[open_date_col] = pd.to_datetime(profile[open_date_col], errors="coerce")
        cutoff = pd.Timestamp("2025-06-30")
        profile["account_age_days"] = (cutoff - profile[open_date_col]).dt.days
        profile["is_new_account"] = (profile["account_age_days"] < 180).astype(int)

    return profile


def add_labels(profile: pd.DataFrame, labels: pd.DataFrame, test_ids: pd.DataFrame) -> pd.DataFrame:
    result = profile.copy()

    if labels is not None:
        label_ids = set(labels["account_id"]) if "account_id" in labels.columns else set()
        result = result.merge(labels, on="account_id", how="left", suffixes=("", "_label"))
    else:
        label_ids = set()

    test_id_set = set()
    if test_ids is not None and "account_id" in test_ids.columns:
        test_id_set = set(test_ids["account_id"])

    def assign_dataset(row):
        if row["account_id"] in label_ids:
            return "train"
        if row["account_id"] in test_id_set:
            return "test"
        return "unlabeled"

    result["dataset"] = result.apply(assign_dataset, axis=1)
    logger.info("Dataset distribution:\n%s", result["dataset"].value_counts().to_string())
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.loader import load_static_tables

    static = load_static_tables()
    profile = build_account_profile(static)
    profile = add_labels(profile, static.get("labels"), static.get("test_ids"))

    out = DATA_PROCESSED_DIR / "merged_accounts.parquet"
    profile.to_parquet(out, index=False)
    print(f"Saved merged profile: {profile.shape}")
    print(f"Dataset distribution:\n{profile['dataset'].value_counts()}")
