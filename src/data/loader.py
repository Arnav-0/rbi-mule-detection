import logging
from pathlib import Path

import pandas as pd

from src.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.utils.constants import TRANSACTION_DTYPES

logger = logging.getLogger(__name__)


def load_transactions(data_dir: Path = None, nrows: int = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    # Map from actual CSV column names to expected names
    COLUMN_RENAME = {
        "transaction_timestamp": "transaction_date",
        "amount": "transaction_amount",
        "txn_type": "transaction_type",
    }

    parts = []
    for i in range(6):
        path = data_dir / f"transactions_part_{i}.csv"
        try:
            df = pd.read_csv(path, nrows=nrows, low_memory=False)
            df.rename(columns=COLUMN_RENAME, inplace=True)
            if "transaction_date" in df.columns:
                df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
            # Apply dtypes where columns exist
            for col, dtype in TRANSACTION_DTYPES.items():
                if col in df.columns and col != "transaction_date":
                    try:
                        df[col] = df[col].astype(dtype)
                    except (ValueError, TypeError):
                        pass
            logger.info(f"Loaded {path.name}: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
            parts.append(df)
        except FileNotFoundError:
            logger.warning(f"File not found, skipping: {path}")

    if not parts:
        logger.warning("No transaction files found — returning empty DataFrame")
        return pd.DataFrame()

    txn = pd.concat(parts, ignore_index=True)
    txn = txn.sort_values(["account_id", "transaction_date"]).reset_index(drop=True)
    logger.info(f"Total transactions: {len(txn):,} rows, {txn.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return txn


def load_static_tables(data_dir: Path = None) -> dict:
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    files = {
        "customers": "customers.csv",
        "accounts": "accounts.csv",
        "linkage": "customer_account_linkage.csv",
        "products": "product_details.csv",
        "labels": "train_labels.csv",
        "test_ids": "test_accounts.csv",
    }

    tables = {}
    for key, filename in files.items():
        path = data_dir / filename
        try:
            tables[key] = pd.read_csv(path)
            logger.info(f"Loaded {filename}: {len(tables[key]):,} rows")
        except FileNotFoundError:
            logger.warning(f"Static table not found: {path}")
            tables[key] = None

    return tables


def load_all(data_dir: Path = None):
    txn = load_transactions(data_dir)
    static = load_static_tables(data_dir)
    return txn, static


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    txn, static = load_all()
    print(f"\nTransactions: {txn.shape}")
    for name, df in static.items():
        if df is not None:
            print(f"{name}: {df.shape}, {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    if len(txn) > 0:
        out = DATA_PROCESSED_DIR / "transactions_clean.parquet"
        txn.to_parquet(out, index=False)
        print(f"\nSaved to {out}")
