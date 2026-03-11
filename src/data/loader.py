import pandas as pd
import logging
from pathlib import Path

from src.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.utils.constants import TRANSACTION_DTYPES, TRANSACTION_RENAME

logger = logging.getLogger(__name__)

TRANSACTION_FILES = [f"transactions_part_{i}.csv" for i in range(6)]

STATIC_FILES = {
    "customers": "customers.csv",
    "accounts": "accounts.csv",
    "linkage": "customer_account_linkage.csv",
    "products": "product_details.csv",
    "labels": "train_labels.csv",
    "test_ids": "test_accounts.csv",
}


def load_transactions(data_dir: Path = None, nrows: int = None) -> pd.DataFrame:
    data_dir = data_dir or DATA_RAW_DIR
    parts = []

    for fname in TRANSACTION_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            logger.warning("File not found, skipping: %s", fpath)
            continue
        df = pd.read_csv(fpath, nrows=nrows)
        parts.append(df)
        logger.info("Loaded %s: %d rows", fname, len(df))

    if not parts:
        logger.warning("No transaction files found in %s", data_dir)
        return pd.DataFrame()

    txn = pd.concat(parts, ignore_index=True)

    # Rename columns to internal pipeline names
    txn.rename(columns=TRANSACTION_RENAME, inplace=True)

    # Parse timestamp and derive fields
    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"], errors="coerce")

    # Derive is_credit from transaction_type (C=credit, D=debit)
    txn["is_credit"] = (txn["transaction_type"] == "C").astype("int8")

    # Ensure amount is numeric
    txn["transaction_amount"] = pd.to_numeric(txn["transaction_amount"], errors="coerce").fillna(0).astype("float32")

    txn.sort_values(["account_id", "transaction_date"], inplace=True)
    txn.reset_index(drop=True, inplace=True)

    mem_mb = txn.memory_usage(deep=True).sum() / 1e6
    logger.info("Transactions loaded: %d rows, %.1f MB", len(txn), mem_mb)
    return txn


def load_static_tables(data_dir: Path = None) -> dict:
    data_dir = data_dir or DATA_RAW_DIR
    tables = {}
    for key, fname in STATIC_FILES.items():
        fpath = data_dir / fname
        if not fpath.exists():
            logger.warning("Static file not found: %s", fpath)
            tables[key] = None
            continue
        tables[key] = pd.read_csv(fpath)
        logger.info("Loaded %s: %d rows", key, len(tables[key]))
    return tables


def load_all(data_dir: Path = None) -> tuple:
    data_dir = data_dir or DATA_RAW_DIR
    txn = load_transactions(data_dir)
    static = load_static_tables(data_dir)
    return txn, static


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    txn, static = load_all()
    print(f"Transactions: {txn.shape}")
    print(f"Columns: {list(txn.columns)}")
    for k, v in static.items():
        print(f"  {k}: {v.shape if v is not None else 'MISSING'}")
