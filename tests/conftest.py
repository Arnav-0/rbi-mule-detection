import numpy as np
import pandas as pd
import pytest

from src.features.registry import get_all_feature_names


@pytest.fixture
def sample_transactions():
    np.random.seed(42)
    n = 1000
    accounts = np.random.choice(["ACC_001", "ACC_002", "ACC_003", "ACC_004", "ACC_005"], n)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "account_id": accounts,
        "transaction_id": [f"TXN_{i}" for i in range(n)],
        "transaction_date": dates,
        "transaction_amount": np.random.exponential(5000, n).astype("float32"),
        "is_credit": np.random.randint(0, 2, n).astype("int8"),
        "counterparty_id": np.random.choice(["CP_A", "CP_B", "CP_C", "CP_D", "CP_E"], n),
        "transaction_type": pd.Categorical(np.random.choice(["NEFT", "UPI", "IMPS"], n)),
        "channel": pd.Categorical(np.random.choice(["branch", "online", "mobile"], n)),
        "balance_after": np.random.uniform(10000, 500000, n).astype("float32"),
        "transaction_hour": np.random.randint(0, 24, n),
        "transaction_dow": np.random.randint(0, 7, n),
        "is_weekend": np.zeros(n, dtype="int8"),
        "is_night": np.zeros(n, dtype="int8"),
    })


@pytest.fixture
def sample_profile():
    return pd.DataFrame({
        "account_id": ["ACC_001", "ACC_002", "ACC_003", "ACC_004", "ACC_005"],
        "customer_id": ["CUST_001", "CUST_002", "CUST_003", "CUST_004", "CUST_005"],
        "account_type": ["savings", "current", "savings", "current", "savings"],
        "account_age_days": [365, 730, 90, 500, 1200],
        "declared_income": [500000, 1200000, 200000, 800000, 300000],
        "current_balance": [50000, 200000, 10000, 150000, 30000],
        "is_new_account": [0, 0, 1, 0, 0],
        "dataset": ["train", "train", "train", "test", "unlabeled"],
    })


@pytest.fixture
def sample_labels():
    return pd.DataFrame({
        "account_id": ["ACC_001", "ACC_002", "ACC_003"],
        "is_mule": [1, 0, 1],
    })


@pytest.fixture
def sample_features():
    np.random.seed(42)
    n = 100
    account_ids = [f"ACC_{i:04d}" for i in range(n)]
    feature_names = get_all_feature_names()
    data = np.random.rand(n, len(feature_names)).astype("float32")
    df = pd.DataFrame(data, index=account_ids, columns=feature_names)
    df.index.name = "account_id"
    return df
