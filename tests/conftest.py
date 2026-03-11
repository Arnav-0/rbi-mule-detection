import pytest
import numpy as np
import pandas as pd
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
        "customer_id": ["CUST_A", "CUST_B", "CUST_C", "CUST_D", "CUST_E"],
        "account_type": ["savings", "current", "savings", "current", "savings"],
        "account_age_days": [365, 730, 90, 500, 45],
        "declared_income": [500000, 1200000, 300000, 800000, 250000],
        "current_balance": [50000, 200000, 15000, 120000, 8000],
        "is_new_account": [0, 0, 1, 0, 1],
        "dataset": ["train", "train", "train", "test", "test"],
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
    feature_names = get_all_feature_names()
    n_accounts = 100
    data = {feat: np.random.randn(n_accounts).astype("float32") for feat in feature_names}
    data["account_id"] = [f"ACC_{i:04d}" for i in range(n_accounts)]
    df = pd.DataFrame(data)
    df.set_index("account_id", inplace=True)
    return df
