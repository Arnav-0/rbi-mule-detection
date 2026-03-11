import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import preprocess_transactions
from src.data.splitter import split_train_val


@pytest.mark.unit
def test_adds_hour_column():
    df = pd.DataFrame({
        "account_id": ["A"],
        "transaction_date": pd.to_datetime(["2024-01-01 14:30:00"]),
        "transaction_amount": [100.0],
        "counterparty_id": ["CP_A"],
    })
    result = preprocess_transactions(df)
    assert "transaction_hour" in result.columns
    assert result["transaction_hour"].iloc[0] == 14


@pytest.mark.unit
def test_adds_weekend_flag():
    # 2024-01-06 is a Saturday
    df = pd.DataFrame({
        "account_id": ["A"],
        "transaction_date": pd.to_datetime(["2024-01-06 10:00:00"]),
        "transaction_amount": [100.0],
        "counterparty_id": ["CP_A"],
    })
    result = preprocess_transactions(df)
    assert result["is_weekend"].iloc[0] == 1


@pytest.mark.unit
def test_fills_null_amounts():
    df = pd.DataFrame({
        "account_id": ["A"],
        "transaction_date": pd.to_datetime(["2024-01-01"]),
        "transaction_amount": [np.nan],
        "counterparty_id": ["CP_A"],
    })
    result = preprocess_transactions(df)
    assert result["transaction_amount"].iloc[0] == 0.0


@pytest.mark.unit
def test_split_preserves_class_ratio():
    n = 200
    profile = pd.DataFrame({
        "account_id": [f"ACC_{i}" for i in range(n)],
        "is_mule": [1] * 20 + [0] * 180,
        "dataset": ["train"] * n,
    })
    train_df, val_df = split_train_val(profile)
    train_ratio = train_df["is_mule"].mean()
    val_ratio = val_df["is_mule"].mean()
    assert abs(train_ratio - val_ratio) < 0.02


@pytest.mark.unit
def test_no_test_in_train():
    profile = pd.DataFrame({
        "account_id": [f"ACC_{i}" for i in range(22)],
        "is_mule": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "dataset": ["train"] * 20 + ["test", "test"],
    })
    train_df, val_df = split_train_val(profile)
    test_ids = {"ACC_20", "ACC_21"}
    assert test_ids.isdisjoint(set(train_df["account_id"].values))
    assert test_ids.isdisjoint(set(val_df["account_id"].values))
