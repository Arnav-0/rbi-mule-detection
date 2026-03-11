import pandas as pd
import numpy as np

from src.data.preprocessor import preprocess_transactions
from src.data.splitter import split_train_val, get_test_accounts, create_cv_folds


def make_txn(n=20):
    return pd.DataFrame({
        "account_id": [f"ACC_{i % 5:03d}" for i in range(n)],
        "transaction_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "transaction_amount": [float(i * 100) for i in range(n)],
        "counterparty_id": [None if i % 5 == 0 else "CP_A" for i in range(n)],
        "is_credit": [i % 2 for i in range(n)],
    })


def make_profile_with_labels(n_mule=5, n_legit=15):
    rows = []
    for i in range(n_mule):
        rows.append({"account_id": f"MULE_{i}", "is_mule": 1, "dataset": "train",
                     "declared_income": 100000, "current_balance": 10000})
    for i in range(n_legit):
        rows.append({"account_id": f"LEGIT_{i}", "is_mule": 0, "dataset": "train",
                     "declared_income": 500000, "current_balance": 100000})
    rows.append({"account_id": "TEST_001", "is_mule": None, "dataset": "test",
                 "declared_income": 200000, "current_balance": 50000})
    return pd.DataFrame(rows)


def test_adds_hour_column():
    txn = make_txn()
    result = preprocess_transactions(txn)
    assert "transaction_hour" in result.columns


def test_adds_weekend_flag():
    # 2024-01-06 is a Saturday
    txn = pd.DataFrame({
        "account_id": ["ACC_001"],
        "transaction_date": [pd.Timestamp("2024-01-06 10:00:00")],
        "transaction_amount": [1000.0],
        "counterparty_id": ["CP_A"],
        "is_credit": [1],
    })
    result = preprocess_transactions(txn)
    assert result["is_weekend"].iloc[0] == 1


def test_fills_null_amounts():
    txn = make_txn()
    txn.loc[0, "transaction_amount"] = float("nan")
    result = preprocess_transactions(txn)
    assert result["transaction_amount"].isnull().sum() == 0


def test_fills_null_counterparty():
    txn = make_txn()
    result = preprocess_transactions(txn)
    assert result["counterparty_id"].isnull().sum() == 0
    assert (result["counterparty_id"] == "UNKNOWN").any()


def test_split_preserves_class_ratio():
    profile = make_profile_with_labels(n_mule=10, n_legit=40)
    train_df, val_df = split_train_val(profile, test_size=0.2)
    original_mule_ratio = 10 / 50
    val_mule_ratio = val_df["is_mule"].mean()
    assert abs(val_mule_ratio - original_mule_ratio) < 0.05


def test_no_test_in_train():
    profile = make_profile_with_labels()
    train_df, val_df = split_train_val(profile)
    all_train_ids = set(train_df["account_id"]) | set(val_df["account_id"])
    assert "TEST_001" not in all_train_ids


def test_get_test_accounts():
    profile = make_profile_with_labels()
    test_df = get_test_accounts(profile)
    assert all(test_df["dataset"] == "test")
    assert "TEST_001" in test_df["account_id"].values


def test_cv_folds_count():
    X = np.random.rand(100, 5)
    y = np.array([0] * 80 + [1] * 20)
    folds = create_cv_folds(X, y, n_splits=5)
    assert len(folds) == 5
