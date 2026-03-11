import pandas as pd
import pytest

from src.data.validator import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


def test_detects_missing_columns(validator):
    df = pd.DataFrame({"transaction_date": [], "transaction_amount": [], "is_credit": []})
    report = validator.validate_transactions(df)
    assert "account_id" in report["missing_columns"]


def test_detects_null_percentages(validator):
    df = pd.DataFrame({
        "account_id": ["ACC_001", None],
        "transaction_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        "transaction_amount": [100.0, 200.0],
        "is_credit": [1, 0],
    })
    report = validator.validate_transactions(df)
    assert report["null_pct"]["account_id"] > 0


def test_validate_labels_counts(validator):
    labels = pd.DataFrame({
        "account_id": [f"ACC_{i}" for i in range(10)],
        "is_mule": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    })
    report = validator.validate_labels(labels)
    assert report["mule_count"] == 3
    assert report["legitimate_count"] == 7
    assert report["mule_pct"] == 30.0


def test_full_validation_runs(validator):
    txn = pd.DataFrame({
        "account_id": ["ACC_001", "ACC_002"],
        "transaction_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        "transaction_amount": [100.0, 200.0],
        "is_credit": [1, 0],
    })
    static = {
        "accounts": pd.DataFrame({"account_id": ["ACC_001"]}),
        "labels": pd.DataFrame({"account_id": ["ACC_001"], "is_mule": [1]}),
        "linkage": None, "customers": None, "products": None, "test_ids": None,
    }
    report = validator.run_full_validation(txn, static)
    assert "transactions" in report
    assert "labels" in report
