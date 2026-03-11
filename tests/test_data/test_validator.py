import pytest
import pandas as pd
import numpy as np
from src.data.validator import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


@pytest.mark.unit
def test_detects_missing_columns(validator):
    df = pd.DataFrame({"transaction_amount": [100], "is_credit": [1]})
    report = validator.validate_transactions(df)
    assert "account_id" in report["missing_columns"]
    assert "transaction_date" in report["missing_columns"]


@pytest.mark.unit
def test_detects_null_percentages(validator):
    df = pd.DataFrame({
        "account_id": ["A", "B", None],
        "transaction_date": pd.to_datetime(["2024-01-01", None, "2024-01-03"]),
        "transaction_amount": [100.0, 200.0, np.nan],
        "is_credit": [1, 0, 1],
    })
    report = validator.validate_transactions(df)
    assert report["null_pct"]["account_id"] > 0
    assert report["null_pct"]["transaction_amount"] > 0


@pytest.mark.unit
def test_validate_labels_counts(validator):
    labels = pd.DataFrame({
        "account_id": [f"ACC_{i}" for i in range(10)],
        "is_mule": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    })
    report = validator.validate_labels(labels)
    assert report["mule_count"] == 3
    assert report["mule_pct"] == 30.0


@pytest.mark.unit
def test_full_validation_runs(validator):
    txn = pd.DataFrame({
        "account_id": ["A", "B"],
        "transaction_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "transaction_amount": [100.0, 200.0],
        "is_credit": [1, 0],
    })
    report = validator.run_full_validation(txn=txn)
    assert "transactions" in report
