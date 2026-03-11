"""Extended validator tests: validate_accounts, validate_consistency, edge cases."""
import pandas as pd
import pytest

from src.data.validator import DataValidator


@pytest.fixture
def v():
    return DataValidator()


# ── validate_accounts ─────────────────────────────────────────────────────────

def test_validate_accounts_basic(v):
    accounts = pd.DataFrame({
        "account_id": ["A1", "A2", "A3"],
        "account_type": ["savings", "current", "savings"],
        "balance": [10000.0, 50000.0, None],
    })
    report = v.validate_accounts(accounts)
    assert report["total_rows"] == 3
    assert report["duplicate_account_ids"] == 0
    assert "savings" in report["account_type_distribution"]


def test_validate_accounts_detects_duplicates(v):
    accounts = pd.DataFrame({
        "account_id": ["A1", "A1", "A2"],
        "account_type": ["savings", "savings", "current"],
    })
    report = v.validate_accounts(accounts)
    assert report["duplicate_account_ids"] == 1


def test_validate_accounts_none_returns_error(v):
    report = v.validate_accounts(None)
    assert "error" in report


# ── validate_consistency ──────────────────────────────────────────────────────

def test_validate_consistency_accounts_without_linkage(v):
    static = {
        "accounts": pd.DataFrame({"account_id": ["A1", "A2", "A3"]}),
        "linkage": pd.DataFrame({"account_id": ["A1"]}),
        "labels": None,
    }
    report = v.validate_consistency(static)
    assert report["accounts_without_linkage"] == 2


def test_validate_consistency_linkage_without_accounts(v):
    static = {
        "accounts": pd.DataFrame({"account_id": ["A1"]}),
        "linkage": pd.DataFrame({"account_id": ["A1", "GHOST"]}),
        "labels": None,
    }
    report = v.validate_consistency(static)
    assert report["linkage_without_accounts"] == 1


def test_validate_consistency_labels_without_accounts(v):
    static = {
        "accounts": pd.DataFrame({"account_id": ["A1"]}),
        "linkage": None,
        "labels": pd.DataFrame({"account_id": ["A1", "ORPHAN"], "is_mule": [0, 1]}),
    }
    report = v.validate_consistency(static)
    assert report["labels_without_accounts"] == 1


def test_validate_consistency_all_null_tables(v):
    static = {"accounts": None, "linkage": None, "labels": None}
    report = v.validate_consistency(static)
    assert isinstance(report, dict)  # should not crash


# ── validate_transactions edge cases ──────────────────────────────────────────

def test_validate_transactions_empty_df(v):
    df = pd.DataFrame(columns=["account_id", "transaction_date", "transaction_amount", "is_credit"])
    report = v.validate_transactions(df)
    assert report["total_rows"] == 0
    assert report["missing_columns"] == []


def test_validate_transactions_detects_negatives(v):
    df = pd.DataFrame({
        "account_id": ["A1", "A2"],
        "transaction_date": [pd.Timestamp("2024-01-01")] * 2,
        "transaction_amount": [-500.0, 1000.0],
        "is_credit": [0, 1],
    })
    report = v.validate_transactions(df)
    assert report["negative_amounts"] == 1


def test_validate_transactions_detects_zeros(v):
    df = pd.DataFrame({
        "account_id": ["A1"],
        "transaction_date": [pd.Timestamp("2024-01-01")],
        "transaction_amount": [0.0],
        "is_credit": [1],
    })
    report = v.validate_transactions(df)
    assert report["zero_amounts"] == 1


# ── full validation with all tables ──────────────────────────────────────────

def test_full_validation_all_tables(v):
    txn = pd.DataFrame({
        "account_id": ["A1", "A2"],
        "transaction_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01")],
        "transaction_amount": [1000.0, 5000.0],
        "is_credit": [1, 0],
    })
    static = {
        "accounts": pd.DataFrame({"account_id": ["A1", "A2"], "account_type": ["savings", "current"]}),
        "linkage": pd.DataFrame({"account_id": ["A1", "A2"], "customer_id": ["C1", "C2"]}),
        "customers": None,
        "products": None,
        "labels": pd.DataFrame({"account_id": ["A1"], "is_mule": [1]}),
        "test_ids": None,
    }
    report = v.run_full_validation(txn, static)
    assert "transactions" in report
    assert "accounts" in report
    assert "labels" in report
    assert report["labels"]["mule_count"] == 1
