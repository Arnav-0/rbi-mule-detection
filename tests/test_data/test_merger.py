import pytest
import pandas as pd
from src.data.merger import build_account_profile, add_labels


@pytest.fixture
def static_tables():
    accounts = pd.DataFrame({
        "account_id": ["ACC_001", "ACC_002", "ACC_003"],
        "account_type": ["savings", "current", "savings"],
        "account_opening_date": ["2024-01-01", "2023-06-15", "2025-01-01"],
    })
    linkage = pd.DataFrame({
        "account_id": ["ACC_001", "ACC_002", "ACC_003"],
        "customer_id": ["CUST_A", "CUST_B", "CUST_C"],
    })
    customers = pd.DataFrame({
        "customer_id": ["CUST_A", "CUST_B", "CUST_C"],
        "age": [30, 45, 25],
        "gender": ["M", "F", "M"],
    })
    products = pd.DataFrame({
        "customer_id": ["CUST_A", "CUST_B", "CUST_C"],
        "product_type": ["basic", "premium", "basic"],
    })
    labels = pd.DataFrame({
        "account_id": ["ACC_001", "ACC_002"],
        "is_mule": [1, 0],
    })
    test_ids = pd.DataFrame({
        "account_id": ["ACC_003"],
    })
    return {
        "accounts": accounts,
        "linkage": linkage,
        "customers": customers,
        "products": products,
        "labels": labels,
        "test_ids": test_ids,
    }


@pytest.mark.unit
def test_merge_preserves_row_count(static_tables):
    profile = build_account_profile(static_tables)
    assert len(profile) <= len(static_tables["accounts"]) * 2


@pytest.mark.unit
def test_no_duplicate_account_ids(static_tables):
    profile = build_account_profile(static_tables)
    assert profile["account_id"].is_unique


@pytest.mark.unit
def test_labels_correctly_assigned(static_tables):
    profile = build_account_profile(static_tables)
    profile = add_labels(profile, static_tables["labels"], static_tables["test_ids"])
    assert profile.loc[profile["account_id"] == "ACC_001", "dataset"].iloc[0] == "train"
    assert profile.loc[profile["account_id"] == "ACC_003", "dataset"].iloc[0] == "test"


@pytest.mark.unit
def test_account_age_computed(static_tables):
    profile = build_account_profile(static_tables)
    assert "account_age_days" in profile.columns
    assert (profile["account_age_days"] > 0).any()
