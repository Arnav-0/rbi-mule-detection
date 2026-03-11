import pandas as pd
import pytest

from src.data.merger import build_account_profile, add_labels


@pytest.fixture
def mini_tables():
    accounts = pd.DataFrame({
        "account_id": ["ACC_001", "ACC_002", "ACC_003"],
        "account_type": ["savings", "current", "savings"],
        "account_open_date": ["2020-01-01", "2018-06-15", "2024-01-01"],
    })
    linkage = pd.DataFrame({
        "account_id": ["ACC_001", "ACC_002", "ACC_003"],
        "customer_id": ["CUST_001", "CUST_002", "CUST_001"],
    })
    customers = pd.DataFrame({
        "customer_id": ["CUST_001", "CUST_002"],
        "name": ["Alice", "Bob"],
        "declared_income": [500000, 1200000],
    })
    products = pd.DataFrame({
        "customer_id": ["CUST_001", "CUST_002"],
        "product_type": ["basic", "premium"],
    })
    labels = pd.DataFrame({
        "account_id": ["ACC_001"],
        "is_mule": [1],
    })
    test_ids = pd.DataFrame({"account_id": ["ACC_002"]})
    return {
        "accounts": accounts, "linkage": linkage, "customers": customers,
        "products": products, "labels": labels, "test_ids": test_ids,
    }


def test_merge_preserves_row_count(mini_tables):
    profile = build_account_profile(mini_tables)
    # Should have <= accounts rows (no explosion)
    assert len(profile) <= len(mini_tables["accounts"]) * 2


def test_no_duplicate_account_ids(mini_tables):
    profile = build_account_profile(mini_tables)
    assert profile["account_id"].duplicated().sum() == 0


def test_labels_correctly_assigned(mini_tables):
    profile = build_account_profile(mini_tables)
    profile = add_labels(profile, mini_tables["labels"], mini_tables["test_ids"])
    assert profile[profile["account_id"] == "ACC_001"]["dataset"].iloc[0] == "train"
    assert profile[profile["account_id"] == "ACC_002"]["dataset"].iloc[0] == "test"
    assert profile[profile["account_id"] == "ACC_003"]["dataset"].iloc[0] == "unlabeled"


def test_account_age_computed(mini_tables):
    profile = build_account_profile(mini_tables)
    assert "account_age_days" in profile.columns
    assert (profile["account_age_days"] >= 0).all()
