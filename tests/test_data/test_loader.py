import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import load_transactions, load_static_tables

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dir(tmp_path):
    import shutil
    csv = FIXTURES_DIR / "sample_transactions.csv"
    for i in range(6):
        shutil.copy(csv, tmp_path / f"transactions_part_{i}.csv")
    return tmp_path


@pytest.mark.unit
def test_load_transactions_returns_dataframe(fixture_dir):
    df = load_transactions(data_dir=fixture_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.unit
def test_load_transactions_sorted(fixture_dir):
    df = load_transactions(data_dir=fixture_dir)
    for i in range(len(df) - 1):
        assert df.iloc[i]["account_id"] <= df.iloc[i + 1]["account_id"]


@pytest.mark.unit
def test_load_transactions_correct_dtypes(fixture_dir):
    df = load_transactions(data_dir=fixture_dir)
    assert df["transaction_amount"].dtype.name == "float32"
    assert df["is_credit"].dtype.name == "int8"


@pytest.mark.unit
def test_load_static_tables_returns_dict(tmp_path):
    tables = load_static_tables(data_dir=tmp_path)
    assert isinstance(tables, dict)
    assert set(tables.keys()) == {"customers", "accounts", "linkage", "products", "labels", "test_ids"}


@pytest.mark.unit
def test_missing_file_handled_gracefully(tmp_path):
    df = load_transactions(data_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
