import pandas as pd
from pathlib import Path

from src.data.loader import load_transactions, load_static_tables

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_transactions_returns_dataframe():
    df = load_transactions(data_dir=FIXTURES_DIR)
    # fixture has 5 rows; may be empty if file naming doesn't match part_0..5
    # The loader looks for transactions_part_0.csv through part_5.csv
    # Our fixture is named differently, so we test empty graceful case
    assert isinstance(df, pd.DataFrame)


def test_load_transactions_from_fixture(tmp_path):
    """Copy fixture as transactions_part_0.csv and load it."""
    src = FIXTURES_DIR / "sample_transactions.csv"
    dst = tmp_path / "transactions_part_0.csv"
    dst.write_text(src.read_text())

    df = load_transactions(data_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_load_transactions_sorted(tmp_path):
    src = FIXTURES_DIR / "sample_transactions.csv"
    dst = tmp_path / "transactions_part_0.csv"
    dst.write_text(src.read_text())

    df = load_transactions(data_dir=tmp_path)
    if len(df) > 1:
        for acc in df["account_id"].unique():
            acc_df = df[df["account_id"] == acc]
            dates = acc_df["transaction_date"].values
            assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))


def test_load_transactions_correct_dtypes(tmp_path):
    src = FIXTURES_DIR / "sample_transactions.csv"
    dst = tmp_path / "transactions_part_0.csv"
    dst.write_text(src.read_text())

    df = load_transactions(data_dir=tmp_path)
    if len(df) > 0:
        assert df["transaction_amount"].dtype == "float32"
        assert str(df["transaction_type"].dtype) == "category"


def test_load_static_tables_returns_dict():
    result = load_static_tables(data_dir=FIXTURES_DIR)
    expected_keys = {"customers", "accounts", "linkage", "products", "labels", "test_ids"}
    assert set(result.keys()) == expected_keys


def test_missing_file_handled_gracefully(tmp_path):
    """Loading from empty directory should not crash."""
    result = load_transactions(data_dir=tmp_path)
    assert isinstance(result, pd.DataFrame)
    static = load_static_tables(data_dir=tmp_path)
    assert all(v is None for v in static.values())
