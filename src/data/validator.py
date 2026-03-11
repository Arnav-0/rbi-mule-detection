import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_transactions(self, txn: pd.DataFrame) -> dict:
        required = ["account_id", "transaction_date", "transaction_amount", "is_credit"]
        report = {}
        report["missing_columns"] = [c for c in required if c not in txn.columns]
        report["total_rows"] = len(txn)
        report["null_pct"] = (txn.isnull().sum() / max(len(txn), 1) * 100).round(2).to_dict()
        report["duplicate_txn_ids"] = int(txn.get("transaction_id", pd.Series()).duplicated().sum())
        if "transaction_date" in txn.columns:
            report["date_range"] = {
                "min": str(txn["transaction_date"].min()),
                "max": str(txn["transaction_date"].max()),
            }
        report["negative_amounts"] = int((txn.get("transaction_amount", pd.Series(dtype="float64")) < 0).sum())
        report["zero_amounts"] = int((txn.get("transaction_amount", pd.Series(dtype="float64")) == 0).sum())
        report["unique_accounts"] = int(txn["account_id"].nunique()) if "account_id" in txn.columns else 0
        return report

    def validate_accounts(self, accounts: pd.DataFrame) -> dict:
        if accounts is None:
            return {"error": "accounts table is None"}
        required = ["account_id"]
        report = {}
        report["missing_columns"] = [c for c in required if c not in accounts.columns]
        report["total_rows"] = len(accounts)
        report["null_pct"] = (accounts.isnull().sum() / max(len(accounts), 1) * 100).round(2).to_dict()
        report["duplicate_account_ids"] = int(accounts["account_id"].duplicated().sum()) if "account_id" in accounts.columns else -1
        if "account_type" in accounts.columns:
            report["account_type_dist"] = accounts["account_type"].value_counts().to_dict()
        return report

    def validate_labels(self, labels: pd.DataFrame) -> dict:
        if labels is None:
            return {"error": "labels table is None"}
        report = {}
        report["total_count"] = len(labels)
        if "is_mule" in labels.columns:
            report["mule_count"] = int(labels["is_mule"].sum())
            report["legitimate_count"] = int((labels["is_mule"] == 0).sum())
            report["mule_pct"] = round(labels["is_mule"].mean() * 100, 2)
        return report

    def validate_consistency(self, static_tables: dict) -> dict:
        report = {}
        accounts = static_tables.get("accounts")
        linkage = static_tables.get("linkage")
        labels = static_tables.get("labels")

        if accounts is not None and linkage is not None:
            acc_ids = set(accounts["account_id"]) if "account_id" in accounts.columns else set()
            link_ids = set(linkage["account_id"]) if "account_id" in linkage.columns else set()
            report["accounts_without_linkage"] = len(acc_ids - link_ids)
            report["linkage_without_accounts"] = len(link_ids - acc_ids)

        if accounts is not None and labels is not None:
            acc_ids = set(accounts["account_id"]) if "account_id" in accounts.columns else set()
            label_ids = set(labels["account_id"]) if "account_id" in labels.columns else set()
            report["labels_without_accounts"] = len(label_ids - acc_ids)

        return report

    def run_full_validation(self, txn=None, static_tables=None) -> dict:
        combined = {}
        if txn is not None:
            combined["transactions"] = self.validate_transactions(txn)
        if static_tables:
            if static_tables.get("accounts") is not None:
                combined["accounts"] = self.validate_accounts(static_tables["accounts"])
            if static_tables.get("labels") is not None:
                combined["labels"] = self.validate_labels(static_tables["labels"])
            combined["consistency"] = self.validate_consistency(static_tables)

        self.logger.info("Validation report:\n%s", json.dumps(combined, indent=2, default=str))
        return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.loader import load_all

    txn, static = load_all()
    validator = DataValidator()
    validator.run_full_validation(txn, static)
