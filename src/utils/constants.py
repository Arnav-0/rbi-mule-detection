TRANSACTION_DTYPES = {
    "account_id": "str",
    "transaction_id": "str",
    "transaction_timestamp": "str",
    "amount": "float32",
    "txn_type": "category",
    "channel": "category",
    "counterparty_id": "str",
    "mcc_code": "str",
}

# Column rename mapping: raw CSV -> internal pipeline names
TRANSACTION_RENAME = {
    "transaction_timestamp": "transaction_date",
    "amount": "transaction_amount",
    "txn_type": "transaction_type",
}

FEATURE_GROUPS = [
    "velocity",
    "amount_patterns",
    "temporal",
    "passthrough",
    "graph_network",
    "profile_mismatch",
    "kyc_behavioral",
    "interactions",
]

MODEL_NAMES = [
    "logistic",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "neural_net",
]

ROUND_AMOUNTS = [1000, 5000, 10000, 50000]

STRUCTURING_LOWER = 45000
STRUCTURING_UPPER = 49999
