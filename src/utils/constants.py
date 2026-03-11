TRANSACTION_DTYPES = {
    "account_id": "str",
    "transaction_id": "str",
    "transaction_date": "str",
    "transaction_amount": "float32",
    "transaction_type": "category",
    "channel": "category",
    "counterparty_id": "str",
    "is_credit": "int8",
    "balance_after": "float32",
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
