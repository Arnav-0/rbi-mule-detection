# RBI Mule Account Detection System — Complete Production-Grade Project Plan

**Project:** Money Laundering / Mule Account Detection System  
**Dataset:** RBI Innovation Hub Hackathon — 7.4M transactions, 40K accounts, July 2020–June 2025  
**Target:** Production-grade ML system for binary mule classification with explainability, fairness, and real-time serving  
**Build Time:** 3–4 weeks at 3–4 hours/day  

---

## 1. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        RBI MULE ACCOUNT DETECTION SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐ │
│  │  RAW DATA     │   │  DATA PIPELINE   │   │  FEATURE STORE   │   │  MODEL TRAINING  │ │
│  │              │   │                  │   │                  │   │                  │ │
│  │ customers    │──>│ Load (chunked)   │──>│ 8 Feature Groups │──>│ 6 Models:        │ │
│  │ accounts     │   │ Validate         │   │ 55+ Features     │   │ - LogReg         │ │
│  │ linkage      │   │ Merge (star)     │   │ Point-in-time    │   │ - RandomForest   │ │
│  │ products     │   │ Type optimize    │   │ correct          │   │ - XGBoost        │ │
│  │ transactions │   │ Time-sort        │   │                  │   │ - LightGBM       │ │
│  │ (6 parts)    │   │                  │   │ Graph features   │   │ - CatBoost       │ │
│  │ labels       │   │ Split:           │   │ (NetworkX)       │   │ - Neural Net     │ │
│  │ test_ids     │   │ Train/Val/Test   │   │                  │   │ (PyTorch)        │ │
│  └──────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘ │
│                                                                          │              │
│         ┌────────────────────────────────────────────────────────────────┘              │
│         v                                                                               │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐                    │
│  │  EVALUATION       │   │  EXPLAINABILITY   │   │  FAIRNESS AUDIT  │                    │
│  │                  │   │                  │   │                  │                    │
│  │ AUC-ROC/PR      │   │ SHAP (global +   │   │ Fairlearn        │                    │
│  │ F1/Precision/   │   │   local)         │   │ Demographic      │                    │
│  │   Recall        │   │ PDP plots        │   │   Parity         │                    │
│  │ Calibration     │   │ Feature imp.     │   │ Equalized Odds   │                    │
│  │ Stat tests      │   │ Natural lang.    │   │ Bias mitigation  │                    │
│  │ Temporal IoU    │   │   explanations   │   │ Model Card       │                    │
│  │ Learning curves │   │                  │   │                  │                    │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘                    │
│           └──────────────────────┼──────────────────────┘                               │
│                                  v                                                      │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐                    │
│  │  MODEL REGISTRY  │   │  FASTAPI SERVER  │   │  DASHBOARD       │                    │
│  │                  │   │                  │   │  (Streamlit)     │                    │
│  │ Best model       │──>│ /predict         │<──│                  │                    │
│  │ SHAP explainer   │   │ /predict/batch   │   │ Overview         │                    │
│  │ Feature pipeline │   │ /account/{id}    │   │ Feature Explorer │                    │
│  │ Metadata         │   │ /model/info      │   │ Model Comparison │                    │
│  │                  │   │ /fairness/report │   │ Explainability   │                    │
│  │ SQLite DB        │   │ /dashboard/stats │   │ Network Graph    │                    │
│  │ W&B artifacts    │   │                  │   │ Fairness Audit   │                    │
│  └──────────────────┘   └──────────────────┘   │ Account Inspect  │                    │
│                                                 │ API Demo         │                    │
│                                                 └──────────────────┘                    │
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │  INFRASTRUCTURE: Docker Compose │ GitHub Actions CI/CD │ W&B Tracking │ pytest   │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

**Data Flow Summary:**
```
Raw CSVs → Chunked Load → Validate → Merge (star schema on account_id)
  → Time-sorted transactions → Point-in-time feature computation
  → 55+ features per account → Train/Val split (time-aware stratified)
  → 6 models trained (Optuna HPO) → Best model selected
  → SHAP explanations computed → Fairness audited
  → Model + pipeline serialized → FastAPI serves predictions
  → Streamlit dashboard consumes API → Hackathon CSV exported
```

---

## 2. Complete Directory Structure

```
rbi-mule-detection/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Lint + test on every push/PR
│       └── model_validation.yml      # Model performance regression check
│
├── data/
│   ├── raw/                          # Raw CSVs (gitignored)
│   │   ├── customers.csv
│   │   ├── accounts.csv
│   │   ├── customer_account_linkage.csv
│   │   ├── product_details.csv
│   │   ├── transactions_part_0.csv
│   │   ├── transactions_part_1.csv
│   │   ├── transactions_part_2.csv
│   │   ├── transactions_part_3.csv
│   │   ├── transactions_part_4.csv
│   │   ├── transactions_part_5.csv
│   │   ├── train_labels.csv
│   │   └── test_accounts.csv
│   ├── processed/                    # Intermediate outputs (gitignored)
│   │   ├── merged_accounts.parquet
│   │   ├── transactions_clean.parquet
│   │   └── features_matrix.parquet
│   └── README.md                     # Download instructions
│
├── notebooks/
│   ├── 01_eda_data_quality.ipynb     # Data exploration & quality audit
│   ├── 02_feature_engineering.ipynb  # Feature prototyping & visualization
│   ├── 03_graph_analysis.ipynb       # Network topology exploration
│   ├── 04_modeling_baseline.ipynb    # Quick baseline models
│   ├── 05_model_comparison.ipynb     # Full 6-model comparison
│   ├── 06_explainability.ipynb       # SHAP & PDP analysis
│   ├── 07_fairness_audit.ipynb       # Fairlearn analysis
│   ├── 08_temporal_iou.ipynb         # Suspicious window detection
│   └── 09_submission.ipynb           # Final hackathon submission
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Chunked CSV loading, memory optimization
│   │   ├── validator.py              # Data quality checks, schema validation
│   │   ├── merger.py                 # Star-schema merge across all CSVs
│   │   ├── splitter.py               # Time-aware stratified train/val/test split
│   │   └── preprocessor.py           # Type casting, missing value handling
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseFeatureGenerator abstract class
│   │   ├── velocity.py               # Group 1: Transaction velocity features
│   │   ├── amount_patterns.py        # Group 2: Amount pattern features
│   │   ├── temporal.py               # Group 3: Temporal features
│   │   ├── passthrough.py            # Group 4: Pass-through features
│   │   ├── graph_network.py          # Group 5: Graph/network features (NetworkX)
│   │   ├── profile_mismatch.py       # Group 6: Profile mismatch features
│   │   ├── kyc_behavioral.py         # Group 7: KYC & behavioral features
│   │   ├── interactions.py           # Group 8: Cross/interaction features
│   │   ├── pipeline.py               # Orchestrates all feature generators
│   │   └── registry.py               # Feature metadata & documentation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseModel abstract class
│   │   ├── logistic.py               # Logistic Regression wrapper
│   │   ├── random_forest.py          # Random Forest wrapper
│   │   ├── xgboost_model.py          # XGBoost + Optuna
│   │   ├── lightgbm_model.py         # LightGBM + Optuna
│   │   ├── catboost_model.py         # CatBoost + Optuna
│   │   ├── neural_net.py             # PyTorch neural network
│   │   ├── trainer.py                # Unified training loop with W&B
│   │   ├── evaluator.py              # All metrics, comparison, stat tests
│   │   ├── selector.py               # Model selection & ensemble logic
│   │   └── calibrator.py             # Probability calibration (Platt/isotonic)
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py         # SHAP computation (Tree + Kernel)
│   │   ├── pdp.py                    # Partial dependence plots
│   │   ├── fairness.py               # Fairlearn audit
│   │   ├── model_card.py             # Model Card generator
│   │   └── natural_language.py       # "Flagged because..." explanations
│   │
│   ├── temporal/
│   │   ├── __init__.py
│   │   └── window_detector.py        # Suspicious time window detection
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py            # /predict, /predict/batch
│   │   │   ├── account.py            # /account/{id}
│   │   │   ├── model.py              # /model/info, /model/features
│   │   │   ├── dashboard.py          # /dashboard/stats
│   │   │   ├── fairness.py           # /fairness/report
│   │   │   └── benchmark.py          # /benchmark/results
│   │   ├── schemas.py                # Pydantic request/response models
│   │   ├── dependencies.py           # Model loading, DB connections
│   │   └── middleware.py             # Logging, error handling
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py                 # SQLAlchemy/SQLite schema
│   │   ├── crud.py                   # Database operations
│   │   └── init_db.py                # Schema creation & seeding
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # Configuration management
│       ├── logging_config.py         # Structured logging
│       ├── metrics.py                # Custom metric implementations
│       └── constants.py              # Threshold values, feature names
│
├── frontend/
│   ├── app.py                        # Streamlit main entry
│   ├── pages/
│   │   ├── 1_Overview.py             # Dataset stats & EDA
│   │   ├── 2_Feature_Explorer.py     # Feature analysis
│   │   ├── 3_Model_Comparison.py     # 6-model comparison
│   │   ├── 4_Explainability.py       # SHAP & PDP
│   │   ├── 5_Network_Graph.py        # Transaction network viz
│   │   ├── 6_Fairness_Audit.py       # Fairlearn results
│   │   ├── 7_Account_Inspector.py    # Single account deep dive
│   │   └── 8_API_Demo.py             # Live prediction demo
│   ├── components/
│   │   ├── charts.py                 # Reusable chart components
│   │   ├── tables.py                 # Formatted data tables
│   │   └── sidebar.py                # Navigation & filters
│   └── assets/
│       └── style.css                 # Custom styling
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures (sample data, mock models)
│   ├── test_data/
│   │   ├── test_loader.py
│   │   ├── test_validator.py
│   │   └── test_merger.py
│   ├── test_features/
│   │   ├── test_velocity.py
│   │   ├── test_amount_patterns.py
│   │   ├── test_temporal.py
│   │   ├── test_passthrough.py
│   │   ├── test_graph_network.py
│   │   └── test_pipeline.py
│   ├── test_models/
│   │   ├── test_trainer.py
│   │   └── test_evaluator.py
│   ├── test_api/
│   │   └── test_routes.py
│   └── test_temporal/
│       └── test_window_detector.py
│
├── docker/
│   ├── Dockerfile.api                # FastAPI server image
│   ├── Dockerfile.dashboard          # Streamlit dashboard image
│   └── Dockerfile.training           # Training environment with GPU
│
├── docs/
│   ├── architecture.md               # System architecture documentation
│   ├── feature_documentation.md      # Complete feature catalog
│   ├── model_card.md                 # Google Model Card framework
│   ├── api_reference.md              # API endpoint documentation
│   └── interview_prep.md             # Interview Q&A for this project
│
├── outputs/                          # Gitignored
│   ├── models/                       # Serialized models (.joblib, .pt)
│   ├── predictions/                  # submission.csv, test predictions
│   ├── reports/                      # Evaluation reports, fairness reports
│   ├── plots/                        # Saved visualizations
│   └── shap_values/                  # Precomputed SHAP arrays
│
├── .env.example                      # Environment variable template
├── .gitignore
├── docker-compose.yml
├── Makefile
├── pyproject.toml                    # Dependencies & project metadata
├── requirements.txt                  # Pinned dependencies
├── setup.py
└── README.md
```

---

## 3. Data Pipeline Specification

### 3.1 Efficient Loading of 7.4M Transactions

```python
# src/data/loader.py — Core loading strategy

import pandas as pd
import numpy as np
from pathlib import Path

# STRATEGY: Chunked reading + memory optimization via dtype downcasting

TRANSACTION_DTYPES = {
    'account_id': 'str',
    'transaction_id': 'str',
    'transaction_date': 'str',          # parse separately
    'transaction_amount': 'float32',     # float32 saves 50% vs float64
    'transaction_type': 'category',      # category saves ~90% for low-cardinality
    'channel': 'category',
    'counterparty_id': 'str',
    'branch_code': 'category',
    'is_credit': 'int8',                # int8 for binary: 0/1
    'balance_after': 'float32',
}

def load_transactions(data_dir: Path, nrows: int = None) -> pd.DataFrame:
    """Load all 6 transaction parts with memory-efficient dtypes."""
    parts = []
    for i in range(6):
        path = data_dir / f"transactions_part_{i}.csv"
        chunk = pd.read_csv(
            path,
            dtype=TRANSACTION_DTYPES,
            parse_dates=['transaction_date'],
            nrows=nrows,  # For development/debugging
        )
        parts.append(chunk)
    
    txn = pd.concat(parts, ignore_index=True)
    
    # Sort by account + time (critical for temporal features)
    txn.sort_values(['account_id', 'transaction_date'], inplace=True)
    txn.reset_index(drop=True, inplace=True)
    
    return txn

# Memory estimate: 
# ~7.4M rows × ~10 cols × (optimized dtypes) ≈ 2-3 GB in RAM
# Without optimization: ~6-8 GB → with optimization: ~2-3 GB
```

**Polars Fallback** (if pandas is too slow):
```python
import polars as pl

def load_transactions_polars(data_dir: Path) -> pl.DataFrame:
    parts = [pl.read_csv(data_dir / f"transactions_part_{i}.csv") for i in range(6)]
    return pl.concat(parts).sort(['account_id', 'transaction_date'])
```

### 3.2 Merge Strategy (Star Schema)

```
                    ┌──────────────┐
                    │  customers   │
                    │ customer_id  │
                    └──────┬───────┘
                           │ (1:N via linkage)
┌──────────────┐   ┌──────┴───────┐   ┌──────────────┐
│  products    │───│   linkage    │───│  accounts    │
│ customer_id  │   │ customer_id  │   │ account_id   │
└──────────────┘   │ account_id   │   └──────┬───────┘
                   └──────────────┘          │ (1:N)
                                     ┌──────┴───────┐
                                     │ transactions │
                                     │ account_id   │
                                     └──────────────┘
```

```python
# src/data/merger.py — Merge all static tables into account-level view

def build_account_profile(data_dir: Path) -> pd.DataFrame:
    """Merge customers + accounts + linkage + products into one account-level table."""
    
    accounts = pd.read_csv(data_dir / 'accounts.csv')
    linkage = pd.read_csv(data_dir / 'customer_account_linkage.csv')
    customers = pd.read_csv(data_dir / 'customers.csv')
    products = pd.read_csv(data_dir / 'product_details.csv')
    
    # Step 1: accounts ← linkage (add customer_id to each account)
    merged = accounts.merge(linkage, on='account_id', how='left')
    
    # Step 2: merged ← customers (add demographics)
    merged = merged.merge(customers, on='customer_id', how='left')
    
    # Step 3: merged ← products (add product holdings)
    merged = merged.merge(products, on='customer_id', how='left')
    
    return merged  # ~40,038 rows, one per account
```

### 3.3 Data Validation Checks

```python
# src/data/validator.py

class DataValidator:
    def validate_transactions(self, txn: pd.DataFrame) -> dict:
        report = {}
        
        # 1. Schema validation
        required_cols = ['account_id', 'transaction_date', 'transaction_amount',
                         'transaction_type', 'is_credit']
        report['missing_columns'] = [c for c in required_cols if c not in txn.columns]
        
        # 2. Null check per column
        report['null_pct'] = (txn.isnull().sum() / len(txn) * 100).to_dict()
        
        # 3. Duplicate transaction IDs
        report['duplicate_txn_ids'] = txn['transaction_id'].duplicated().sum()
        
        # 4. Date range validation
        report['date_range'] = {
            'min': txn['transaction_date'].min(),
            'max': txn['transaction_date'].max()
        }
        
        # 5. Amount sanity (no negative amounts in absolute values)
        report['negative_amounts'] = (txn['transaction_amount'] < 0).sum()
        
        # 6. Orphan accounts (in transactions but not in accounts table)
        # Run after merge
        
        # 7. Label coverage
        # Check that all train_labels account_ids exist in accounts
        
        return report
```

### 3.4 Train/Validation/Test Split Strategy

**Approach: Stratified split with temporal awareness**

```python
# src/data/splitter.py

from sklearn.model_selection import StratifiedKFold

def split_data(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    """
    Split strategy:
    1. Hackathon test set = test_accounts.csv (16,015 accounts) — NO labels, for submission only
    2. Train labels (24,023 accounts) → split into Train (80%) + Validation (20%)
    3. Stratified on mule label to preserve class ratio
    4. Temporal validation: ensure features computed before split cutoff
    """
    
    # Merge features with labels
    labeled = features_df.merge(labels_df, on='account_id', how='inner')
    
    # Stratified split: 80/20
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # For single split:
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        labeled,
        test_size=0.2,
        stratify=labeled['is_mule'],
        random_state=42
    )
    
    return train_df, val_df

# POINT-IN-TIME RULE:
# All features MUST be computed using ONLY data available up to a cutoff date.
# For the full dataset (July 2020 – June 2025), features use the entire history.
# For temporal validation: use data up to Dec 2024 for features,
# Jan-June 2025 for testing model on "future" accounts.
```

**Data Leakage Prevention Checklist:**
1. Never use test account IDs during training
2. Features computed on transaction history — no look-ahead
3. Graph features: exclude edges involving test accounts from training graph
4. Rolling window features: use only past transactions relative to each transaction
5. Label encoding: fit on train, transform on val/test

---

## 4. Feature Engineering Specification (55+ Features)

> **This is the core differentiator.** Each feature includes: name, computation logic, rationale, and expected discriminative power (High/Medium/Low).

### POINT-IN-TIME CORRECTNESS RULES

```
RULE 1: For each account, features are computed using ONLY transactions 
        with transaction_date <= feature_cutoff_date
RULE 2: Rolling windows look BACKWARD only (e.g., "last 30 days" = 30 days before cutoff)
RULE 3: Graph features use the transaction graph built from training period only
RULE 4: Profile features (income, KYC) use static data — no temporal concern
RULE 5: Never use the label (is_mule) as a feature or to condition feature computation
```

### GROUP 1: Transaction Velocity Features (10 features)

| # | Feature Name | Computation | Rationale | Power |
|---|-------------|-------------|-----------|-------|
| 1 | `txn_count_1d` | Count of transactions in last 1 day | Burst detection — mules have sudden daily spikes | Medium |
| 2 | `txn_count_7d` | Count of transactions in last 7 days | Weekly activity level — mules show abnormal weekly volumes | High |
| 3 | `txn_count_30d` | Count in last 30 days | Monthly activity baseline | High |
| 4 | `txn_count_90d` | Count in last 90 days | Quarterly activity for seasonal normalization | Medium |
| 5 | `txn_amount_mean_30d` | Mean transaction amount in last 30 days | Average ticket size — mules often have higher averages | High |
| 6 | `txn_amount_max_30d` | Max single transaction in last 30 days | Largest single transfer — high values suspicious on dormant accounts | High |
| 7 | `txn_amount_std_30d` | Std dev of amounts in last 30 days | Variability — structuring creates artificially low std near thresholds | Medium |
| 8 | `txn_amount_sum_30d` | Total volume in last 30 days | Throughput — mules process high volumes | High |
| 9 | `velocity_acceleration` | `txn_count_7d / max(txn_count_30d / 4, 1)` | Rate of change — ratio of recent to baseline activity; >2 = acceleration | High |
| 10 | `frequency_change_ratio` | `txn_count_30d / max(txn_count_90d / 3, 1)` | Longer-term trend change — captures dormant-to-active transitions | High |

```python
# src/features/velocity.py — Pseudocode

def compute_velocity_features(txn: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    # Filter to transactions before cutoff
    txn_valid = txn[txn['transaction_date'] <= cutoff]
    
    features = {}
    for window_days in [1, 7, 30, 90]:
        window_start = cutoff - pd.Timedelta(days=window_days)
        window_txn = txn_valid[txn_valid['transaction_date'] > window_start]
        
        grouped = window_txn.groupby('account_id')
        features[f'txn_count_{window_days}d'] = grouped.size()
        
        if window_days == 30:
            features['txn_amount_mean_30d'] = grouped['transaction_amount'].mean()
            features['txn_amount_max_30d'] = grouped['transaction_amount'].max()
            features['txn_amount_std_30d'] = grouped['transaction_amount'].std()
            features['txn_amount_sum_30d'] = grouped['transaction_amount'].sum()
    
    result = pd.DataFrame(features).fillna(0)
    
    # Derived: acceleration & frequency change
    result['velocity_acceleration'] = (
        result['txn_count_7d'] / (result['txn_count_30d'] / 4).clip(lower=1)
    )
    result['frequency_change_ratio'] = (
        result['txn_count_30d'] / (result['txn_count_90d'] / 3).clip(lower=1)
    )
    
    return result
```

### GROUP 2: Amount Pattern Features (8 features)

| # | Feature Name | Computation | Rationale | Power |
|---|-------------|-------------|-----------|-------|
| 11 | `round_amount_ratio` | % of txns where amount % 1000 == 0 | Mules use round numbers (1K, 5K, 10K, 50K) disproportionately | High |
| 12 | `structuring_score` | % of txns with amount in [45000, 49999] | Structuring: deliberately staying below 50K reporting threshold | High |
| 13 | `structuring_score_broad` | % of txns in [40000, 49999] | Broader structuring band to catch variants | Medium |
| 14 | `amount_entropy` | Shannon entropy of amount distribution (binned into 20 bins) | Low entropy = repetitive amounts (mule signal); high entropy = normal variance | High |
| 15 | `amount_skewness` | Skewness of amount distribution per account | Positive skew = mostly small txns with a few large (pass-through pattern) | Medium |
| 16 | `amount_kurtosis` | Kurtosis of amount distribution | High kurtosis = heavy tails, extreme amounts | Medium |
| 17 | `pct_above_10k` | % of transactions > ₹10,000 | High-value transaction proportion — mules process higher values | Medium |
| 18 | `amount_concentration` | Gini coefficient of transaction amounts | Measures inequality — mules may have very unequal distribution | Medium |

```python
# src/features/amount_patterns.py — Key computations

from scipy.stats import entropy, skew, kurtosis

def round_amount_ratio(amounts: pd.Series) -> float:
    if len(amounts) == 0: return 0
    round_mask = (amounts % 1000 == 0) | (amounts % 5000 == 0) | (amounts % 10000 == 0)
    return round_mask.mean()

def structuring_score(amounts: pd.Series, lower=45000, upper=49999) -> float:
    if len(amounts) == 0: return 0
    return ((amounts >= lower) & (amounts <= upper)).mean()

def amount_entropy(amounts: pd.Series, bins=20) -> float:
    if len(amounts) < 2: return 0
    hist, _ = np.histogram(amounts, bins=bins)
    return entropy(hist + 1e-10)  # Add epsilon to avoid log(0)
```

### GROUP 3: Temporal Features (8 features)

| # | Feature Name | Computation | Rationale | Power |
|---|-------------|-------------|-----------|-------|
| 19 | `dormancy_days` | Days between first and most recent transaction (if gap > 90 days before recent burst) | Dormant→active = classic mule activation pattern (#1) | High |
| 20 | `max_gap_days` | Largest gap (in days) between consecutive transactions | Long gaps followed by bursts = suspicious | High |
| 21 | `burst_after_dormancy` | Binary: 1 if max_gap > 90 days AND txn_count_30d > 10 | Combines dormancy with burst — direct mule pattern #1 signal | High |
| 22 | `unusual_hour_ratio` | % of transactions between 11PM–5AM | Unusual hours suggest automated/scripted transfers | Medium |
| 23 | `weekend_ratio` | % of transactions on Saturday/Sunday | Weekend-heavy activity unusual for salary accounts | Medium |
| 24 | `night_weekend_combo` | unusual_hour_ratio × weekend_ratio | Compound signal: nighttime + weekend = higher suspicion | Medium |
| 25 | `monthly_txn_cv` | Coefficient of variation of monthly transaction counts | Stable legitimate accounts have low CV; mules spike and stop | High |
| 26 | `days_since_account_open_to_first_txn` | Gap between account creation and first transaction | New account fraud: quick activation for laundering | Medium |

```python
# src/features/temporal.py

def compute_dormancy(txn_dates: pd.Series) -> dict:
    if len(txn_dates) < 2:
        return {'dormancy_days': 0, 'max_gap_days': 0}
    
    sorted_dates = txn_dates.sort_values()
    gaps = sorted_dates.diff().dt.days.dropna()
    
    return {
        'dormancy_days': gaps.max() if gaps.max() > 90 else 0,
        'max_gap_days': gaps.max(),
    }
```

### GROUP 4: Pass-Through Features (7 features)

| # | Feature Name | Computation | Rationale | Power |
|---|-------------|-------------|-----------|-------|
| 27 | `credit_debit_time_delta_median` | Median time (hours) between a credit and the next debit for same account | Rapid pass-through: money in → money out quickly (pattern #3) | High |
| 28 | `credit_debit_time_delta_min` | Minimum time between credit→debit pair | Fastest turnaround — sub-hour = very suspicious | High |
| 29 | `matched_amount_ratio` | % of credits where a debit of similar amount (±5%) occurs within 24hrs | How often incoming = outgoing (pass-through matching) | High |
| 30 | `net_flow_ratio` | `total_credits / max(total_debits, 1)` | Ratio near 1.0 = pass-through; >>1 = accumulation | High |
| 31 | `rapid_turnover_score` | % of total volume that passes through within 48 hours | Overall pass-through intensity | High |
| 32 | `credit_debit_symmetry` | `1 - abs(n_credits - n_debits) / max(n_credits + n_debits, 1)` | Perfect symmetry (near 1.0) = pass-through behavior | Medium |
| 33 | `max_single_day_volume` | Highest total amount transacted in a single day | Burst days with massive volume = laundering windows | Medium |

```python
# src/features/passthrough.py

def compute_passthrough_features(txn: pd.DataFrame) -> pd.DataFrame:
    results = {}
    
    for account_id, group in txn.groupby('account_id'):
        credits = group[group['is_credit'] == 1].sort_values('transaction_date')
        debits = group[group['is_credit'] == 0].sort_values('transaction_date')
        
        # Credit→Debit time deltas
        time_deltas = []
        matched_count = 0
        
        for _, credit in credits.iterrows():
            # Find next debit after this credit
            subsequent_debits = debits[debits['transaction_date'] > credit['transaction_date']]
            if len(subsequent_debits) > 0:
                next_debit = subsequent_debits.iloc[0]
                delta_hours = (next_debit['transaction_date'] - credit['transaction_date']).total_seconds() / 3600
                time_deltas.append(delta_hours)
                
                # Check if amounts match (±5%)
                if abs(next_debit['transaction_amount'] - credit['transaction_amount']) / max(credit['transaction_amount'], 1) < 0.05:
                    if delta_hours < 24:
                        matched_count += 1
        
        total_credits = credits['transaction_amount'].sum()
        total_debits = debits['transaction_amount'].sum()
        
        results[account_id] = {
            'credit_debit_time_delta_median': np.median(time_deltas) if time_deltas else 999,
            'credit_debit_time_delta_min': np.min(time_deltas) if time_deltas else 999,
            'matched_amount_ratio': matched_count / max(len(credits), 1),
            'net_flow_ratio': total_credits / max(total_debits, 1),
            'credit_debit_symmetry': 1 - abs(len(credits) - len(debits)) / max(len(credits) + len(debits), 1),
        }
    
    return pd.DataFrame.from_dict(results, orient='index')
```

### GROUP 5: Graph/Network Features (10 features)

| # | Feature Name | Computation | Rationale | Power |
|---|-------------|-------------|-----------|-------|
| 34 | `in_degree` | Number of unique accounts that sent money TO this account | Fan-in: many senders → potential collection point (pattern #4) | High |
| 35 | `out_degree` | Number of unique accounts that received money FROM this account | Fan-out: many receivers → potential distribution point | High |
| 36 | `fan_in_ratio` | `in_degree / max(out_degree, 1)` | High ratio = collection hub; low ratio = distribution hub | High |
| 37 | `fan_out_ratio` | `out_degree / max(in_degree, 1)` | Inverse of fan_in — identifies distribution nodes | High |
| 38 | `betweenness_centrality` | NetworkX betweenness centrality score | Bridge accounts connecting otherwise separate clusters (intermediary) | High |
| 39 | `pagerank` | PageRank score on transaction graph | Importance in money flow network — high PageRank = money flows through here | High |
| 40 | `community_id` | Louvain community detection cluster ID | Identifies collusion clusters (pattern #12) | High |
| 41 | `community_mule_density` | % of known mules in same community (train only) | If community has many mules, remaining members are suspicious | High |
| 42 | `clustering_coefficient` | Local clustering coefficient | High clustering = tight-knit groups (collusion); low = bridge/intermediary | Medium |
| 43 | `total_counterparties` | Total unique counterparties (in + out) | Mules interact with many accounts in short periods | Medium |

```python
# src/features/graph_network.py

import networkx as nx
from community import community_louvain

def build_transaction_graph(txn: pd.DataFrame) -> nx.DiGraph:
    """Build directed weighted graph from transactions."""
    G = nx.DiGraph()
    
    # Aggregate edges: sum of amounts and count of transactions
    edges = txn.groupby(['account_id', 'counterparty_id']).agg(
        total_amount=('transaction_amount', 'sum'),
        txn_count=('transaction_amount', 'count')
    ).reset_index()
    
    for _, row in edges.iterrows():
        G.add_edge(
            row['account_id'], row['counterparty_id'],
            weight=row['total_amount'],
            count=row['txn_count']
        )
    
    return G

def compute_graph_features(G: nx.DiGraph, target_accounts: list) -> pd.DataFrame:
    """Compute network topology features for target accounts."""
    
    # Convert to undirected for some metrics
    G_undirected = G.to_undirected()
    
    # Centrality metrics (computed on full graph, extracted for targets)
    pagerank = nx.pagerank(G, weight='weight', max_iter=100)
    betweenness = nx.betweenness_centrality(G, k=min(1000, len(G)))  # Approximate for speed
    clustering = nx.clustering(G_undirected)
    
    # Community detection on undirected graph
    communities = community_louvain.best_partition(G_undirected, weight='weight')
    
    features = {}
    for acc in target_accounts:
        if acc in G:
            features[acc] = {
                'in_degree': G.in_degree(acc),
                'out_degree': G.out_degree(acc),
                'fan_in_ratio': G.in_degree(acc) / max(G.out_degree(acc), 1),
                'fan_out_ratio': G.out_degree(acc) / max(G.in_degree(acc), 1),
                'betweenness_centrality': betweenness.get(acc, 0),
                'pagerank': pagerank.get(acc, 0),
                'community_id': communities.get(acc, -1),
                'clustering_coefficient': clustering.get(acc, 0),
                'total_counterparties': G.in_degree(acc) + G.out_degree(acc),
            }
        else:
            # Account not in graph (no transactions with counterparties)
            features[acc] = {k: 0 for k in [
                'in_degree', 'out_degree', 'fan_in_ratio', 'fan_out_ratio',
                'betweenness_centrality', 'pagerank', 'community_id',
                'clustering_coefficient', 'total_counterparties'
            ]}
    
    return pd.DataFrame.from_dict(features, orient='index')

# COMMUNITY MULE DENSITY (train-only to avoid leakage):
def compute_community_mule_density(graph_features: pd.DataFrame, labels: pd.DataFrame):
    merged = graph_features.merge(labels, on='account_id', how='left')
    community_density = merged.groupby('community_id')['is_mule'].mean()
    graph_features['community_mule_density'] = (
        graph_features['community_id'].map(community_density).fillna(0)
    )
    return graph_features
```

### GROUP 6: Profile Mismatch Features (5 features)

| # | Feature Name | Computation | Rationale | Power |
|---|-------------|-------------|-----------|-------|
| 44 | `txn_volume_vs_income` | `txn_amount_sum_30d / max(declared_income, 1)` | Transactions disproportionate to declared income = suspicious | High |
| 45 | `account_age_vs_activity` | `txn_count_30d / max(account_age_days, 1)` | New accounts with high activity = pattern #6 | High |
| 46 | `avg_txn_vs_balance` | `txn_amount_mean_30d / max(current_balance, 1)` | Average transaction much larger than balance = pass-through | Medium |
| 47 | `product_txn_mismatch` | Binary: 1 if savings account but avg txn > ₹50K | Product type vs behavior inconsistency | Medium |
| 48 | `balance_volatility` | Std dev of daily ending balance / mean balance | High volatility = rapid inflow/outflow cycles | High |

### GROUP 7: KYC & Behavioral Features (4 features)

| # | Feature Name | Computation | Rationale | Power |
|---|-------------|-------------|-----------|-------|
| 49 | `mobile_change_flag` | 1 if mobile number changed in dataset period | Pattern #8: mobile change precedes suspicious activity | Medium |
| 50 | `activity_change_post_mobile` | Ratio of txn count in 30 days after vs 30 days before mobile change | Spike after mobile change = mule activation signal | High |
| 51 | `kyc_completeness` | Count of non-null KYC fields / total KYC fields | Incomplete KYC on high-activity account = suspicious | Medium |
| 52 | `linked_account_count` | Number of accounts linked to same customer_id | Multiple accounts for one customer = potential mule setup | Medium |

### GROUP 8: Interaction/Cross Features (5 features)

| # | Feature Name | Computation | Rationale | Power |
|---|-------------|-------------|-----------|-------|
| 53 | `dormancy_x_burst` | `dormancy_days × txn_count_7d` | Combines dormancy with recent burst magnitude | High |
| 54 | `round_x_structuring` | `round_amount_ratio × structuring_score` | Double signal: round amounts AND near threshold | High |
| 55 | `fanin_x_passthrough_speed` | `fan_in_ratio × (1 / max(credit_debit_time_delta_median, 0.1))` | Collection hub with fast turnover = strong mule signal | High |
| 56 | `new_account_x_high_value` | `(1 if account_age < 180 else 0) × pct_above_10k` | New account with high-value transactions = pattern #6 | High |
| 57 | `velocity_x_centrality` | `velocity_acceleration × betweenness_centrality` | Accelerating activity on a network bridge account | High |

### Feature Computation Order & Dependencies

```
STAGE 1 (No dependencies): 
  → Load transactions, accounts, profiles

STAGE 2 (Depends on Stage 1):
  → Group 1: Velocity features (needs sorted transactions)
  → Group 2: Amount patterns (needs transaction amounts)
  → Group 3: Temporal features (needs sorted transaction dates)
  → Group 6: Profile mismatch (needs account profiles + some velocity features)
  → Group 7: KYC/behavioral (needs customer data)

STAGE 3 (Depends on Stage 1):
  → Group 4: Pass-through features (needs credit/debit pairs sorted by time)
  → Group 5: Graph features (needs transaction graph built from all account pairs)

STAGE 4 (Depends on Stages 2 + 3):
  → Group 8: Interaction features (combines features from all other groups)

STAGE 5 (Depends on Stage 4):
  → Feature matrix assembly
  → Missing value imputation (fill with 0 for count features, median for continuous)
  → Feature scaling (StandardScaler for NN, leave raw for tree models)
```

**TOTAL: 57 features across 8 groups**

---

## 5. Model Training Specification

### 5.1 Model Configurations & Optuna Search Spaces

**Model 1: Logistic Regression (Baseline)**
```python
# No Optuna needed — simple grid
from sklearn.linear_model import LogisticRegression

param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': 'saga',
    'class_weight': 'balanced',
    'max_iter': 1000,
}
# Requires scaled features (StandardScaler)
```

**Model 2: Random Forest**
```python
# Optuna search space
def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42,
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score
```

**Model 3: XGBoost**
```python
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'scale_pos_weight': n_negative / n_positive,  # Class imbalance
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',  # GPU acceleration
        'random_state': 42,
    }
    # Early stopping with validation set
    return score
```

**Model 4: LightGBM**
```python
def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'is_unbalance': True,
        'metric': 'auc',
        'verbose': -1,
        'device': 'gpu',
    }
    return score
```

**Model 5: CatBoost**
```python
def catboost_objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'auto_class_weights': 'Balanced',
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'random_seed': 42,
        'verbose': 0,
    }
    return score
```

**Model 6: PyTorch Neural Network**
```python
class MuleDetectorNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Optuna for NN
def nn_objective(trial):
    n_layers = trial.suggest_int('n_layers', 2, 5)
    hidden_dims = [trial.suggest_int(f'hidden_{i}', 32, 512) for i in range(n_layers)]
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Focal loss for class imbalance
    # Train for 50 epochs with early stopping (patience=10)
    return val_auc
```

### 5.2 Class Imbalance Strategy

```
EXPECTED CLASS DISTRIBUTION: ~5-15% mules (estimated from financial fraud benchmarks)

THREE STRATEGIES TO COMPARE:

1. CLASS WEIGHTS (Preferred for tree models):
   - XGBoost: scale_pos_weight = n_neg / n_pos
   - LightGBM: is_unbalance=True
   - CatBoost: auto_class_weights='Balanced'
   - LogReg/RF: class_weight='balanced'

2. SMOTE (Compare for tree models):
   - Apply ONLY to training fold (never validation/test)
   - Use SMOTE-ENN (combined over+undersampling)
   - from imblearn.over_sampling import SMOTENN

3. FOCAL LOSS (For Neural Network):
   - class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0):
           ...
       def forward(self, inputs, targets):
           BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
           pt = torch.exp(-BCE_loss)
           F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
           return F_loss.mean()

COMPARISON TABLE (to be filled during training):
| Strategy      | Model    | AUC-ROC | AUC-PR | F1    | Notes |
|---------------|----------|---------|--------|-------|-------|
| class_weight  | XGBoost  |         |        |       |       |
| SMOTE         | XGBoost  |         |        |       |       |
| class_weight  | LightGBM |         |        |       |       |
| SMOTE         | LightGBM |         |        |       |       |
| focal_loss    | NN       |         |        |       |       |
| class_weight  | NN       |         |        |       |       |
```

### 5.3 Cross-Validation Scheme

```python
# PRIMARY: Stratified 5-Fold (preserves mule ratio)
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ALTERNATIVE (if temporal leakage is a concern):
# TimeSeriesSplit — but since features are pre-computed at a cutoff,
# standard stratified CV is acceptable.

# For Optuna:
# Use 3-fold CV during HPO (speed), 5-fold for final evaluation
```

### 5.4 Feature Selection Pipeline

```python
# THREE-STAGE FEATURE SELECTION:

# Stage 1: Remove near-zero variance features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)

# Stage 2: Mutual Information ranking
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
# Keep features with MI > 0.01

# Stage 3: Boruta (wrapper method using Random Forest)
from boruta import BorutaPy
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
boruta.fit(X_train.values, y_train.values)
# Keep features confirmed by Boruta

# COMPARISON: Train best model (LightGBM) with:
# (a) All 57 features
# (b) Boruta-selected features
# (c) Top-20 by mutual information
# Report AUC for each → choose best subset
```

---

## 6. Evaluation Specification

### 6.1 Complete Metrics Suite

```python
# src/models/evaluator.py

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, roc_curve, brier_score_loss,
    classification_report
)
from sklearn.calibration import calibration_curve

class ModelEvaluator:
    def evaluate(self, y_true, y_prob, y_pred=None, threshold=0.5):
        if y_pred is None:
            y_pred = (y_prob >= threshold).astype(int)
        
        return {
            # Primary
            'auc_roc': roc_auc_score(y_true, y_prob),
            'auc_pr': average_precision_score(y_true, y_prob),
            
            # At optimal threshold
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            
            # Calibration
            'brier_score': brier_score_loss(y_true, y_prob),
            
            # Confusion matrix
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
    
    def find_optimal_threshold(self, y_true, y_prob, method='youden'):
        """Find optimal classification threshold."""
        if method == 'youden':
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            j_stat = tpr - fpr
            return thresholds[np.argmax(j_stat)]
        elif method == 'f1':
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
            f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            return thresholds[np.argmax(f1s)]
```

### 6.2 Statistical Comparison

```python
from scipy.stats import wilcoxon
import numpy as np

def compare_models(model_a_scores: list, model_b_scores: list, alpha=0.05):
    """
    Compare two models using paired Wilcoxon signed-rank test.
    model_a_scores, model_b_scores = AUC-ROC from each CV fold.
    """
    stat, p_value = wilcoxon(model_a_scores, model_b_scores)
    
    return {
        'mean_diff': np.mean(model_a_scores) - np.mean(model_b_scores),
        'p_value': p_value,
        'significant': p_value < alpha,
        'confidence_interval': _bootstrap_ci(model_a_scores, model_b_scores),
    }

def _bootstrap_ci(a, b, n_bootstrap=1000, alpha=0.05):
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, len(a), len(a))
        diffs.append(np.mean(np.array(a)[idx]) - np.mean(np.array(b)[idx]))
    return (np.percentile(diffs, alpha/2*100), np.percentile(diffs, (1-alpha/2)*100))
```

### 6.3 Suspicious Time Window Detection (Temporal IoU)

```python
# src/temporal/window_detector.py

def detect_suspicious_window(txn: pd.DataFrame, account_id: str,
                              window_size: int = 30,
                              anomaly_threshold: float = 2.0) -> dict:
    """
    Sliding window anomaly detection to find suspicious activity period.
    
    Algorithm:
    1. Compute daily transaction volume for the account
    2. Calculate rolling mean and std over 90-day baseline
    3. Score each day: z-score = (daily_volume - rolling_mean) / rolling_std
    4. Find contiguous period where z-score > threshold
    5. Extend window to include lead-up (7 days before) and tail (7 days after)
    """
    account_txn = txn[txn['account_id'] == account_id].copy()
    account_txn['date'] = account_txn['transaction_date'].dt.date
    
    # Daily aggregation
    daily = account_txn.groupby('date').agg(
        volume=('transaction_amount', 'sum'),
        count=('transaction_amount', 'count')
    ).reindex(
        pd.date_range(account_txn['date'].min(), account_txn['date'].max()),
        fill_value=0
    )
    
    # Rolling baseline (90-day window)
    rolling_mean = daily['volume'].rolling(90, min_periods=30).mean()
    rolling_std = daily['volume'].rolling(90, min_periods=30).std().clip(lower=1)
    
    # Z-scores
    z_scores = (daily['volume'] - rolling_mean) / rolling_std
    
    # Find anomalous period
    anomalous = z_scores > anomaly_threshold
    
    if anomalous.any():
        first_anomaly = anomalous.idxmax()
        last_anomaly = anomalous[::-1].idxmax()
        
        # Extend by 7 days each side
        suspicious_start = first_anomaly - pd.Timedelta(days=7)
        suspicious_end = last_anomaly + pd.Timedelta(days=7)
        
        return {
            'account_id': account_id,
            'suspicious_start': suspicious_start.isoformat(),
            'suspicious_end': suspicious_end.isoformat(),
        }
    
    return {
        'account_id': account_id,
        'suspicious_start': None,
        'suspicious_end': None,
    }

# TEMPORAL IoU METRIC:
def temporal_iou(pred_start, pred_end, true_start, true_end):
    """Intersection over Union for time windows."""
    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection = (intersection_end - intersection_start).total_seconds()
    union = (max(pred_end, true_end) - min(pred_start, true_start)).total_seconds()
    
    return intersection / union if union > 0 else 0.0
```

### 6.4 Model Comparison Output Table

```
FINAL MODEL COMPARISON TABLE (Template — filled during training):

| Model              | AUC-ROC      | AUC-PR       | F1           | Precision    | Recall       | Brier Score | Latency (ms) | Size (MB) |
|--------------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-----------|
| Logistic Regression| 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX        | X.X         | X.X       |
| Random Forest      | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX        | X.X         | X.X       |
| XGBoost            | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX        | X.X         | X.X       |
| LightGBM           | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX        | X.X         | X.X       |
| CatBoost           | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX        | X.X         | X.X       |
| Neural Network     | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX ± 0.XX | 0.XX        | X.X         | X.X       |

Statistical Significance (Wilcoxon p-values):
| Pair                  | p-value | Significant? |
|-----------------------|---------|-------------|
| LightGBM vs XGBoost   |         |             |
| LightGBM vs CatBoost  |         |             |
| Best Tree vs NN       |         |             |
```

---

## 7. Explainability Specification

### 7.1 SHAP Computation Plan

```python
# src/explainability/shap_explainer.py

import shap

class MuleExplainer:
    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        
        if model_type in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'neural_net':
            # KernelSHAP with background sample (slower)
            self.explainer = None  # Set with background data
        elif model_type == 'logistic':
            self.explainer = shap.LinearExplainer(model, X_train)
    
    def explain_global(self, X: pd.DataFrame) -> shap.Explanation:
        """Global feature importance — SHAP summary."""
        shap_values = self.explainer.shap_values(X)
        return shap_values
    
    def explain_local(self, X_single: pd.DataFrame) -> dict:
        """Single account explanation."""
        shap_values = self.explainer.shap_values(X_single)
        
        # Natural language explanation
        feature_contributions = sorted(
            zip(X_single.columns, shap_values[0]),
            key=lambda x: abs(x[1]), reverse=True
        )
        
        explanation = "This account was flagged because:\n"
        for feat, val in feature_contributions[:5]:
            direction = "increased" if val > 0 else "decreased"
            explanation += f"  - {feat} {direction} the risk score by {abs(val):.3f}\n"
        
        return {
            'shap_values': shap_values,
            'top_features': feature_contributions[:10],
            'natural_language': explanation,
        }

# PLOTS TO GENERATE:
# 1. shap.summary_plot(shap_values, X, plot_type="dot")        # Beeswarm
# 2. shap.summary_plot(shap_values, X, plot_type="bar")        # Global importance
# 3. shap.plots.waterfall(shap_values[account_idx])            # Per-account
# 4. shap.dependence_plot("feature_name", shap_values, X)      # Dependence
# 5. shap.plots.heatmap(shap_values[:100])                     # Top 100 accounts
```

### 7.2 Fairlearn Audit Plan

```python
# src/explainability/fairness.py

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    demographic_parity_ratio,
)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

SENSITIVE_FEATURES = [
    'age_group',           # <25, 25-45, 45-65, >65
    'geography_tier',      # Metro, Tier-1, Tier-2, Rural
    'account_type',        # Savings, Current, NRE
    'gender',              # If available in customer data
]

def run_fairness_audit(y_true, y_pred, sensitive_df: pd.DataFrame):
    """Run fairness audit across all sensitive features."""
    results = {}
    
    for feature in SENSITIVE_FEATURES:
        if feature not in sensitive_df.columns:
            continue
        
        mf = MetricFrame(
            metrics={
                'accuracy': accuracy_score,
                'recall': recall_score,
                'precision': precision_score,
                'f1': f1_score,
                'selection_rate': lambda y_t, y_p: y_p.mean(),
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_df[feature],
        )
        
        results[feature] = {
            'by_group': mf.by_group.to_dict(),
            'overall': mf.overall.to_dict(),
            'difference': mf.difference().to_dict(),
            'ratio': mf.ratio().to_dict(),
            'demographic_parity_diff': demographic_parity_difference(
                y_true, y_pred, sensitive_features=sensitive_df[feature]
            ),
            'equalized_odds_diff': equalized_odds_difference(
                y_true, y_pred, sensitive_features=sensitive_df[feature]
            ),
        }
    
    return results

# THRESHOLD: Flag if demographic_parity_ratio < 0.8 (80% rule)
# MITIGATION: If bias found, apply ThresholdOptimizer or ExponentiatedGradient
```

### 7.3 Model Card Template

```markdown
# Model Card: Mule Account Detection System

## Model Details
- **Model type:** [Best model, e.g., LightGBM]
- **Version:** 1.0
- **Training date:** [Date]
- **Training data:** 24,023 labeled accounts from RBI Innovation Hub dataset
- **Features:** 57 engineered features across 8 groups
- **Framework:** scikit-learn / LightGBM / PyTorch

## Intended Use
- **Primary use:** Detect bank accounts used as intermediaries in money laundering schemes
- **Users:** Banking compliance teams, financial regulators (RBI)
- **Out of scope:** Real-time fraud blocking (this is batch scoring)

## Performance Metrics
| Metric | Value |
|--------|-------|
| AUC-ROC | X.XXX |
| AUC-PR | X.XXX |
| F1 (optimal threshold) | X.XXX |
| Precision | X.XXX |
| Recall | X.XXX |
| Temporal IoU (mean) | X.XXX |

## Fairness Analysis
| Group | Selection Rate | Recall | Precision |
|-------|---------------|--------|-----------|
| Age < 25 | | | |
| Age 25-45 | | | |
| Age 45-65 | | | |
| Age > 65 | | | |
| Metro | | | |
| Rural | | | |

## Limitations
- Trained on Indian banking data (July 2020–June 2025) — may not generalize
- Graph features depend on counterparty coverage in dataset
- Class imbalance may affect precision at very low thresholds
- Temporal window detection uses heuristic z-score approach

## Ethical Considerations
- False positives can freeze legitimate accounts — threshold should favor precision
- Age/geography bias must be monitored in production
- Model should augment human review, not replace it
```

---

## 8. API Specification

### FastAPI Endpoints

**Base URL:** `http://localhost:8000/api/v1`

#### `POST /predict`
Single account prediction with explanation.

```json
// Request
{
    "account_id": "ACC_12345",
    "features": {  // Optional: if not provided, computed from DB
        "txn_count_7d": 45,
        "structuring_score": 0.12,
        // ... all 57 features
    }
}

// Response (200 OK)
{
    "account_id": "ACC_12345",
    "probability": 0.87,
    "prediction": "mule",
    "threshold_used": 0.52,
    "suspicious_window": {
        "start": "2024-11-15T00:00:00",
        "end": "2025-01-20T00:00:00"
    },
    "explanation": {
        "top_features": [
            {"name": "matched_amount_ratio", "value": 0.82, "shap_contribution": 0.34},
            {"name": "velocity_acceleration", "value": 5.2, "shap_contribution": 0.28},
            {"name": "dormancy_days", "value": 180, "shap_contribution": 0.22}
        ],
        "natural_language": "This account was flagged because: 82% of credits were followed by matching debits within 24 hours, activity increased 5.2x over baseline, and the account was dormant for 180 days before the burst."
    },
    "inference_time_ms": 3.2
}
```

#### `POST /predict/batch`
```json
// Request
{
    "account_ids": ["ACC_001", "ACC_002", ..., "ACC_16015"]
}

// Response (200 OK)
{
    "predictions": [
        {"account_id": "ACC_001", "probability": 0.87, "prediction": "mule"},
        {"account_id": "ACC_002", "probability": 0.12, "prediction": "legitimate"},
        ...
    ],
    "total_accounts": 16015,
    "mules_detected": 1234,
    "processing_time_seconds": 8.5
}
```

#### `GET /model/info`
```json
// Response
{
    "model_type": "LightGBM",
    "version": "1.0.0",
    "training_date": "2025-03-01T00:00:00",
    "auc_roc": 0.954,
    "auc_pr": 0.891,
    "n_features": 57,
    "optimal_threshold": 0.52,
    "training_samples": 19218,
    "validation_samples": 4805
}
```

#### `GET /model/features`
```json
// Response
{
    "features": [
        {"name": "matched_amount_ratio", "group": "passthrough", "importance": 0.142, "rank": 1},
        {"name": "velocity_acceleration", "group": "velocity", "importance": 0.098, "rank": 2},
        ...
    ]
}
```

#### `GET /account/{account_id}`
```json
// Response
{
    "account_id": "ACC_12345",
    "profile": {
        "account_type": "savings",
        "account_age_days": 1200,
        "kyc_status": "complete",
        "declared_income": 500000
    },
    "features": { /* all 57 feature values */ },
    "prediction": {
        "probability": 0.87,
        "label": "mule",
        "threshold": 0.52
    },
    "explanation": { /* SHAP values */ },
    "suspicious_window": { "start": "...", "end": "..." },
    "transaction_summary": {
        "total_transactions": 234,
        "total_volume": 15000000,
        "date_range": {"first": "2020-08-15", "last": "2025-05-20"}
    },
    "network": {
        "in_degree": 15,
        "out_degree": 3,
        "community_id": 7,
        "pagerank": 0.0045
    }
}
```

#### `GET /dashboard/stats`
```json
// Response
{
    "total_accounts": 40038,
    "total_transactions": 7400000,
    "labeled_accounts": 24023,
    "mule_count": 2100,
    "mule_percentage": 8.7,
    "date_range": {"start": "2020-07-01", "end": "2025-06-30"},
    "model_performance": { "auc_roc": 0.954, "f1": 0.82 }
}
```

#### `GET /fairness/report`
```json
// Response
{
    "audit_date": "2025-03-01",
    "sensitive_features_analyzed": ["age_group", "geography_tier", "account_type"],
    "results": {
        "age_group": {
            "demographic_parity_difference": 0.03,
            "equalized_odds_difference": 0.05,
            "pass_80_percent_rule": true,
            "by_group": { /* per-group metrics */ }
        }
    }
}
```

#### `GET /benchmark/results`
```json
// Response
{
    "models": [
        {
            "name": "LightGBM",
            "auc_roc": {"mean": 0.954, "std": 0.008, "ci_95": [0.942, 0.966]},
            "auc_pr": {"mean": 0.891, "std": 0.012},
            "f1": {"mean": 0.82, "std": 0.015},
            "latency_ms": 2.1,
            "model_size_mb": 4.2
        },
        // ... all 6 models
    ],
    "statistical_tests": [
        {"pair": "LightGBM vs XGBoost", "p_value": 0.12, "significant": false}
    ]
}
```

#### `GET /health`
```json
{"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

---

## 9. Frontend Specification (Streamlit Dashboard)

### Recommendation: Streamlit over React

**Why Streamlit:** 3-4 day build time vs 10+ days for React. For resume impact, the ML depth matters more than frontend framework. Streamlit's interactive widgets (sliders, selectboxes, plotly charts) cover all dashboard needs. Can always mention "production version would use React" in interviews.

### Page Layouts

**Page 1: Overview** (`1_Overview.py`)
```
┌─────────────────────────────────────────────────────┐
│  RBI MULE DETECTION SYSTEM — Overview Dashboard     │
├──────────────┬──────────────┬───────────────────────┤
│ 📊 40,038     │ 📈 7.4M       │ ⚠️ 8.7% Mule Rate    │
│ Accounts     │ Transactions │ (of labeled set)      │
├──────────────┴──────────────┴───────────────────────┤
│  [Line Chart] Transaction Volume Over Time           │
│  (Monthly aggregated, color-coded by mule/legit)    │
├─────────────────────────────────────────────────────┤
│  [Histogram] Transaction Amount Distribution         │
│  (Overlaid: mule vs legitimate accounts)            │
├──────────────────────┬──────────────────────────────┤
│  [Pie Chart]         │  [Bar Chart]                 │
│  Label Distribution  │  Top 10 Mule Patterns Found  │
│  Mule vs Legitimate  │  (by feature importance)     │
└──────────────────────┴──────────────────────────────┘
```

**Page 2: Feature Explorer** (`2_Feature_Explorer.py`)
```
┌─────────────────────────────────────────────────────┐
│  Feature Engineering Explorer                        │
├─────────────────────────────────────────────────────┤
│  [Selectbox] Choose Feature Group: [All | Velocity |│
│   Amount | Temporal | PassThrough | Graph | ...]    │
├─────────────────────────────────────────────────────┤
│  [Heatmap] Feature Correlation Matrix (top 20)      │
├─────────────────────────────────────────────────────┤
│  [Side-by-side Histograms]                          │
│  Feature: [dropdown]  │  Mule █████  Legit █████    │
├─────────────────────────────────────────────────────┤
│  [Bar Chart] Top 20 Features by Importance          │
│  (SHAP-based + built-in model importance)           │
└─────────────────────────────────────────────────────┘
```

**Page 3: Model Comparison** (`3_Model_Comparison.py`)
```
┌─────────────────────────────────────────────────────┐
│  6-Model Comparison Dashboard                        │
├─────────────────────────────────────────────────────┤
│  [Plotly] Overlaid ROC Curves (all 6 models)        │
├─────────────────────────────────────────────────────┤
│  [Plotly] Overlaid PR Curves                         │
├─────────────────────────────────────────────────────┤
│  [Table] Model Comparison                            │
│  Model | AUC-ROC | AUC-PR | F1 | Prec | Rec | ms  │
├──────────────────────┬──────────────────────────────┤
│  [Heatmap]           │  [Bar + Error Bars]           │
│  Confusion Matrices  │  AUC with 95% CI             │
│  (2x3 grid)         │                               │
├──────────────────────┴──────────────────────────────┤
│  [Line Chart] Calibration Plots (all models)         │
└─────────────────────────────────────────────────────┘
```

**Page 4: Explainability** (`4_Explainability.py`)
```
┌─────────────────────────────────────────────────────┐
│  Model Explainability                                │
├─────────────────────────────────────────────────────┤
│  [SHAP Beeswarm] Global Feature Importance           │
├─────────────────────────────────────────────────────┤
│  [SHAP Bar Chart] Top 15 Features                    │
├─────────────────────────────────────────────────────┤
│  [Selectbox] Account ID: [ACC_12345 ▼]              │
│  [SHAP Waterfall] Individual Explanation             │
│  "This account flagged because..."                   │
├─────────────────────────────────────────────────────┤
│  [Selectbox] Feature: [velocity_acceleration ▼]     │
│  [PDP Plot] Partial Dependence                      │
└─────────────────────────────────────────────────────┘
```

**Page 5: Network Graph** (`5_Network_Graph.py`)
```
┌─────────────────────────────────────────────────────┐
│  Transaction Network Graph                           │
├─────────────────────────────────────────────────────┤
│  [pyvis Interactive HTML]                            │
│  Nodes = accounts (red=mule, blue=legit, gray=test) │
│  Edges = transactions (thickness=volume)             │
│  Colors by community (Louvain)                      │
│                                                      │
│  [Sidebar Filters]                                   │
│  - Min edge weight: [slider]                        │
│  - Show community: [multiselect]                    │
│  - Highlight mules only: [checkbox]                 │
├─────────────────────────────────────────────────────┤
│  [Click node] → Account details popup               │
└─────────────────────────────────────────────────────┘
```

**Page 6: Fairness Audit** (`6_Fairness_Audit.py`)
```
┌─────────────────────────────────────────────────────┐
│  Responsible AI — Fairness Audit                     │
├─────────────────────────────────────────────────────┤
│  [Selectbox] Sensitive Feature: [age_group ▼]       │
├─────────────────────────────────────────────────────┤
│  [Grouped Bar Chart] Metrics by Group               │
│  Selection Rate | Recall | Precision | F1           │
├─────────────────────────────────────────────────────┤
│  [Table] Fairness Summary                            │
│  Feature | Dem. Parity Diff | Eq. Odds Diff | Pass? │
├─────────────────────────────────────────────────────┤
│  [Note] Mitigation recommendations if bias found    │
└─────────────────────────────────────────────────────┘
```

**Page 7: Account Inspector** (`7_Account_Inspector.py`)
```
┌─────────────────────────────────────────────────────┐
│  🔍 Account Inspector                                │
│  Account ID: [________________] [Search]            │
├──────────────────────┬──────────────────────────────┤
│  Profile             │  Prediction                  │
│  Type: Savings       │  Probability: 0.87 🔴         │
│  Age: 1200 days      │  Verdict: MULE              │
│  Income: ₹5L         │  Threshold: 0.52            │
├──────────────────────┴──────────────────────────────┤
│  [Timeline Chart] Transaction History                │
│  Amount on Y-axis, Date on X-axis                   │
│  Suspicious window highlighted in red background    │
│  Credits=green dots, Debits=red dots                │
├─────────────────────────────────────────────────────┤
│  [SHAP Waterfall] Why was this flagged?              │
├──────────────────────┬──────────────────────────────┤
│  [Table]             │  [Heatmap]                   │
│  All 57 Features     │  Amount × Hour-of-Day       │
│  with values         │  (Weekly pattern)           │
└──────────────────────┴──────────────────────────────┘
```

**Page 8: API Demo** (`8_API_Demo.py`)
```
┌─────────────────────────────────────────────────────┐
│  🚀 Live Prediction API Demo                         │
├─────────────────────────────────────────────────────┤
│  Enter Account ID or paste feature JSON:            │
│  [text_area]                                         │
│  [Predict] button                                   │
├─────────────────────────────────────────────────────┤
│  Result:                                             │
│  Probability: 0.87 | Label: MULE                    │
│  Top 3 Contributing Features:                       │
│  1. matched_amount_ratio (SHAP: +0.34)              │
│  2. velocity_acceleration (SHAP: +0.28)             │
│  3. dormancy_days (SHAP: +0.22)                     │
├─────────────────────────────────────────────────────┤
│  Threshold Sensitivity:                              │
│  [slider] Threshold: 0.52                           │
│  At 0.3: MULE | At 0.5: MULE | At 0.7: MULE       │
└─────────────────────────────────────────────────────┘
```

---

## 10. Database Schema (SQLite)

```sql
-- Predictions cache
CREATE TABLE predictions (
    account_id TEXT PRIMARY KEY,
    probability REAL NOT NULL,
    prediction TEXT NOT NULL,  -- 'mule' or 'legitimate'
    threshold REAL NOT NULL,
    suspicious_start TEXT,
    suspicious_end TEXT,
    model_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature cache
CREATE TABLE features (
    account_id TEXT PRIMARY KEY,
    feature_json TEXT NOT NULL,  -- JSON blob of all 57 features
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SHAP explanations
CREATE TABLE explanations (
    account_id TEXT PRIMARY KEY,
    shap_values_json TEXT NOT NULL,  -- JSON array of SHAP values
    top_features_json TEXT NOT NULL, -- Top 10 contributing features
    natural_language TEXT NOT NULL,   -- Human-readable explanation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model registry
CREATE TABLE model_registry (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    auc_roc REAL,
    auc_pr REAL,
    f1_score REAL,
    threshold REAL,
    n_features INTEGER,
    model_path TEXT NOT NULL,  -- Path to serialized model
    training_date TIMESTAMP,
    is_active BOOLEAN DEFAULT 0,
    metadata_json TEXT          -- Hyperparameters, CV results
);

-- Fairness audit results
CREATE TABLE fairness_reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES model_registry(model_id),
    sensitive_feature TEXT NOT NULL,
    demographic_parity_diff REAL,
    equalized_odds_diff REAL,
    pass_80_rule BOOLEAN,
    details_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Benchmark results
CREATE TABLE benchmark_results (
    benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,
    auc_roc_mean REAL,
    auc_roc_std REAL,
    auc_pr_mean REAL,
    auc_pr_std REAL,
    f1_mean REAL,
    f1_std REAL,
    latency_ms REAL,
    model_size_mb REAL,
    cv_folds_json TEXT,  -- Per-fold scores
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_predictions_probability ON predictions(probability);
CREATE INDEX idx_predictions_prediction ON predictions(prediction);
CREATE INDEX idx_model_registry_active ON model_registry(is_active);
```

---

## 11. Task Breakdown (Phased, Ordered)

### Phase 1: Data Pipeline + EDA (Days 1–3)

| Task | Description | Files | Time | Depends |
|------|------------|-------|------|---------|
| 1.1 | Project setup: create directory structure, `pyproject.toml`, `.gitignore`, `.env.example` | Root files | 1h | — |
| 1.2 | Install dependencies: `pip install pandas numpy scikit-learn xgboost lightgbm catboost torch optuna shap fairlearn networkx python-louvain streamlit fastapi uvicorn plotly seaborn wandb imbalanced-learn boruta scipy` | `requirements.txt` | 30m | 1.1 |
| 1.3 | Implement data loader with memory-optimized dtypes + chunked reading | `src/data/loader.py` | 2h | 1.2 |
| 1.4 | Implement data validator (schema checks, nulls, duplicates, date ranges) | `src/data/validator.py` | 1.5h | 1.3 |
| 1.5 | Implement star-schema merger (accounts ← linkage ← customers ← products) | `src/data/merger.py` | 1.5h | 1.3 |
| 1.6 | Implement preprocessor (type casting, missing value handling, categorical encoding) | `src/data/preprocessor.py` | 1h | 1.5 |
| 1.7 | Implement train/val splitter (stratified + time-aware) | `src/data/splitter.py` | 1h | 1.6 |
| 1.8 | EDA notebook: data quality report, distribution plots, class balance analysis | `notebooks/01_eda_data_quality.ipynb` | 3h | 1.3–1.7 |

### Phase 2: Feature Engineering (Days 4–8)

| Task | Description | Files | Time | Depends |
|------|------------|-------|------|---------|
| 2.1 | Base feature generator class (abstract interface for all groups) | `src/features/base.py` | 1h | 1.7 |
| 2.2 | Group 1: Velocity features (10 features: rolling windows, acceleration) | `src/features/velocity.py` | 3h | 2.1 |
| 2.3 | Group 2: Amount pattern features (8 features: round amounts, structuring, entropy) | `src/features/amount_patterns.py` | 2.5h | 2.1 |
| 2.4 | Group 3: Temporal features (8 features: dormancy, bursts, time-of-day) | `src/features/temporal.py` | 2.5h | 2.1 |
| 2.5 | Group 4: Pass-through features (7 features: credit-debit matching, net flow) | `src/features/passthrough.py` | 3h | 2.1 |
| 2.6 | Group 5: Graph/network features — build transaction graph + compute topology | `src/features/graph_network.py` | 4h | 2.1 |
| 2.7 | Group 6: Profile mismatch features (5 features: income ratio, age vs activity) | `src/features/profile_mismatch.py` | 1.5h | 2.1, 2.2 |
| 2.8 | Group 7: KYC/behavioral features (4 features: mobile change, KYC completeness) | `src/features/kyc_behavioral.py` | 1.5h | 2.1 |
| 2.9 | Group 8: Interaction/cross features (5 features: multiplicative combinations) | `src/features/interactions.py` | 1.5h | 2.2–2.8 |
| 2.10 | Feature pipeline orchestrator (runs all groups, handles dependencies, outputs parquet) | `src/features/pipeline.py` | 2h | 2.2–2.9 |
| 2.11 | Feature registry (metadata, descriptions, expected ranges for documentation) | `src/features/registry.py` | 1h | 2.10 |
| 2.12 | Feature engineering notebook (prototyping, visualization, mule vs legit distributions) | `notebooks/02_feature_engineering.ipynb` | 3h | 2.10 |
| 2.13 | Graph analysis notebook (network visualization, community structure) | `notebooks/03_graph_analysis.ipynb` | 2h | 2.6 |

### Phase 3: Model Training + Evaluation (Days 9–14)

| Task | Description | Files | Time | Depends |
|------|------------|-------|------|---------|
| 3.1 | Base model class (abstract interface: train, predict, save, load) | `src/models/base.py` | 1h | 2.10 |
| 3.2 | Logistic Regression wrapper (with StandardScaler pipeline) | `src/models/logistic.py` | 1h | 3.1 |
| 3.3 | Random Forest wrapper + Optuna objective | `src/models/random_forest.py` | 1.5h | 3.1 |
| 3.4 | XGBoost wrapper + Optuna objective + GPU support | `src/models/xgboost_model.py` | 2h | 3.1 |
| 3.5 | LightGBM wrapper + Optuna objective + GPU support | `src/models/lightgbm_model.py` | 2h | 3.1 |
| 3.6 | CatBoost wrapper + Optuna objective + GPU support | `src/models/catboost_model.py` | 2h | 3.1 |
| 3.7 | PyTorch Neural Network + Focal Loss + Optuna objective | `src/models/neural_net.py` | 3h | 3.1 |
| 3.8 | Unified trainer (cross-validation loop, W&B logging, early stopping) | `src/models/trainer.py` | 3h | 3.2–3.7 |
| 3.9 | Model evaluator (all metrics, optimal threshold, confusion matrices) | `src/models/evaluator.py` | 2h | 3.8 |
| 3.10 | Probability calibrator (Platt scaling, isotonic regression) | `src/models/calibrator.py` | 1h | 3.9 |
| 3.11 | Model selector (compare all models, statistical tests, select best) | `src/models/selector.py` | 1.5h | 3.9 |
| 3.12 | Feature selection pipeline (variance, MI, Boruta — compare subsets) | `src/models/trainer.py` (extend) | 2h | 3.8 |
| 3.13 | Baseline modeling notebook | `notebooks/04_modeling_baseline.ipynb` | 2h | 3.8 |
| 3.14 | Full model comparison notebook (all 6 models, all plots) | `notebooks/05_model_comparison.ipynb` | 4h | 3.9–3.11 |

### Phase 4: Explainability + Fairness (Days 15–17)

| Task | Description | Files | Time | Depends |
|------|------------|-------|------|---------|
| 4.1 | SHAP explainer (TreeSHAP + KernelSHAP + Linear, global + local) | `src/explainability/shap_explainer.py` | 3h | 3.14 |
| 4.2 | PDP computation (top 10 features) | `src/explainability/pdp.py` | 1.5h | 4.1 |
| 4.3 | Natural language explanation generator | `src/explainability/natural_language.py` | 2h | 4.1 |
| 4.4 | Fairlearn audit (demographic parity, equalized odds, 80% rule) | `src/explainability/fairness.py` | 2.5h | 3.14 |
| 4.5 | Model Card generator (template + auto-fill from results) | `src/explainability/model_card.py` | 1.5h | 4.1, 4.4 |
| 4.6 | Suspicious time window detector | `src/temporal/window_detector.py` | 2.5h | 3.14 |
| 4.7 | Explainability notebook | `notebooks/06_explainability.ipynb` | 3h | 4.1–4.3 |
| 4.8 | Fairness audit notebook | `notebooks/07_fairness_audit.ipynb` | 2h | 4.4 |
| 4.9 | Temporal IoU notebook | `notebooks/08_temporal_iou.ipynb` | 2h | 4.6 |

### Phase 5: API + Serving (Days 18–19)

| Task | Description | Files | Time | Depends |
|------|------------|-------|------|---------|
| 5.1 | SQLite database schema + initialization | `src/db/models.py`, `src/db/init_db.py` | 1.5h | 4.5 |
| 5.2 | Database CRUD operations | `src/db/crud.py` | 1.5h | 5.1 |
| 5.3 | Pydantic request/response schemas | `src/api/schemas.py` | 1.5h | — |
| 5.4 | FastAPI app setup + dependencies (model loading, DB) | `src/api/main.py`, `src/api/dependencies.py` | 2h | 5.1–5.3 |
| 5.5 | Prediction routes (`/predict`, `/predict/batch`) | `src/api/routes/predict.py` | 2h | 5.4 |
| 5.6 | Account route (`/account/{id}`) | `src/api/routes/account.py` | 1.5h | 5.4 |
| 5.7 | Model, dashboard, fairness, benchmark routes | `src/api/routes/model.py` + others | 2h | 5.4 |
| 5.8 | Seed database with all predictions, features, SHAP values | Script | 2h | 5.2, 4.1 |

### Phase 6: Frontend Dashboard (Days 20–23)

| Task | Description | Files | Time | Depends |
|------|------------|-------|------|---------|
| 6.1 | Streamlit app structure + sidebar navigation | `frontend/app.py`, `frontend/components/` | 1h | 5.8 |
| 6.2 | Page 1: Overview (stats, EDA charts, class distribution) | `frontend/pages/1_Overview.py` | 2.5h | 6.1 |
| 6.3 | Page 2: Feature Explorer (correlation, distributions, importance) | `frontend/pages/2_Feature_Explorer.py` | 3h | 6.1 |
| 6.4 | Page 3: Model Comparison (ROC/PR curves, table, calibration) | `frontend/pages/3_Model_Comparison.py` | 3h | 6.1 |
| 6.5 | Page 4: Explainability (SHAP plots, per-account, PDP) | `frontend/pages/4_Explainability.py` | 3h | 6.1 |
| 6.6 | Page 5: Network Graph (pyvis interactive visualization) | `frontend/pages/5_Network_Graph.py` | 3h | 6.1 |
| 6.7 | Page 6: Fairness Audit (grouped metrics, recommendations) | `frontend/pages/6_Fairness_Audit.py` | 2h | 6.1 |
| 6.8 | Page 7: Account Inspector (search, timeline, SHAP, heatmap) | `frontend/pages/7_Account_Inspector.py` | 3h | 6.1 |
| 6.9 | Page 8: API Demo (live prediction with explanation) | `frontend/pages/8_API_Demo.py` | 1.5h | 6.1, 5.5 |

### Phase 7: Documentation + CI/CD (Days 24–25)

| Task | Description | Files | Time | Depends |
|------|------------|-------|------|---------|
| 7.1 | Write comprehensive README.md | `README.md` | 2h | All |
| 7.2 | Architecture documentation | `docs/architecture.md` | 1.5h | All |
| 7.3 | Feature documentation (all 57 features) | `docs/feature_documentation.md` | 1.5h | 2.11 |
| 7.4 | API reference documentation | `docs/api_reference.md` | 1h | 5.7 |
| 7.5 | Interview preparation guide | `docs/interview_prep.md` | 2h | All |
| 7.6 | Unit tests for data pipeline | `tests/test_data/` | 2h | 1.3–1.7 |
| 7.7 | Unit tests for features | `tests/test_features/` | 3h | 2.2–2.9 |
| 7.8 | Integration tests for API | `tests/test_api/` | 1.5h | 5.5–5.7 |
| 7.9 | Docker configuration (API + Dashboard + Training) | `docker/`, `docker-compose.yml` | 2h | All |
| 7.10 | GitHub Actions CI pipeline | `.github/workflows/ci.yml` | 1.5h | 7.6–7.8 |
| 7.11 | Makefile with all commands | `Makefile` | 1h | All |

### Phase 8: Hackathon Submission (Day 26)

| Task | Description | Files | Time | Depends |
|------|------------|-------|------|---------|
| 8.1 | Generate predictions for all 16,015 test accounts | `notebooks/09_submission.ipynb` | 2h | 3.14, 4.6 |
| 8.2 | Format submission CSV (account_id, probability, suspicious_start, suspicious_end) | Same notebook | 1h | 8.1 |
| 8.3 | Validate submission format & sanity check predictions | Same notebook | 1h | 8.2 |
| 8.4 | Final model selection & documentation of choices | `docs/model_card.md` | 1h | 8.3 |

**TOTAL ESTIMATED TIME: ~110 hours (3.5 weeks at 4.5 hours/day)**

---

## 12. Docker Compose Configuration

```yaml
# docker-compose.yml

version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./outputs/models:/app/models:ro
      - ./data/processed:/app/data:ro
      - ./outputs/db:/app/db
    environment:
      - MODEL_PATH=/app/models/best_model.joblib
      - DB_PATH=/app/db/mule_detection.db
      - FEATURES_PATH=/app/data/features_matrix.parquet
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "8501:8501"
    volumes:
      - ./outputs:/app/outputs:ro
      - ./data/processed:/app/data:ro
    environment:
      - API_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped

  training:
    build:
      context: .
      dockerfile: docker/Dockerfile.training
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./src:/app/src
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    profiles:
      - training  # Only starts with: docker compose --profile training up

# Dockerfile.api
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements-api.txt .
# RUN pip install --no-cache-dir -r requirements-api.txt
# COPY src/ ./src/
# COPY outputs/models/ ./models/
# EXPOSE 8000
# CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Dockerfile.dashboard
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements-dashboard.txt .
# RUN pip install --no-cache-dir -r requirements-dashboard.txt
# COPY frontend/ ./frontend/
# COPY outputs/ ./outputs/
# EXPOSE 8501
# CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 13. Makefile

```makefile
# Makefile

.PHONY: setup data features train evaluate explain fairness serve dashboard test lint submit clean all

# Environment
PYTHON := python3
PIP := pip3

# ─── SETUP ───────────────────────────────────────────
setup:
	$(PIP) install -r requirements.txt --break-system-packages
	$(PIP) install -e . --break-system-packages
	mkdir -p data/raw data/processed outputs/models outputs/predictions outputs/reports outputs/plots outputs/shap_values outputs/db

# ─── DATA PIPELINE ───────────────────────────────────
data:
	$(PYTHON) -m src.data.loader
	$(PYTHON) -m src.data.validator
	$(PYTHON) -m src.data.merger
	$(PYTHON) -m src.data.preprocessor

validate:
	$(PYTHON) -m src.data.validator --verbose

# ─── FEATURE ENGINEERING ──────────────────────────────
features:
	$(PYTHON) -m src.features.pipeline --output data/processed/features_matrix.parquet

features-fast:
	$(PYTHON) -m src.features.pipeline --skip-graph --output data/processed/features_matrix.parquet

# ─── MODEL TRAINING ──────────────────────────────────
train:
	$(PYTHON) -m src.models.trainer --all-models --optuna-trials 100

train-quick:
	$(PYTHON) -m src.models.trainer --model lightgbm --optuna-trials 20

train-baseline:
	$(PYTHON) -m src.models.trainer --model logistic

# ─── EVALUATION ──────────────────────────────────────
evaluate:
	$(PYTHON) -m src.models.evaluator --all-models --output outputs/reports/

compare:
	$(PYTHON) -m src.models.selector --output outputs/reports/model_comparison.json

# ─── EXPLAINABILITY ──────────────────────────────────
explain:
	$(PYTHON) -m src.explainability.shap_explainer --output outputs/shap_values/
	$(PYTHON) -m src.explainability.pdp --output outputs/plots/

fairness:
	$(PYTHON) -m src.explainability.fairness --output outputs/reports/fairness_report.json

model-card:
	$(PYTHON) -m src.explainability.model_card --output docs/model_card.md

# ─── TEMPORAL DETECTION ──────────────────────────────
temporal:
	$(PYTHON) -m src.temporal.window_detector --output outputs/predictions/suspicious_windows.csv

# ─── API SERVER ──────────────────────────────────────
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ─── DASHBOARD ───────────────────────────────────────
dashboard:
	streamlit run frontend/app.py --server.port 8501

# ─── DATABASE ────────────────────────────────────────
db-init:
	$(PYTHON) -m src.db.init_db

db-seed:
	$(PYTHON) -m src.db.crud --seed

# ─── HACKATHON SUBMISSION ────────────────────────────
submit:
	$(PYTHON) -m src.models.trainer --predict-test --output outputs/predictions/submission.csv
	$(PYTHON) -c "import pandas as pd; df=pd.read_csv('outputs/predictions/submission.csv'); print(f'Submission: {len(df)} rows, columns: {list(df.columns)}')"

# ─── TESTING ─────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-data:
	pytest tests/test_data/ -v

test-features:
	pytest tests/test_features/ -v

test-api:
	pytest tests/test_api/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html

# ─── CODE QUALITY ────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

# ─── DOCKER ──────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-train:
	docker compose --profile training up training

# ─── CLEAN ───────────────────────────────────────────
clean:
	rm -rf data/processed/*
	rm -rf outputs/models/*
	rm -rf outputs/predictions/*
	rm -rf outputs/reports/*
	rm -rf outputs/plots/*
	rm -rf __pycache__ .pytest_cache .ruff_cache

# ─── FULL PIPELINE ───────────────────────────────────
all: setup data features train evaluate explain fairness model-card temporal submit
	@echo "✅ Full pipeline complete!"
```

---

## 14. Environment Variables

```bash
# .env.example

# ─── Weights & Biases ────────────────────────────────
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=rbi-mule-detection
WANDB_ENTITY=your_wandb_username

# ─── Data Paths ──────────────────────────────────────
DATA_RAW_DIR=data/raw
DATA_PROCESSED_DIR=data/processed
OUTPUTS_DIR=outputs

# ─── Model Configuration ─────────────────────────────
MODEL_PATH=outputs/models/best_model.joblib
SHAP_VALUES_PATH=outputs/shap_values/shap_values.npy
FEATURES_PATH=data/processed/features_matrix.parquet

# ─── Database ─────────────────────────────────────────
DB_PATH=outputs/db/mule_detection.db

# ─── API Configuration ───────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO

# ─── Dashboard ────────────────────────────────────────
API_URL=http://localhost:8000
STREAMLIT_PORT=8501

# ─── GPU Configuration ───────────────────────────────
CUDA_VISIBLE_DEVICES=0

# ─── Optuna ──────────────────────────────────────────
OPTUNA_N_TRIALS=100
OPTUNA_TIMEOUT=3600  # Max seconds per model

# ─── Feature Engineering ─────────────────────────────
FEATURE_CUTOFF_DATE=2025-06-30
STRUCTURING_THRESHOLD=50000
DORMANCY_THRESHOLD_DAYS=90
```

---

## 15. Testing Strategy

### What to Test

**Unit Tests (must pass on every commit):**

1. **Data Pipeline Tests** (`tests/test_data/`)
   - `test_loader.py`: Verify chunked loading works, dtypes are correct, row counts match
   - `test_validator.py`: Test validation catches nulls, bad dates, duplicate IDs
   - `test_merger.py`: Verify merge produces expected shape, no duplicate account_ids

2. **Feature Tests** (`tests/test_features/`)
   - For each feature group: create synthetic data with known properties, verify feature values
   - Example: Create account with 10 transactions at ₹49,000 → verify `structuring_score` ≈ 1.0
   - Example: Create account with 90-day gap + burst → verify `dormancy_days` = 90
   - Example: Create pass-through pattern → verify `matched_amount_ratio` > 0.8
   - Test point-in-time correctness: verify no future data leaks into features
   - Test edge cases: account with 0 transactions, 1 transaction, all nulls

3. **Model Tests** (`tests/test_models/`)
   - Test that all 6 models produce predictions in [0, 1]
   - Test that trained model AUC > random baseline (0.5)
   - Test model serialization/deserialization roundtrip
   - Test evaluator produces all expected metrics

4. **API Tests** (`tests/test_api/`)
   - Test all endpoints return correct HTTP status codes
   - Test `/predict` returns valid probability and explanation
   - Test `/predict/batch` handles 100+ accounts
   - Test `/health` returns 200

```python
# tests/conftest.py — Shared fixtures

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_transactions():
    """Create 1000 synthetic transactions for testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'account_id': np.random.choice(['ACC_001', 'ACC_002', 'ACC_003'], n),
        'transaction_date': pd.date_range('2024-01-01', periods=n, freq='h'),
        'transaction_amount': np.random.exponential(5000, n),
        'is_credit': np.random.randint(0, 2, n),
        'counterparty_id': np.random.choice(['CP_A', 'CP_B', 'CP_C', 'CP_D'], n),
        'transaction_type': np.random.choice(['NEFT', 'UPI', 'IMPS'], n),
    })

@pytest.fixture
def sample_features():
    """Create feature matrix for 100 accounts."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'account_id': [f'ACC_{i:03d}' for i in range(n)],
        'txn_count_7d': np.random.randint(0, 50, n),
        'structuring_score': np.random.uniform(0, 1, n),
        'dormancy_days': np.random.randint(0, 365, n),
        'matched_amount_ratio': np.random.uniform(0, 1, n),
        'pagerank': np.random.uniform(0, 0.01, n),
    })

# Example test:
# tests/test_features/test_amount_patterns.py

def test_structuring_score_all_below_threshold():
    amounts = pd.Series([48000, 49000, 49500, 47000, 46000])
    score = structuring_score(amounts)
    assert score == 1.0  # All amounts in [45000, 49999]

def test_structuring_score_none_below_threshold():
    amounts = pd.Series([1000, 5000, 10000, 100000])
    score = structuring_score(amounts)
    assert score == 0.0

def test_round_amount_ratio():
    amounts = pd.Series([1000, 5000, 10000, 1234, 5678])
    ratio = round_amount_ratio(amounts)
    assert ratio == 0.6  # 3 out of 5 are round
```

---

## 16. README Structure

```markdown
# 🔍 RBI Mule Account Detection System

> Production-grade ML system for detecting money laundering intermediary accounts
> using 7.4M banking transactions across 40,000 accounts.

## 🏆 Key Results
- **AUC-ROC: 0.XX** (LightGBM) across 6-model comparison
- **57 engineered features** including graph topology, velocity patterns, and behavioral anomalies
- **Full explainability**: SHAP explanations for every prediction
- **Fairness audited**: Fairlearn demographic analysis across 4 protected groups
- Built for the **RBI Innovation Hub Hackathon** (July 2020 – June 2025 dataset)

## 📊 Architecture
[System architecture diagram]

## 🚀 Quick Start
[Setup, data download, run pipeline instructions]

## 🧬 Feature Engineering (The Differentiator)
[Summary of 8 feature groups, 57 features]

## 🤖 Models
[6-model comparison table with results]

## 📈 Dashboard
[Screenshots of all 8 dashboard pages]

## 🔬 Explainability
[SHAP examples, natural language explanations]

## ⚖️ Fairness
[Fairlearn audit results]

## 🐳 Docker
[docker compose up instructions]

## 📚 Documentation
[Links to detailed docs]

## 🧪 Testing
[How to run tests]

## 🎯 Interview Guide
[Key questions this project answers]

## 📝 License
MIT
```

---

## 17. GitHub Actions CI/CD Pipeline

```yaml
# .github/workflows/ci.yml

name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --tb=short --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml

  model-validation:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Validate model performance
        run: |
          python -c "
          import json
          with open('outputs/reports/model_comparison.json') as f:
              results = json.load(f)
          best_auc = results['best_model']['auc_roc']
          assert best_auc > 0.85, f'Model AUC {best_auc} below threshold 0.85'
          print(f'✅ Model validation passed: AUC-ROC = {best_auc}')
          "
```

```yaml
# .github/workflows/model_validation.yml

name: Model Performance Check

on:
  workflow_dispatch:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6AM

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run model regression test
        run: python -m src.models.evaluator --regression-test
```

---

## Interview Preparation Guide (Bonus)

### Q1: "Walk me through your feature engineering process. Why these features?"

**Answer framework:** "I designed 57 features across 8 groups, each targeting specific money laundering patterns identified in the RBI dataset. For example, **velocity features** detect sudden activity bursts (mule pattern #1), **structuring scores** catch transactions deliberately kept below ₹50K reporting thresholds (pattern #2), **pass-through features** identify rapid credit-to-debit turnaround (pattern #3), and **graph features** like PageRank and community detection reveal network collusion (patterns #4 and #12). The interaction features capture compound signals — for instance, `dormancy_x_burst` combines a 180-day gap with a subsequent 7-day transaction spike, which is extremely rare in legitimate accounts but common in mule activation."

### Q2: "Why did XGBoost/LightGBM outperform the neural network?"

**Answer:** "For tabular data with well-engineered features, gradient boosted trees consistently outperform neural networks. The key reasons are: (1) tree models natively handle feature interactions without manual specification, (2) they don't require feature scaling, (3) they handle missing values gracefully, and (4) our feature engineering already captured the complex patterns — the NN had no latent representation advantage. This aligns with benchmarks like the 'Tabular Data' paper (Grinsztajn et al., 2022) showing trees dominate on structured data with fewer than ~10K features."

### Q3: "How did you prevent data leakage?"

**Answer:** "Five key measures: (1) all rolling window features look backward only from the cutoff date, (2) graph features exclude test account edges during training, (3) community mule density uses only training labels — never validation/test labels, (4) SMOTE is applied only within training folds during cross-validation, and (5) the entire feature pipeline takes a `cutoff_date` parameter ensuring no future information contaminates features."

### Q4: "Explain PageRank for fraud detection."

**Answer:** "PageRank measures how much money 'flows through' an account in the transaction network. A mule account — even if it has few direct connections — has high PageRank because it sits on critical money flow paths between many senders and receivers. It's the same principle as web search: a web page linked by many important pages is itself important. A mule account receiving from many fraud sources and distributing to many destinations becomes a high-PageRank node, even though the account holder may appear ordinary."

### Q5: "How would you scale this to 100x?"

**Answer:** "Three levels of scaling: (1) **Data processing**: Replace pandas with Spark or Polars for distributed/parallel computation of the 740M transactions. (2) **Feature engineering**: Pre-aggregate transaction features using time-partitioned tables in a data warehouse (BigQuery/Redshift) with SQL window functions. (3) **Model serving**: Deploy the LightGBM model behind a load-balanced API with Redis caching of feature vectors. Graph features would use a graph database (Neo4j) for real-time traversal. The feature pipeline would become a scheduled Airflow DAG computing incremental features daily."
