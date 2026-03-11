# RBI Mule Account Detection — Complete Execution Guide (Claude Code CLI)

> **How to use:** Copy-paste each command block into your terminal. Each task is self-contained.
> Run them in order within each phase. Tasks marked **[PARALLEL]** can run simultaneously in separate terminals.
> Every task includes built-in validation — it won't finish until tests pass.

---

## PRE-REQUISITES

```bash
# Run this ONCE before starting
mkdir -p ~/rbi-mule-detection && cd ~/rbi-mule-detection
git init

# Copy your RBI hackathon CSV files into data/raw/ after Task 1.1 creates the structure
# Files needed: customers.csv, accounts.csv, customer_account_linkage.csv,
#   product_details.csv, transactions_part_0.csv through transactions_part_5.csv,
#   train_labels.csv, test_accounts.csv
```

---

## PHASE 1: Foundation + Data Pipeline (Tasks 1.1 – 1.9)

---

### TASK 1.1 — Project Scaffolding + Dependencies

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)

Create the complete production ML project scaffolding. Do the following:

1. Create the full directory structure:
rbi-mule-detection/
├── .github/workflows/
├── data/raw/
├── data/processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data/__init__.py
│   ├── features/__init__.py
│   ├── models/__init__.py
│   ├── explainability/__init__.py
│   ├── temporal/__init__.py
│   ├── api/__init__.py
│   ├── api/routes/__init__.py
│   ├── db/__init__.py
│   └── utils/__init__.py
├── frontend/
│   ├── pages/
│   ├── components/
│   └── assets/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data/__init__.py
│   ├── test_features/__init__.py
│   ├── test_models/__init__.py
│   ├── test_api/__init__.py
│   └── test_temporal/__init__.py
├── docker/
│   └── otel/
├── docs/
├── outputs/models/
├── outputs/predictions/
├── outputs/reports/
├── outputs/plots/
├── outputs/shap_values/
├── outputs/db/
└── outputs/logs/

2. Create pyproject.toml with:
   [project]
   name = 'rbi-mule-detection'
   version = '1.0.0'
   description = 'Production-grade mule account detection for RBI Innovation Hub'
   requires-python = '>=3.10'
   dependencies = [
       'pandas>=2.0', 'numpy>=1.24', 'scipy>=1.11', 'pyarrow>=14.0',
       'scikit-learn>=1.3', 'xgboost>=2.0', 'lightgbm>=4.0', 'catboost>=1.2',
       'torch>=2.1', 'optuna>=3.4',
       'imbalanced-learn>=0.11',
       'networkx>=3.2', 'python-louvain>=0.16',
       'shap>=0.43', 'fairlearn>=0.9',
       'fastapi>=0.104', 'uvicorn>=0.24', 'pydantic>=2.5',
       'streamlit>=1.28', 'plotly>=5.18', 'matplotlib>=3.8', 'seaborn>=0.13',
       'pyvis>=0.3',
       'joblib>=1.3', 'nbformat>=5.9',
       'pytest>=7.4', 'pytest-cov>=4.1', 'httpx>=0.25',
       'ruff>=0.1',
   ]
   [tool.pytest.ini_options]
   testpaths = ['tests']
   markers = ['unit: unit tests', 'integration: requires real data']

3. Create requirements.txt mirroring the same dependencies with pinned versions.

4. Create .gitignore covering: data/raw/, data/processed/, outputs/, __pycache__/, *.pyc, .env, *.egg-info/, .pytest_cache/, .ruff_cache/, wandb/, *.joblib, *.pt, *.parquet, .ipynb_checkpoints/

5. Create .env.example with ALL variables:
   WANDB_API_KEY=your_key_here
   WANDB_PROJECT=rbi-mule-detection
   DATA_RAW_DIR=data/raw
   DATA_PROCESSED_DIR=data/processed
   OUTPUTS_DIR=outputs
   MODEL_PATH=outputs/models/best_model.joblib
   DB_PATH=outputs/db/mule_detection.db
   API_HOST=0.0.0.0
   API_PORT=8000
   STREAMLIT_PORT=8501
   CUDA_VISIBLE_DEVICES=0
   OPTUNA_N_TRIALS=100
   FEATURE_CUTOFF_DATE=2025-06-30
   STRUCTURING_THRESHOLD=50000
   DORMANCY_THRESHOLD_DAYS=90

6. Create src/utils/config.py:
   - Use os.environ.get() with defaults matching .env.example
   - Expose DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUTS_DIR, MODEL_PATH, DB_PATH as pathlib.Path objects
   - RANDOM_SEED = 42
   - Auto-create directories if they don't exist

7. Create src/utils/constants.py:
   - TRANSACTION_DTYPES dict for memory-optimized loading:
     {'account_id': 'str', 'transaction_id': 'str', 'transaction_date': 'str',
      'transaction_amount': 'float32', 'transaction_type': 'category',
      'channel': 'category', 'counterparty_id': 'str', 'is_credit': 'int8', 'balance_after': 'float32'}
   - FEATURE_GROUPS = ['velocity', 'amount_patterns', 'temporal', 'passthrough', 'graph_network', 'profile_mismatch', 'kyc_behavioral', 'interactions']
   - MODEL_NAMES = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'neural_net']
   - ROUND_AMOUNTS = [1000, 5000, 10000, 50000]
   - STRUCTURING_LOWER = 45000
   - STRUCTURING_UPPER = 49999

8. Create src/utils/logging_config.py:
   - setup_logger(name) function returning a configured logger
   - StreamHandler with format: '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
   - FileHandler writing to outputs/logs/app.log

9. Create src/utils/metrics.py with these functions:
   - temporal_iou(pred_start, pred_end, true_start, true_end) -> float
   - find_optimal_threshold(y_true, y_prob, method='youden') -> float (using sklearn ROC curve)
   - find_f1_threshold(y_true, y_prob) -> float (using PR curve)
   - compute_all_metrics(y_true, y_prob, threshold=None) -> dict with keys: auc_roc, auc_pr, f1, precision, recall, brier_score, confusion_matrix, threshold_used

10. Create Makefile with targets:
   - setup: pip install -e . --break-system-packages
   - data: python -m src.data.loader && python -m src.data.validator
   - features: python -m src.features.pipeline
   - features-fast: python -m src.features.pipeline --skip-graph
   - train: python -m src.models.trainer --all-models
   - train-quick: python -m src.models.trainer --model lightgbm --optuna-trials 20
   - evaluate: python -m src.models.evaluator --all-models
   - explain: python -m src.explainability.shap_explainer
   - fairness: python -m src.explainability.fairness
   - temporal: python -m src.temporal.window_detector
   - serve: uvicorn src.api.main:app --reload --port 8000
   - dashboard: streamlit run frontend/app.py
   - submit: python -m src.models.trainer --predict-test --output outputs/predictions/submission.csv
   - test: pytest tests/ -v --tb=short
   - test-data: pytest tests/test_data/ -v
   - test-features: pytest tests/test_features/ -v
   - lint: ruff check src/ tests/
   - clean: rm -rf data/processed/* outputs/* __pycache__
   - all: setup data features train evaluate explain fairness temporal submit

11. Create data/README.md with download instructions
12. Create a minimal README.md with project title and 'Full setup instructions coming soon'

13. Install all dependencies:
   pip install pandas numpy scipy pyarrow scikit-learn xgboost lightgbm catboost optuna imbalanced-learn networkx python-louvain shap fairlearn fastapi uvicorn pydantic streamlit plotly matplotlib seaborn pyvis joblib nbformat pytest pytest-cov httpx ruff torch --break-system-packages

VALIDATION after creating everything:
- Run: find . -name '*.py' -path '*/src/*' | head -20 (should show all __init__.py files)
- Run: python -c 'from src.utils.config import DATA_RAW_DIR, OUTPUTS_DIR; print(f\"Config OK: {DATA_RAW_DIR}\")' — should print path
- Run: python -c 'from src.utils.constants import TRANSACTION_DTYPES, FEATURE_GROUPS; print(f\"Constants OK: {len(FEATURE_GROUPS)} groups\")' — should show 8
- Run: python -c 'from src.utils.metrics import compute_all_metrics; print(\"Metrics OK\")'
- Run: python -c 'import xgboost, lightgbm, catboost, torch, shap; print(\"All ML libs OK\")'
- Run: make --dry-run all (should show command chain)
- Run: cat .env.example | wc -l (should be 15+ lines)
Print 'TASK 1.1 COMPLETE — Scaffolding verified' when all validations pass.
"
```

---

### TASK 1.2 — Data Loader (Chunked, Memory-Optimized)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Project scaffolding exists. src/utils/config.py has DATA_RAW_DIR, DATA_PROCESSED_DIR as Path objects. src/utils/constants.py has TRANSACTION_DTYPES dict.

Create the data loader for 7.4M banking transactions with memory optimization.

1. Create src/data/loader.py:
   import pandas as pd, numpy as np, logging
   from pathlib import Path
   from src.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
   from src.utils.constants import TRANSACTION_DTYPES

   Function load_transactions(data_dir: Path = None, nrows: int = None) -> pd.DataFrame:
   - Default data_dir = DATA_RAW_DIR
   - Load transactions_part_0.csv through transactions_part_5.csv
   - Use TRANSACTION_DTYPES for memory optimization (float32, category, int8)
   - parse_dates=['transaction_date']
   - pd.concat all 6 parts into one DataFrame
   - Sort by ['account_id', 'transaction_date'], reset index
   - Log: row count, memory usage before and after dtype optimization
   - Handle FileNotFoundError gracefully (log warning, skip missing files)
   - Return single sorted DataFrame (~7.4M rows)

   Function load_static_tables(data_dir: Path = None) -> dict:
   - Load: customers.csv, accounts.csv, customer_account_linkage.csv, product_details.csv, train_labels.csv, test_accounts.csv
   - Return dict: {'customers': df, 'accounts': df, 'linkage': df, 'products': df, 'labels': df, 'test_ids': df}
   - If a file is missing, set its value to None and log warning

   Function load_all(data_dir: Path = None) -> tuple[pd.DataFrame, dict]:
   - Calls both functions above
   - Returns (transactions_df, static_tables_dict)

   if __name__ == '__main__':
   - Load all data, print shapes and memory for each table
   - Save transactions to DATA_PROCESSED_DIR / 'transactions_clean.parquet'

2. Create tests/test_data/test_loader.py:
   - Create tests/test_data/fixtures/ directory with a tiny CSV fixture (5 rows mimicking transaction format)
   - test_load_transactions_returns_dataframe(): load fixture, verify it's a DataFrame
   - test_load_transactions_sorted(): verify sorted by account_id then transaction_date
   - test_load_transactions_correct_dtypes(): verify float32 for amounts, category for types
   - test_load_static_tables_returns_dict(): verify all 6 keys present
   - test_missing_file_handled_gracefully(): load from empty dir, verify no crash

VALIDATION:
- python -c 'from src.data.loader import load_transactions, load_static_tables, load_all; print(\"Loader imports OK\")'
- python -m pytest tests/test_data/test_loader.py -v — all tests must pass
Print 'TASK 1.2 COMPLETE — Data loader verified' when all pass.
"
```

---

### TASK 1.3 — Data Validator

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Data loader exists in src/data/loader.py.

Create comprehensive data quality validator.

1. Create src/data/validator.py:

   class DataValidator:
       def __init__(self):
           self.logger = logging.getLogger(__name__)

       def validate_transactions(self, txn: pd.DataFrame) -> dict:
           report = {}
           required = ['account_id', 'transaction_date', 'transaction_amount', 'is_credit']
           report['missing_columns'] = [c for c in required if c not in txn.columns]
           report['total_rows'] = len(txn)
           report['null_pct'] = (txn.isnull().sum() / len(txn) * 100).round(2).to_dict()
           report['duplicate_txn_ids'] = txn.get('transaction_id', pd.Series()).duplicated().sum()
           report['date_range'] = {'min': str(txn['transaction_date'].min()), 'max': str(txn['transaction_date'].max())}
           report['negative_amounts'] = int((txn['transaction_amount'] < 0).sum())
           report['zero_amounts'] = int((txn['transaction_amount'] == 0).sum())
           report['unique_accounts'] = txn['account_id'].nunique()
           return report

       def validate_accounts(self, accounts: pd.DataFrame) -> dict:
           Check required columns, null percentages, duplicate account_ids, account_type distribution

       def validate_labels(self, labels: pd.DataFrame) -> dict:
           Total count, mule_count, mule_pct, legitimate_count

       def validate_consistency(self, static_tables: dict) -> dict:
           Cross-table checks: accounts without linkage, linkage without accounts, labels without accounts

       def run_full_validation(self, txn=None, static_tables=None) -> dict:
           Run all validators, print formatted JSON summary, return combined report

   if __name__ == '__main__':
       Load data and run full validation

2. Create tests/test_data/test_validator.py:
   - test_detects_missing_columns(): df without 'account_id' → appears in missing_columns
   - test_detects_null_percentages(): df with NaN → null_pct > 0
   - test_validate_labels_counts(): 3 mules out of 10 → mule_pct = 30.0
   - test_full_validation_runs(): sample data → no exceptions raised

VALIDATION:
- python -c 'from src.data.validator import DataValidator; print(\"Validator OK\")'
- python -m pytest tests/test_data/test_validator.py -v — all tests must pass
Print 'TASK 1.3 COMPLETE — Data validator verified' when all pass.
"
```

---

### TASK 1.4 — Data Merger (Star Schema)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Loader in src/data/loader.py returns static_tables dict with keys: accounts, linkage, customers, products, labels, test_ids.

Create star-schema merger joining all static tables into one account-level profile.

Merge strategy:
  accounts (40,038 rows)
    LEFT JOIN linkage ON account_id → adds customer_id
    LEFT JOIN customers ON customer_id → adds demographics
    LEFT JOIN products ON customer_id → adds product holdings

1. Create src/data/merger.py:

   def build_account_profile(static_tables: dict) -> pd.DataFrame:
   - Perform the 3-step merge chain above
   - Log rows before/after each join
   - Verify no row explosion (output should be <= max input rows + small margin)
   - Handle duplicate column names with suffixes
   - Derive: account_age_days = (cutoff - account_open_date).days if column exists
   - Derive: is_new_account = 1 if account_age_days < 180 else 0
   - Return merged DataFrame

   def add_labels(profile: pd.DataFrame, labels: pd.DataFrame, test_ids: pd.DataFrame) -> pd.DataFrame:
   - Merge labels onto profile (LEFT JOIN on account_id)
   - Add 'dataset' column: 'train' if in labels, 'test' if in test_ids, 'unlabeled' otherwise
   - Return complete profile

   if __name__ == '__main__':
   - Load data, build profile, add labels
   - Save to data/processed/merged_accounts.parquet
   - Print shape and dataset distribution

2. Create tests/test_data/test_merger.py:
   - Create tiny DataFrames mimicking all 6 tables (3-5 rows each)
   - test_merge_preserves_row_count(): output rows reasonable
   - test_no_duplicate_account_ids(): account_id unique after merge
   - test_labels_correctly_assigned(): train/test/unlabeled markers correct
   - test_account_age_computed(): if account_open_date present, age > 0

VALIDATION:
- python -c 'from src.data.merger import build_account_profile, add_labels; print(\"Merger OK\")'
- python -m pytest tests/test_data/test_merger.py -v — all tests must pass
Print 'TASK 1.4 COMPLETE — Data merger verified' when all pass.
"
```

---

### TASK 1.5 — Data Preprocessor + Splitter

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Merger produces account-level profile with 'dataset' and 'is_mule' columns.

Create preprocessor and train/val splitter.

1. Create src/data/preprocessor.py:

   def preprocess_transactions(txn: pd.DataFrame) -> pd.DataFrame:
   - Fill missing transaction_amount with 0
   - Fill missing counterparty_id with 'UNKNOWN'
   - Ensure transaction_date is datetime
   - Add derived columns:
     transaction_hour (0-23), transaction_dow (0=Mon, 6=Sun),
     is_weekend (1 if Sat/Sun), is_night (1 if hour 23-5),
     amount_log = np.log1p(transaction_amount)
   - Return preprocessed DataFrame

   def preprocess_profile(profile: pd.DataFrame) -> pd.DataFrame:
   - Fill missing numeric columns with median
   - Fill missing categorical columns with 'UNKNOWN'
   - Return preprocessed DataFrame

2. Create src/data/splitter.py:

   from sklearn.model_selection import train_test_split, StratifiedKFold

   def split_train_val(profile: pd.DataFrame, test_size=0.2, random_state=42):
   - Filter to rows where dataset == 'train'
   - Stratified split on is_mule column
   - Return (train_df, val_df)

   def get_test_accounts(profile: pd.DataFrame) -> pd.DataFrame:
   - Return rows where dataset == 'test'

   def create_cv_folds(X, y, n_splits=5, random_state=42):
   - StratifiedKFold, return list of (train_idx, val_idx)

3. Create tests/test_data/test_preprocessor.py:
   - test_adds_hour_column(): verify transaction_hour exists
   - test_adds_weekend_flag(): Saturday → is_weekend=1
   - test_fills_null_amounts(): NaN becomes 0
   - test_split_preserves_class_ratio(): mule ratio within ±2%
   - test_no_test_in_train(): no test account_ids in train split

VALIDATION:
- python -c 'from src.data.preprocessor import preprocess_transactions; from src.data.splitter import split_train_val; print(\"Preprocessor+Splitter OK\")'
- python -m pytest tests/test_data/test_preprocessor.py -v — all tests must pass
Print 'TASK 1.5 COMPLETE — Preprocessor and splitter verified' when all pass.
"
```

---

### TASK 1.6 — Base Feature Generator + Feature Registry

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Data pipeline complete. Now building the feature engineering layer. All features computed per account_id.

Create the abstract base class and complete registry of all 57 features.

1. Create src/features/base.py:
   from abc import ABC, abstractmethod
   import pandas as pd, numpy as np
   from typing import Optional

   class BaseFeatureGenerator(ABC):
       def __init__(self, name: str, group: str):
           self.name = name
           self.group = group

       @abstractmethod
       def compute(self, txn: pd.DataFrame, profile: pd.DataFrame,
                   cutoff_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
           '''Return DataFrame indexed by account_id with feature columns.'''
           pass

       @abstractmethod
       def get_feature_names(self) -> list[str]:
           pass

       def get_feature_descriptions(self) -> dict[str, str]:
           return {}

       def validate_output(self, features: pd.DataFrame) -> bool:
           expected = set(self.get_feature_names())
           actual = set(features.columns)
           missing = expected - actual
           if missing:
               raise ValueError(f'{self.name}: Missing features: {missing}')
           if features.isin([float('inf'), float('-inf')]).any().any():
               import warnings
               warnings.warn(f'{self.name}: Replacing infinite values with 0')
               features.replace([float('inf'), float('-inf')], 0, inplace=True)
           return True

2. Create src/features/registry.py:
   FEATURE_REGISTRY dict with ALL 57 features. Each entry:
     'feature_name': {'group': str, 'description': str, 'dtype': str, 'power': 'High'|'Medium'}

   The 57 features organized by group:
   VELOCITY (10): txn_count_1d, txn_count_7d, txn_count_30d, txn_count_90d, txn_amount_mean_30d, txn_amount_max_30d, txn_amount_std_30d, txn_amount_sum_30d, velocity_acceleration, frequency_change_ratio
   AMOUNT_PATTERNS (8): round_amount_ratio, structuring_score, structuring_score_broad, amount_entropy, amount_skewness, amount_kurtosis, pct_above_10k, amount_concentration
   TEMPORAL (8): dormancy_days, max_gap_days, burst_after_dormancy, unusual_hour_ratio, weekend_ratio, night_weekend_combo, monthly_txn_cv, days_to_first_txn
   PASSTHROUGH (7): credit_debit_time_delta_median, credit_debit_time_delta_min, matched_amount_ratio, net_flow_ratio, rapid_turnover_score, credit_debit_symmetry, max_single_day_volume
   GRAPH_NETWORK (10): in_degree, out_degree, fan_in_ratio, fan_out_ratio, betweenness_centrality, pagerank, community_id, community_mule_density, clustering_coefficient, total_counterparties
   PROFILE_MISMATCH (5): txn_volume_vs_income, account_age_vs_activity, avg_txn_vs_balance, product_txn_mismatch, balance_volatility
   KYC_BEHAVIORAL (4): mobile_change_flag, activity_change_post_mobile, kyc_completeness, linked_account_count
   INTERACTIONS (5): dormancy_x_burst, round_x_structuring, fanin_x_passthrough_speed, new_account_x_high_value, velocity_x_centrality

   Helper functions:
   - get_features_by_group(group: str) -> list[str]
   - get_all_feature_names() -> list[str]
   - get_high_power_features() -> list[str]
   - print_feature_summary() -> None

VALIDATION:
- python -c 'from src.features.registry import FEATURE_REGISTRY, get_all_feature_names; assert len(FEATURE_REGISTRY) == 57, f\"Expected 57, got {len(FEATURE_REGISTRY)}\"; print(f\"Registry OK: {len(FEATURE_REGISTRY)} features\")'
- python -c 'from src.features.base import BaseFeatureGenerator; print(\"Base class OK\")'
Print 'TASK 1.6 COMPLETE — Feature base and registry verified' when all pass.
"
```

---

### TASK 1.7 — Test Infrastructure (conftest + fixtures)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Data pipeline and feature base exist. Need shared test fixtures.

Create comprehensive test infrastructure.

1. Create tests/conftest.py with these fixtures:

   @pytest.fixture
   def sample_transactions():
       np.random.seed(42)
       n = 1000
       accounts = np.random.choice(['ACC_001', 'ACC_002', 'ACC_003', 'ACC_004', 'ACC_005'], n)
       dates = pd.date_range('2024-01-01', periods=n, freq='h')
       return pd.DataFrame({
           'account_id': accounts,
           'transaction_id': [f'TXN_{i}' for i in range(n)],
           'transaction_date': dates,
           'transaction_amount': np.random.exponential(5000, n).astype('float32'),
           'is_credit': np.random.randint(0, 2, n).astype('int8'),
           'counterparty_id': np.random.choice(['CP_A', 'CP_B', 'CP_C', 'CP_D', 'CP_E'], n),
           'transaction_type': pd.Categorical(np.random.choice(['NEFT', 'UPI', 'IMPS'], n)),
           'channel': pd.Categorical(np.random.choice(['branch', 'online', 'mobile'], n)),
           'balance_after': np.random.uniform(10000, 500000, n).astype('float32'),
           'transaction_hour': np.random.randint(0, 24, n),
           'transaction_dow': np.random.randint(0, 7, n),
           'is_weekend': np.zeros(n, dtype='int8'),
           'is_night': np.zeros(n, dtype='int8'),
       })

   @pytest.fixture
   def sample_profile():
       5 accounts with: account_id, customer_id, account_type, account_age_days, declared_income, current_balance, is_new_account, dataset columns

   @pytest.fixture
   def sample_labels():
       3 labeled accounts: ACC_001 (mule=1), ACC_002 (mule=0), ACC_003 (mule=1)

   @pytest.fixture
   def sample_features():
       100 accounts × 57 random features (seeded), indexed by account_id

2. Add pytest markers to pyproject.toml if not already:
   [tool.pytest.ini_options]
   markers = ['unit', 'integration']

3. Run: python -m pytest tests/ --collect-only to verify setup

VALIDATION:
- python -m pytest tests/ --collect-only 2>&1 | tail -5 — should show tests collected with no errors
- python -c 'import tests.conftest; print(\"conftest imports OK\")'
Print 'TASK 1.7 COMPLETE — Test infrastructure verified' when all pass.
"
```

---

### TASK 1.8 — EDA Notebook

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Data loader, validator, merger all exist. Need exploratory data analysis notebook.

Create a comprehensive EDA notebook using nbformat (programmatic .ipynb creation).

Create notebooks/01_eda_data_quality.ipynb using nbformat.v4:

Cells to create:
CELL 1 (markdown): '# RBI Mule Account Detection — EDA & Data Quality'
CELL 2 (code): Import pandas, numpy, matplotlib, seaborn, plotly.express, src.data.loader
CELL 3 (code): Load all data using load_all(), print shapes of each table
CELL 4 (markdown): '## Data Quality Report'
CELL 5 (code): Run DataValidator, visualize null percentages as horizontal bar chart
CELL 6 (markdown): '## Label Distribution'
CELL 7 (code): Bar chart of mule vs legitimate counts, print exact percentages
CELL 8 (markdown): '## Transaction Volume Over Time'
CELL 9 (code): Monthly transaction count line chart, split by mule/legit if labels available
CELL 10 (markdown): '## Amount Distribution'
CELL 11 (code): Histogram of amounts (log x-scale), overlay mule (red) vs legit (blue)
CELL 12 (markdown): '## Temporal Patterns'
CELL 13 (code): Heatmap of count by hour_of_day × day_of_week using seaborn
CELL 14 (markdown): '## Key Findings'
CELL 15 (markdown): Bullet point summary placeholder

Save as valid .ipynb using nbformat.write().

VALIDATION:
- python -c \"import nbformat; nb = nbformat.read('notebooks/01_eda_data_quality.ipynb', as_version=4); print(f'EDA notebook: {len(nb.cells)} cells')\" — should show 15 cells
Print 'TASK 1.8 COMPLETE — EDA notebook verified' when all pass.
"
```

---

### TASK 1.9 — Docker + Makefile Polish

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Full data pipeline exists. Need Docker configuration for deployment.

1. Create docker/Dockerfile.api:
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY src/ ./src/
   COPY outputs/models/ ./models/
   COPY outputs/db/ ./db/
   EXPOSE 8000
   HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
   CMD [\"uvicorn\", \"src.api.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]

2. Create docker/Dockerfile.dashboard:
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY frontend/ ./frontend/
   COPY outputs/ ./outputs/
   COPY data/processed/ ./data/processed/
   EXPOSE 8501
   CMD [\"streamlit\", \"run\", \"frontend/app.py\", \"--server.port=8501\", \"--server.address=0.0.0.0\", \"--server.headless=true\"]

3. Create docker-compose.yml:
   Services: api (port 8000), dashboard (port 8501, depends on api)
   Health checks for both. Volume mounts for models and data.
   Environment variables from .env.

4. Verify Makefile has all targets from Task 1.1 and add:
   - docker-build: docker compose build
   - docker-up: docker compose up -d
   - docker-down: docker compose down

VALIDATION:
- test -f docker/Dockerfile.api && echo 'API Dockerfile exists'
- test -f docker/Dockerfile.dashboard && echo 'Dashboard Dockerfile exists'
- test -f docker-compose.yml && echo 'docker-compose exists'
- grep 'docker-build' Makefile && echo 'Makefile has docker target'
Print 'TASK 1.9 COMPLETE — Docker configuration verified' when all pass.
"
```

---

## PHASE 2: Feature Engineering (Tasks 2.1 – 2.8)

---

### TASK 2.1 — **[PARALLEL-T1]** Velocity Features (Group 1: 10 features)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: BaseFeatureGenerator in src/features/base.py with compute(), get_feature_names(), validate_output() methods. Feature registry has all 57 feature definitions.

Create velocity features — Group 1, 10 features for detecting sudden activity changes.

1. Create src/features/velocity.py:

   class VelocityFeatureGenerator(BaseFeatureGenerator):
       def __init__(self):
           super().__init__('velocity', 'velocity')

       def get_feature_names(self) -> list[str]:
           return ['txn_count_1d', 'txn_count_7d', 'txn_count_30d', 'txn_count_90d',
                   'txn_amount_mean_30d', 'txn_amount_max_30d', 'txn_amount_std_30d',
                   'txn_amount_sum_30d', 'velocity_acceleration', 'frequency_change_ratio']

       def compute(self, txn, profile, cutoff_date=None):
           if cutoff_date is None: cutoff_date = pd.Timestamp('2025-06-30')
           txn_valid = txn[txn['transaction_date'] <= cutoff_date].copy()

           all_accounts = profile['account_id'].unique() if profile is not None else txn_valid['account_id'].unique()
           result = pd.DataFrame(index=all_accounts)
           result.index.name = 'account_id'

           For each window in [1, 7, 30, 90] days:
             window_start = cutoff_date - pd.Timedelta(days=window)
             window_txn = txn_valid[txn_valid['transaction_date'] > window_start]
             grp = window_txn.groupby('account_id')
             result[f'txn_count_{window}d'] = grp.size().reindex(all_accounts, fill_value=0)
             If window == 30:
               result['txn_amount_mean_30d'] = grp['transaction_amount'].mean()
               result['txn_amount_max_30d'] = grp['transaction_amount'].max()
               result['txn_amount_std_30d'] = grp['transaction_amount'].std()
               result['txn_amount_sum_30d'] = grp['transaction_amount'].sum()

           result['velocity_acceleration'] = result['txn_count_7d'] / (result['txn_count_30d'] / 4).clip(lower=1)
           result['frequency_change_ratio'] = result['txn_count_30d'] / (result['txn_count_90d'] / 3).clip(lower=1)
           result = result.fillna(0)
           self.validate_output(result)
           return result

   CRITICAL: Use vectorized pandas groupby operations, NOT loops over accounts.

2. Create tests/test_features/test_velocity.py:
   - test_count_windows(): 5 txns in last 7 days → txn_count_7d = 5
   - test_velocity_acceleration(): burst → acceleration > 2.0
   - test_empty_account(): 0 txns → all zeros
   - test_no_future_leakage(): txns after cutoff excluded

VALIDATION:
- python -m pytest tests/test_features/test_velocity.py -v — all tests must pass
Print 'TASK 2.1 COMPLETE — Velocity features verified' when all pass.
"
```

---

### TASK 2.2 — **[PARALLEL-T2]** Amount Pattern Features (Group 2: 8 features)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: BaseFeatureGenerator in src/features/base.py.

Create amount pattern features — Group 2, 8 features for detecting structuring and round-amount abuse.

1. Create src/features/amount_patterns.py:

   class AmountPatternFeatureGenerator(BaseFeatureGenerator):
       Features: round_amount_ratio, structuring_score, structuring_score_broad, amount_entropy, amount_skewness, amount_kurtosis, pct_above_10k, amount_concentration

       compute():
       For each account (groupby):
       - round_amount_ratio: % where amount % 1000 == 0 OR % 5000 == 0 OR % 10000 == 0
       - structuring_score: % of txns with amount in [45000, 49999]
       - structuring_score_broad: % in [40000, 49999]
       - amount_entropy: scipy.stats.entropy on np.histogram(amounts, bins=20)[0] + 1e-10
       - amount_skewness: scipy.stats.skew(amounts) if len >= 3 else 0
       - amount_kurtosis: scipy.stats.kurtosis(amounts) if len >= 3 else 0
       - pct_above_10k: % of txns with amount > 10000
       - amount_concentration: Gini coefficient

       Gini formula: sorted_arr = np.sort(arr); n = len(arr); index = np.arange(1,n+1); return (2*np.sum(index*sorted_arr))/(n*np.sum(sorted_arr)) - (n+1)/n if sum > 0 else 0

       Handle accounts with <2 transactions: return 0 for distribution features.

2. Create tests/test_features/test_amount_patterns.py:
   - test_all_round_amounts(): amounts=[1000,5000,50000] → ratio=1.0
   - test_structuring_detection(): amounts=[48000,49000,49500,1000] → score=0.75
   - test_no_structuring(): amounts=[1000,100000] → score=0.0
   - test_entropy_uniform_high(): random uniform → entropy > 2
   - test_gini_equal(): all same amount → gini ≈ 0

VALIDATION:
- python -m pytest tests/test_features/test_amount_patterns.py -v — all tests must pass
Print 'TASK 2.2 COMPLETE — Amount pattern features verified' when all pass.
"
```

---

### TASK 2.3 — **[PARALLEL-T3]** Temporal Features (Group 3: 8 features)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: BaseFeatureGenerator in src/features/base.py. Preprocessor adds transaction_hour, is_weekend, is_night columns.

Create temporal features — Group 3, 8 features for detecting dormancy, bursts, and unusual timing.

1. Create src/features/temporal.py:

   class TemporalFeatureGenerator(BaseFeatureGenerator):
       Features: dormancy_days, max_gap_days, burst_after_dormancy, unusual_hour_ratio, weekend_ratio, night_weekend_combo, monthly_txn_cv, days_to_first_txn

       compute():
       For each account:
       1. Sort txns by date. Compute gaps = date.diff().dt.days
       2. max_gap_days = gaps.max() or 0
       3. dormancy_days = max_gap_days if > 90, else 0
       4. burst_after_dormancy = 1 if dormancy_days > 0 AND txn_count in last 30 days > 10
       5. unusual_hour_ratio = proportion of txns with hour in [23,0,1,2,3,4,5]
       6. weekend_ratio = proportion of txns on Saturday/Sunday
       7. night_weekend_combo = unusual_hour_ratio * weekend_ratio
       8. monthly_txn_cv: group by month, count per month, std/mean
       9. days_to_first_txn: (first_txn_date - account_open_date).days if available

       Derive is_night and is_weekend from transaction_date if columns don't exist.

2. Create tests/test_features/test_temporal.py:
   - test_dormancy_gap(): 100-day gap → dormancy_days=100
   - test_no_dormancy(): daily txns → dormancy_days=0
   - test_burst_detection(): gap + 20 recent txns → burst=1
   - test_unusual_hours(): all txns at 2AM → ratio=1.0
   - test_weekend_ratio(): 3/10 on weekend → 0.3

VALIDATION:
- python -m pytest tests/test_features/test_temporal.py -v — all tests must pass
Print 'TASK 2.3 COMPLETE — Temporal features verified' when all pass.
"
```

---

### TASK 2.4 — Pass-Through Features (Group 4: 7 features)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: BaseFeatureGenerator in src/features/base.py. This is CRITICAL for detecting mule pattern #3 (rapid pass-through: money in → money out quickly).

Create pass-through features — Group 4, 7 features.

1. Create src/features/passthrough.py:

   class PassThroughFeatureGenerator(BaseFeatureGenerator):
       Features: credit_debit_time_delta_median, credit_debit_time_delta_min, matched_amount_ratio, net_flow_ratio, rapid_turnover_score, credit_debit_symmetry, max_single_day_volume

       compute():
       For each account:
       - Separate credits (is_credit==1) and debits (is_credit==0), sort by date
       - OPTIMIZATION: Use pd.merge_asof for credit→debit matching:
         merge_asof(credits, debits, on='transaction_date', by='account_id', direction='forward')
         This finds the NEXT debit after each credit efficiently
       - Time delta = (debit_date - credit_date).total_seconds() / 3600
       - Matched if abs(credit_amt - debit_amt) / max(credit_amt, 1) < 0.05 AND delta < 24h
       - credit_debit_time_delta_median: median of all time deltas (999 if no pairs)
       - credit_debit_time_delta_min: minimum delta (999 if no pairs)
       - matched_amount_ratio: count(matched) / max(n_credits, 1)
       - net_flow_ratio = sum(credits) / max(sum(debits), 1)
       - rapid_turnover_score = sum(matched_credit_amounts where delta<48h) / max(sum(all_credits), 1)
       - credit_debit_symmetry = 1 - abs(n_credits - n_debits) / max(n_credits + n_debits, 1)
       - max_single_day_volume = txn.groupby(date).amount.sum().max()

2. Create tests/test_features/test_passthrough.py:
   - test_rapid_passthrough(): credit at T, debit at T+30min same amount → delta≈0.5h, matched=1.0
   - test_no_passthrough(): only credits → delta=999, matched=0
   - test_net_flow_balanced(): equal credits/debits → ratio ≈ 1.0
   - test_symmetry_perfect(): 5 credits 5 debits → symmetry=1.0

VALIDATION:
- python -m pytest tests/test_features/test_passthrough.py -v — all tests must pass
Print 'TASK 2.4 COMPLETE — Pass-through features verified' when all pass.
"
```

---

### TASK 2.5 — Graph/Network Features (Group 5: 10 features)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: BaseFeatureGenerator in src/features/base.py. Dependencies: networkx, python-louvain (community detection). This is computationally expensive — design for efficiency.

Create graph/network features — Group 5, 10 features using NetworkX for transaction network analysis.

1. Create src/features/graph_network.py:

   class GraphNetworkFeatureGenerator(BaseFeatureGenerator):
       def __init__(self, labels_df=None):
           super().__init__('graph_network', 'graph_network')
           self.labels_df = labels_df  # For community_mule_density (train only)

       Features: in_degree, out_degree, fan_in_ratio, fan_out_ratio, betweenness_centrality, pagerank, community_id, community_mule_density, clustering_coefficient, total_counterparties

       compute():
       STEP 1: Build directed weighted graph
         - Aggregate edges: txn.groupby(['account_id', 'counterparty_id']).agg(weight=('transaction_amount','sum'), count=('transaction_amount','count'))
         - G = nx.DiGraph(), add edges with weight and count attributes

       STEP 2: Compute centrality metrics
         - in/out degree: G.in_degree(node), G.out_degree(node)
         - fan_in_ratio = in_degree / max(out_degree, 1)
         - fan_out_ratio = out_degree / max(in_degree, 1)
         - pagerank = nx.pagerank(G, weight='weight', max_iter=100)
         - betweenness = nx.betweenness_centrality(G, k=min(500, len(G))) — APPROXIMATE for speed
         - G_undir = G.to_undirected()
         - clustering = nx.clustering(G_undir)

       STEP 3: Community detection
         - from community import community_louvain
         - communities = community_louvain.best_partition(G_undir, weight='weight')
         - total_counterparties = in_degree + out_degree

       STEP 4: Community mule density (TRAIN ONLY — no leakage!)
         - If self.labels_df provided: for each community, compute mean(is_mule) of labeled accounts
         - Map back to each account via community_id
         - If labels_df is None (test/inference): use saved densities or 0

       STEP 5: Accounts not in graph get all zeros

2. Create tests/test_features/test_graph_network.py:
   - test_fan_in(): 5 senders → 1 account → in_degree=5
   - test_fan_out(): 1 account → 5 receivers → out_degree=5
   - test_pagerank_higher_for_hub(): hub has higher PageRank than leaf
   - test_community_detection(): 2 cliques → 2 communities
   - test_missing_account_gets_zeros(): account not in graph → all 0

VALIDATION:
- python -m pytest tests/test_features/test_graph_network.py -v — all tests must pass
Print 'TASK 2.5 COMPLETE — Graph/Network features verified' when all pass.
"
```

---

### TASK 2.6 — Profile Mismatch + KYC Features (Groups 6 & 7: 9 features)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: BaseFeatureGenerator exists. Velocity features provide txn_amount_sum_30d, txn_count_30d, txn_amount_mean_30d. Profile has declared_income, account_age_days, current_balance, account_type columns.

Create profile mismatch (5 features) and KYC behavioral (4 features).

1. Create src/features/profile_mismatch.py:

   class ProfileMismatchFeatureGenerator(BaseFeatureGenerator):
       Features: txn_volume_vs_income, account_age_vs_activity, avg_txn_vs_balance, product_txn_mismatch, balance_volatility

       compute() takes extra kwarg: velocity_features (DataFrame with velocity columns)
       - txn_volume_vs_income = velocity_features['txn_amount_sum_30d'] / max(profile['declared_income'], 1)
       - account_age_vs_activity = velocity_features['txn_count_30d'] / max(profile['account_age_days'], 1)
       - avg_txn_vs_balance = velocity_features['txn_amount_mean_30d'] / max(profile['current_balance'], 1)
       - product_txn_mismatch = 1 if account_type=='savings' AND txn_amount_mean_30d > 50000 else 0
       - balance_volatility = std(daily_balance) / max(mean(daily_balance), 1) from balance_after column

2. Create src/features/kyc_behavioral.py:

   class KYCBehavioralFeatureGenerator(BaseFeatureGenerator):
       Features: mobile_change_flag, activity_change_post_mobile, kyc_completeness, linked_account_count

       - mobile_change_flag: 1 if mobile_change_date not null (adapt to actual columns)
       - activity_change_post_mobile: txn count 30d after vs 30d before mobile change
       - kyc_completeness: count non-null KYC fields / total KYC fields
       - linked_account_count: accounts per customer_id from linkage table

       Handle missing columns gracefully — default to 0.

3. Create tests/test_features/test_profile_mismatch.py:
   Basic tests with synthetic data verifying each feature computes correctly.

VALIDATION:
- python -c 'from src.features.profile_mismatch import ProfileMismatchFeatureGenerator; from src.features.kyc_behavioral import KYCBehavioralFeatureGenerator; print(\"Groups 6+7 OK\")'
- python -m pytest tests/test_features/test_profile_mismatch.py -v — all tests must pass
Print 'TASK 2.6 COMPLETE — Profile and KYC features verified' when all pass.
"
```

---

### TASK 2.7 — Interaction Features (Group 8: 5 features)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: All 7 previous feature groups exist. Interaction features MULTIPLY features from different groups to capture compound signals.

Create interaction features — Group 8, 5 features.

Create src/features/interactions.py:

class InteractionFeatureGenerator(BaseFeatureGenerator):
    Features:
    1. dormancy_x_burst = dormancy_days × txn_count_7d
    2. round_x_structuring = round_amount_ratio × structuring_score
    3. fanin_x_passthrough_speed = fan_in_ratio × (1 / max(credit_debit_time_delta_median, 0.1))
    4. new_account_x_high_value = is_new_account × pct_above_10k
    5. velocity_x_centrality = velocity_acceleration × betweenness_centrality

    def compute_from_features(self, all_features: pd.DataFrame, profile: pd.DataFrame = None) -> pd.DataFrame:
        Takes the COMPLETE feature matrix (all other groups already computed).
        Simply performs column multiplications.
        Returns DataFrame with 5 interaction columns.
        Handle missing columns gracefully (default to 0).

No tests needed for this — it's just multiplication. Will be validated in pipeline test.

VALIDATION:
- python -c 'from src.features.interactions import InteractionFeatureGenerator; g = InteractionFeatureGenerator(); print(f\"Interaction features: {g.get_feature_names()}\"); assert len(g.get_feature_names()) == 5'
Print 'TASK 2.7 COMPLETE — Interaction features verified' when all pass.
"
```

---

### TASK 2.8 — Feature Pipeline Orchestrator

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: All 8 feature groups exist: velocity, amount_patterns, temporal, passthrough, graph_network, profile_mismatch, kyc_behavioral, interactions. Each has a FeatureGenerator class with compute() method.

Create the master pipeline that runs ALL generators in correct dependency order and outputs the final 57-feature matrix.

1. Create src/features/pipeline.py:

   class FeaturePipeline:
       def __init__(self, cutoff_date=None, skip_graph=False):
           self.cutoff = cutoff_date or pd.Timestamp('2025-06-30')
           self.skip_graph = skip_graph

       def run(self, txn, profile, labels=None) -> pd.DataFrame:
           from src.features.velocity import VelocityFeatureGenerator
           from src.features.amount_patterns import AmountPatternFeatureGenerator
           from src.features.temporal import TemporalFeatureGenerator
           from src.features.passthrough import PassThroughFeatureGenerator
           from src.features.graph_network import GraphNetworkFeatureGenerator
           from src.features.profile_mismatch import ProfileMismatchFeatureGenerator
           from src.features.kyc_behavioral import KYCBehavioralFeatureGenerator
           from src.features.interactions import InteractionFeatureGenerator

           # Stage 1: Independent groups (could be parallel)
           velocity = VelocityFeatureGenerator().compute(txn, profile, self.cutoff)
           amount = AmountPatternFeatureGenerator().compute(txn, profile, self.cutoff)
           temporal = TemporalFeatureGenerator().compute(txn, profile, self.cutoff)
           passthrough = PassThroughFeatureGenerator().compute(txn, profile, self.cutoff)

           # Stage 2: Depends on velocity
           profile_mismatch = ProfileMismatchFeatureGenerator().compute(txn, profile, self.cutoff, velocity_features=velocity)
           kyc = KYCBehavioralFeatureGenerator().compute(txn, profile, self.cutoff)

           # Stage 3: Graph (expensive, can skip)
           if not self.skip_graph:
               graph = GraphNetworkFeatureGenerator(labels_df=labels).compute(txn, profile, self.cutoff)
           else:
               graph = pd.DataFrame(0, index=velocity.index, columns=GraphNetworkFeatureGenerator().get_feature_names())

           # Merge all (join on account_id index)
           all_features = velocity.join([amount, temporal, passthrough, profile_mismatch, kyc, graph], how='outer')

           # Stage 4: Interactions (needs complete matrix)
           interactions = InteractionFeatureGenerator().compute_from_features(all_features, profile)
           all_features = all_features.join(interactions, how='left')

           # Clean up
           all_features = all_features.fillna(0).replace([np.inf, -np.inf], 0)
           print(f'Feature matrix: {all_features.shape[0]} accounts × {all_features.shape[1]} features')
           return all_features

       def run_and_save(self, txn, profile, labels, output_path):
           features = self.run(txn, profile, labels)
           Path(output_path).parent.mkdir(parents=True, exist_ok=True)
           features.to_parquet(output_path)
           return features

   if __name__ == '__main__':
       import argparse
       parser = argparse.ArgumentParser()
       parser.add_argument('--skip-graph', action='store_true')
       parser.add_argument('--output', default='data/processed/features_matrix.parquet')
       args = parser.parse_args()
       # Load data, preprocess, run pipeline, save

2. Create tests/test_features/test_pipeline.py:
   Use small synthetic dataset (50 txns, 5 accounts)
   - test_pipeline_runs_without_error(): with skip_graph=True
   - test_pipeline_output_has_57_columns(): verify all feature names present
   - test_pipeline_no_nulls(): no NaN in output
   - test_pipeline_no_inf(): no infinite values

VALIDATION:
- python -c 'from src.features.pipeline import FeaturePipeline; print(\"Pipeline OK\")'
- python -m pytest tests/test_features/test_pipeline.py -v — all tests must pass
- python -m pytest tests/test_features/ -v --tb=short — ALL feature tests must pass
Print 'TASK 2.8 COMPLETE — Feature pipeline verified (57 features)' when all pass.
"
```

---

## PHASE 3: Model Training + Evaluation (Tasks 3.1 – 3.8)

---

### TASK 3.1 — Model Base Class + Training Infrastructure

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Feature pipeline outputs a parquet file with 57 features per account. Labels in train_labels.csv. Dependencies: scikit-learn, xgboost, lightgbm, catboost, torch, optuna.

Create model base class, trainer, evaluator, calibrator, and selector.

1. Create src/models/base.py:
   from abc import ABC, abstractmethod
   import numpy as np, joblib
   from pathlib import Path

   class BaseModelWrapper(ABC):
       def __init__(self, name: str, model_type: str):
           self.name = name
           self.model_type = model_type
           self.model = None
           self.is_fitted = False

       @abstractmethod
       def get_optuna_params(self, trial) -> dict: ...
       @abstractmethod
       def build_model(self, params: dict): ...
       def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs): self.model.fit(X_train, y_train); self.is_fitted = True
       def predict_proba(self, X) -> np.ndarray: return self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X)
       def save(self, path: Path): joblib.dump(self.model, path)
       def load(self, path: Path): self.model = joblib.load(path); self.is_fitted = True
       def get_feature_importance(self) -> dict: return {}

2. Create src/models/trainer.py:
   class ModelTrainer:
       def train_with_optuna(self, model_wrapper, X_train, y_train, X_val, y_val, n_trials=100):
           Create Optuna study (maximize AUC-ROC), each trial gets params, builds model, fits, evaluates. Return best model + params + score.

       def train_all_models(self, X_train, y_train, X_val, y_val, n_trials=50):
           Import all 6 model wrappers, train each, return dict of results.

   Add __main__ block with argparse: --model NAME, --all-models, --optuna-trials N, --predict-test

3. Create src/models/evaluator.py:
   class ModelEvaluator:
       def evaluate(self, y_true, y_prob, threshold=None) -> dict: All metrics
       def compare_models(self, results: dict) -> pd.DataFrame: Comparison table + Wilcoxon tests
       def plot_roc_curves, plot_pr_curves, plot_calibration, plot_confusion_matrices: Save to outputs/plots/

4. Create src/models/calibrator.py:
   class ProbabilityCalibrator: Platt scaling and isotonic regression via CalibratedClassifierCV

5. Create src/models/selector.py:
   class ModelSelector: select_best by metric, statistical_comparison with pairwise Wilcoxon p-values

VALIDATION:
- python -c 'from src.models.base import BaseModelWrapper; from src.models.trainer import ModelTrainer; from src.models.evaluator import ModelEvaluator; from src.models.calibrator import ProbabilityCalibrator; from src.models.selector import ModelSelector; print(\"Model infra OK\")'
Print 'TASK 3.1 COMPLETE — Model infrastructure verified' when all pass.
"
```

---

### TASK 3.2 — **[PARALLEL-T1]** Logistic Regression + Random Forest

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: BaseModelWrapper in src/models/base.py with get_optuna_params, build_model, fit, predict_proba, save, load, get_feature_importance.

Create Logistic Regression and Random Forest model wrappers.

1. Create src/models/logistic.py:
   class LogisticRegressionWrapper(BaseModelWrapper):
   - Uses sklearn Pipeline: StandardScaler + LogisticRegression
   - get_optuna_params: C (0.001-10 log), penalty ('l1','l2'), solver='saga'
   - class_weight='balanced', max_iter=1000
   - get_feature_importance: abs(model.named_steps['classifier'].coef_[0])

2. Create src/models/random_forest.py:
   class RandomForestWrapper(BaseModelWrapper):
   - RandomForestClassifier
   - get_optuna_params: n_estimators(100-1000 step 100), max_depth(5-30), min_samples_split(2-20), min_samples_leaf(1-10), max_features(['sqrt','log2'])
   - class_weight='balanced', n_jobs=-1, random_state=42
   - get_feature_importance: model.feature_importances_

VALIDATION:
- python -c 'from src.models.logistic import LogisticRegressionWrapper; from src.models.random_forest import RandomForestWrapper; print(\"LR+RF OK\")'
Print 'TASK 3.2 COMPLETE — LogReg and RandomForest verified' when all pass.
"
```

---

### TASK 3.3 — **[PARALLEL-T2]** XGBoost + CatBoost

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: BaseModelWrapper in src/models/base.py.

Create XGBoost and CatBoost wrappers.

1. Create src/models/xgboost_model.py:
   class XGBoostWrapper(BaseModelWrapper):
   - xgboost.XGBClassifier
   - get_optuna_params: n_estimators(100-2000), max_depth(3-12), learning_rate(0.01-0.3 log), subsample(0.6-1.0), colsample_bytree(0.6-1.0), min_child_weight(1-10), gamma(0-5), reg_alpha(1e-8 to 10 log), reg_lambda(1e-8 to 10 log)
   - scale_pos_weight = n_neg / n_pos (compute from data)
   - eval_metric='auc', tree_method='hist', random_state=42
   - fit(): use eval_set=[(X_val,y_val)], early_stopping_rounds=50

2. Create src/models/catboost_model.py:
   class CatBoostWrapper(BaseModelWrapper):
   - catboost.CatBoostClassifier
   - get_optuna_params: iterations(100-2000), depth(4-10), learning_rate(0.01-0.3 log), l2_leaf_reg(1-10), bagging_temperature(0-1), random_strength(0-10), border_count(32-255)
   - auto_class_weights='Balanced', eval_metric='AUC', verbose=0, random_seed=42
   - fit(): use eval_set with early stopping

VALIDATION:
- python -c 'from src.models.xgboost_model import XGBoostWrapper; from src.models.catboost_model import CatBoostWrapper; print(\"XGB+CatBoost OK\")'
Print 'TASK 3.3 COMPLETE — XGBoost and CatBoost verified' when all pass.
"
```

---

### TASK 3.4 — **[PARALLEL-T3]** LightGBM + Neural Network

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: BaseModelWrapper in src/models/base.py.

Create LightGBM and PyTorch Neural Network wrappers.

1. Create src/models/lightgbm_model.py:
   class LightGBMWrapper(BaseModelWrapper):
   - lightgbm.LGBMClassifier
   - get_optuna_params: n_estimators(100-2000), max_depth(3-12), learning_rate(0.01-0.3 log), num_leaves(20-300), subsample(0.6-1.0), colsample_bytree(0.6-1.0), min_child_samples(5-100), reg_alpha/lambda(1e-8 to 10 log)
   - is_unbalance=True, metric='auc', verbose=-1
   - fit(): callbacks=[lightgbm.early_stopping(50), lightgbm.log_evaluation(0)]

2. Create src/models/neural_net.py:

   class MuleDetectorNN(nn.Module):
       __init__(input_dim, hidden_dims=[256,128,64,32], dropout=0.3)
       Each layer: Linear → BatchNorm1d → ReLU → Dropout. Final: Linear(prev, 1)

   class FocalLoss(nn.Module):
       __init__(alpha=0.25, gamma=2.0)
       forward: alpha * (1-pt)^gamma * BCE for class imbalance

   class NeuralNetWrapper(BaseModelWrapper):
       get_optuna_params: n_layers(2-5), hidden dims(32-512), lr(1e-5 to 1e-2 log), dropout(0.1-0.5), batch_size([64,128,256,512]), weight_decay(1e-6 to 1e-2 log)
       fit(): StandardScaler → tensors → DataLoader → train 100 epochs → early stopping patience=15 → track val AUC
       predict_proba(): scale → forward → sigmoid → numpy
       save/load: torch.save/load for model, joblib for scaler

VALIDATION:
- python -c 'from src.models.lightgbm_model import LightGBMWrapper; from src.models.neural_net import NeuralNetWrapper; print(\"LGBM+NN OK\")'
Print 'TASK 3.4 COMPLETE — LightGBM and NeuralNet verified' when all pass.
"
```

---

### TASK 3.5 — Model Comparison Notebook

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: All 6 model wrappers exist. Trainer, evaluator, selector exist. Feature matrix available.

Create the full 6-model comparison notebook using nbformat.

Create notebooks/05_model_comparison.ipynb with sections:
1. Setup & Data Loading (features_matrix.parquet + labels)
2. Train/Val Split (stratified 80/20)
3. Train All 6 Models (20 Optuna trials each for notebook speed)
4. Model Comparison Table (AUC-ROC, AUC-PR, F1, Precision, Recall, Brier)
5. Overlaid ROC Curves (plotly, all 6 models color-coded)
6. Overlaid PR Curves (plotly)
7. 2×3 Confusion Matrices Grid (matplotlib)
8. Calibration Plots (top 3 models)
9. Statistical Significance (5-fold CV + Wilcoxon)
10. Feature Importance Top 20 (best model)
11. Learning Curves (best model: 10/25/50/75/100% data)
12. Save Best Model (joblib to outputs/models/best_model.joblib)

Build with nbformat.v4. Each section = markdown + code cell.

VALIDATION:
- python -c \"import nbformat; nb=nbformat.read('notebooks/05_model_comparison.ipynb', as_version=4); print(f'Model notebook: {len(nb.cells)} cells')\" — should show 20+ cells
Print 'TASK 3.5 COMPLETE — Model comparison notebook verified' when all pass.
"
```

---

### TASK 3.6 — **[PARALLEL-T1]** SHAP Explainability + Natural Language

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Trained models exist. SHAP library available. Need global + local explanations for regulatory compliance.

Create SHAP explainability, PDP, and natural language explanation modules.

1. Create src/explainability/shap_explainer.py:
   class MuleExplainer:
       __init__(model, model_type, X_background=None):
       - Tree models → shap.TreeExplainer
       - Logistic → shap.LinearExplainer
       - Neural net → shap.KernelExplainer with background[:100]

       compute_shap_values(X) -> np.ndarray
       explain_global(X, save_dir): beeswarm plot, bar plot, heatmap, save .npy
       explain_local(X_single, account_id): waterfall plot, top features, SHAP values

2. Create src/explainability/pdp.py:
   class PDPAnalyzer: compute_pdp for individual features, plot_top_features in 2×5 grid

3. Create src/explainability/natural_language.py:
   class NaturalLanguageExplainer:
       FEATURE_DESCRIPTIONS dict mapping ALL 57 features to plain English descriptions
       (e.g. 'matched_amount_ratio': 'credits were followed by matching debits within 24 hours')

       explain(shap_values, feature_values, feature_names, top_n=5) -> str:
       Returns 'This account was flagged because:' + top 5 reasons with values

VALIDATION:
- python -c 'from src.explainability.shap_explainer import MuleExplainer; from src.explainability.natural_language import NaturalLanguageExplainer; print(\"Explainability OK\")'
Print 'TASK 3.6 COMPLETE — Explainability modules verified' when all pass.
"
```

---

### TASK 3.7 — **[PARALLEL-T2]** Fairness Audit + Model Card

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Trained models, predictions, and profile data available. Fairlearn library installed.

Create fairness audit and Model Card generator.

1. Create src/explainability/fairness.py:
   class FairnessAuditor:
       SENSITIVE_FEATURES = ['age_group', 'geography_tier', 'account_type']

       prepare_sensitive_features(profile): derive age_group (<25, 25-45, 45-65, >65), geography_tier, keep account_type
       audit(y_true, y_pred, sensitive_df): fairlearn MetricFrame with accuracy, recall, precision, f1, selection_rate. Compute demographic_parity_difference, equalized_odds_difference. Check 80% rule.
       generate_report(results, save_path): JSON report + formatted summary
       mitigate_if_needed(): ThresholdOptimizer for group-specific thresholds if bias found

2. Create src/explainability/model_card.py:
   class ModelCardGenerator:
       generate(model_info, eval_results, fairness_results, save_path):
       Create markdown Model Card (Google framework): Model Details, Intended Use, Performance, Fairness, Limitations, Ethical Considerations
       Save to docs/model_card.md

VALIDATION:
- python -c 'from src.explainability.fairness import FairnessAuditor; from src.explainability.model_card import ModelCardGenerator; print(\"Fairness+ModelCard OK\")'
Print 'TASK 3.7 COMPLETE — Fairness and Model Card verified' when all pass.
"
```

---

### TASK 3.8 — **[PARALLEL-T3]** Suspicious Time Window Detector

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: For the hackathon submission, we need suspicious_start and suspicious_end timestamps per flagged account.

Create suspicious time window detector for temporal IoU scoring.

1. Create src/temporal/window_detector.py:

   class SuspiciousWindowDetector:
       __init__(z_threshold=2.0, min_window_days=7, extend_days=7)

       detect(txn, account_id) -> dict:
       Algorithm:
       1. Filter transactions for this account
       2. Daily aggregates: total_amount, txn_count
       3. Fill missing days with 0
       4. 90-day rolling mean and std of daily volume
       5. Z-score each day: (daily_vol - rolling_mean) / rolling_std
       6. Find contiguous periods where z_score > threshold
       7. Select longest anomalous period
       8. Extend by extend_days each side, clip to actual date range
       Return: {account_id, suspicious_start (ISO), suspicious_end (ISO), peak_z_score, anomalous_days}

       detect_all(txn, account_ids) -> pd.DataFrame: batch detection
       compute_temporal_iou(pred_start, pred_end, true_start, true_end) -> float

2. Create tests/test_temporal/test_window_detector.py:
   - test_detect_clear_anomaly(): steady then spike → window detected
   - test_no_anomaly(): uniform → None
   - test_iou_perfect(): same window → 1.0
   - test_iou_partial(): 50% overlap → correct IoU
   - test_iou_no_overlap(): → 0.0

VALIDATION:
- python -m pytest tests/test_temporal/test_window_detector.py -v — all tests must pass
Print 'TASK 3.8 COMPLETE — Time window detector verified' when all pass.
"
```

---

## PHASE 4: API + Dashboard (Tasks 4.1 – 4.8)

---

### TASK 4.1 — Database + FastAPI Server

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: All ML components built. Need API serving layer with SQLite for caching predictions and explanations.

Create complete FastAPI backend + SQLite database in ONE task.

FILES TO CREATE:
  src/db/models.py — SQLite schema
  src/db/init_db.py — Create tables + indexes
  src/db/crud.py — All CRUD operations
  src/api/schemas.py — Pydantic request/response models
  src/api/dependencies.py — Model loading, DB connections
  src/api/middleware.py — Error handling, logging
  src/api/main.py — FastAPI app
  src/api/routes/predict.py — POST /predict, POST /predict/batch
  src/api/routes/account.py — GET /account/{account_id}
  src/api/routes/model.py — GET /model/info, GET /model/features
  src/api/routes/dashboard.py — GET /dashboard/stats
  src/api/routes/fairness.py — GET /fairness/report
  src/api/routes/benchmark.py — GET /benchmark/results

DATABASE TABLES (SQLite):
  predictions: account_id PK, probability, prediction, threshold, suspicious_start, suspicious_end, model_version, created_at
  features: account_id PK, feature_json TEXT, computed_at
  explanations: account_id PK, shap_values_json, top_features_json, natural_language, created_at
  model_registry: model_id PK, model_type, version, auc_roc, auc_pr, f1_score, threshold, n_features, model_path, is_active, metadata_json
  fairness_reports: report_id PK, model_id FK, sensitive_feature, demographic_parity_diff, equalized_odds_diff, pass_80_rule, details_json, created_at
  benchmark_results: benchmark_id PK, model_type, metrics JSON, created_at

PYDANTIC SCHEMAS:
  PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse, ModelInfoResponse, AccountResponse, FairnessResponse, BenchmarkResponse, HealthResponse

API ENDPOINTS:
  POST /predict — single account prediction + SHAP explanation
  POST /predict/batch — batch prediction
  GET /account/{account_id} — full analysis from DB
  GET /model/info — current model metadata
  GET /model/features — feature list with importance
  GET /dashboard/stats — aggregate statistics
  GET /fairness/report — fairness audit results
  GET /benchmark/results — model comparison
  GET /health — health check

FastAPI app with CORS, startup model loading, error handlers.

ALSO CREATE: tests/test_api/test_routes.py:
  test_health(), test_predict_returns_json(), test_model_info()

VALIDATION:
- python -c 'from src.api.main import app; print(f\"FastAPI app: {len(app.routes)} routes\")'
- python -m pytest tests/test_api/test_routes.py -v
Print 'TASK 4.1 COMPLETE — API server verified' when all pass.
"
```

---

### TASK 4.2 — Streamlit App Structure + Components

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: FastAPI backend exists. Need Streamlit frontend with 8 pages.

Create the Streamlit app structure and reusable components.

1. Create frontend/app.py:
   st.set_page_config(page_title='RBI Mule Detection', page_icon='🔍', layout='wide')
   Load custom CSS. Title + description. Sidebar info.

2. Create frontend/components/charts.py:
   Reusable plotly functions: plot_roc_curves, plot_pr_curves, plot_confusion_matrix, plot_feature_importance, plot_distribution_comparison, plot_timeline, plot_calibration

3. Create frontend/components/tables.py:
   format_model_comparison, format_fairness_table

4. Create frontend/assets/style.css:
   Dark theme, metric card styling

VALIDATION:
- python -c \"import ast; ast.parse(open('frontend/app.py').read()); print('Streamlit app OK')\"
Print 'TASK 4.2 COMPLETE — Streamlit structure verified' when all pass.
"
```

---

### TASK 4.3 — **[PARALLEL-T1]** Dashboard Pages 1+4 (Overview + Explainability)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Streamlit app structure exists. Charts module available. Data in data/processed/, SHAP in outputs/shap_values/.

Create Dashboard Pages 1 (Overview) and 4 (Explainability).

1. Create frontend/pages/1_Overview.py:
   Three st.metric columns: Total Accounts, Total Transactions, Mule Rate
   Class distribution pie chart (plotly)
   Monthly volume line chart
   Amount distribution histogram (log scale, mule vs legit overlay)
   Use @st.cache_data. Handle missing files gracefully.

2. Create frontend/pages/4_Explainability.py:
   Global SHAP beeswarm (load from outputs/shap_values/, display as image)
   SHAP bar chart of mean |SHAP| values
   Per-account explanation: selectbox for account_id → waterfall plot + natural language text
   PDP plots: dropdown for feature

VALIDATION:
- python -c \"import ast; ast.parse(open('frontend/pages/1_Overview.py').read()); ast.parse(open('frontend/pages/4_Explainability.py').read()); print('Pages 1+4 OK')\"
Print 'TASK 4.3 COMPLETE — Overview and Explainability pages verified' when all pass.
"
```

---

### TASK 4.4 — **[PARALLEL-T2]** Dashboard Pages 2+5 (Features + Network)

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)

Create Dashboard Pages 2 (Feature Explorer) and 5 (Network Graph).

1. Create frontend/pages/2_Feature_Explorer.py:
   Group selectbox (All, Velocity, Amount, Temporal, etc.)
   Correlation heatmap (plotly, top 20 features)
   Side-by-side histograms: mule vs legit (dropdown per feature) + KS stat
   Feature importance bar chart

2. Create frontend/pages/5_Network_Graph.py:
   Build subgraph (top 500 accounts by PageRank for performance)
   pyvis Network visualization: red=mule, blue=legit, gray=test
   Edge thickness by volume, community colors
   Sidebar: min weight slider, community filter, mule-only checkbox
   Embed via st.components.v1.html(height=700)

VALIDATION:
- python -c \"import ast; ast.parse(open('frontend/pages/2_Feature_Explorer.py').read()); ast.parse(open('frontend/pages/5_Network_Graph.py').read()); print('Pages 2+5 OK')\"
Print 'TASK 4.4 COMPLETE — Feature Explorer and Network Graph verified' when all pass.
"
```

---

### TASK 4.5 — **[PARALLEL-T3]** Dashboard Pages 3+6+7+8

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)

Create Dashboard Pages 3 (Model Comparison), 6 (Fairness), 7 (Account Inspector), 8 (API Demo).

1. frontend/pages/3_Model_Comparison.py:
   Load from outputs/reports/. Comparison table (highlight best). Overlaid ROC+PR curves (plotly). 2×3 confusion matrices. Calibration. Statistical tests. AUC bar chart with CI error bars.

2. frontend/pages/6_Fairness_Audit.py:
   Selectbox for sensitive feature. Grouped bar chart of metrics by group. Summary table. Recommendations.

3. frontend/pages/7_Account_Inspector.py:
   text_input for account_id. Profile card. Prediction gauge (plotly indicator). Transaction timeline scatter (credits green, debits red, suspicious window red rectangle). SHAP waterfall. Feature table. Amount×hour heatmap.

4. frontend/pages/8_API_Demo.py:
   text_input for account_id. Predict button. Probability + label + top 3 SHAP features. Threshold sensitivity slider.

VALIDATION:
- python -c \"
import ast
for p in ['3_Model_Comparison', '6_Fairness_Audit', '7_Account_Inspector', '8_API_Demo']:
    ast.parse(open(f'frontend/pages/{p}.py').read())
print('Pages 3,6,7,8 OK')
\"
Print 'TASK 4.5 COMPLETE — All remaining dashboard pages verified' when all pass.
"
```

---

## PHASE 5: Documentation + DevOps + Submission (Tasks 5.1 – 5.5)

---

### TASK 5.1 — **[PARALLEL-T1]** Documentation

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)

Create comprehensive documentation.

1. Create README.md (professional, 150+ lines):
   - Title: '🔍 RBI Mule Account Detection System'
   - Key Results section (AUC, feature count — use placeholders like X.XXX)
   - Architecture ASCII diagram
   - Quick Start (git clone, pip install, make all)
   - Feature Engineering summary (8 groups, 57 features)
   - Model Comparison table (placeholder)
   - Dashboard screenshots placeholder
   - Docker instructions
   - Testing: make test
   - Project Structure tree
   - Interview Guide teaser
   - License: MIT

2. Create docs/architecture.md: Pipeline flow, component descriptions, technology rationale
3. Create docs/feature_documentation.md: All 57 features — name, group, computation, rationale, power
4. Create docs/api_reference.md: All endpoints with request/response JSON examples
5. Create docs/interview_prep.md: 10 interview questions with detailed answers (feature engineering, trees vs NN, class imbalance, data leakage, PageRank, SHAP for regulators, fairness, scaling, streaming, time windows)

VALIDATION:
- wc -l README.md — should be 150+ lines
- test -f docs/architecture.md && test -f docs/feature_documentation.md && test -f docs/api_reference.md && test -f docs/interview_prep.md && echo 'All docs exist'
Print 'TASK 5.1 COMPLETE — Documentation verified' when all pass.
"
```

---

### TASK 5.2 — **[PARALLEL-T2]** CI/CD + Test Suite

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)

Create CI/CD pipeline and ensure all tests pass.

1. Create .github/workflows/ci.yml:
   Triggers: push/PR to main. Jobs:
   - lint: ruff check src/ tests/
   - test: pytest tests/ -v --cov=src --cov-report=xml
   - model-validation (only main): check best model AUC > 0.85
   Use ubuntu-latest, python 3.11.

2. Create .github/workflows/model_validation.yml:
   Weekly schedule + manual trigger. Model regression test.

3. Update tests/conftest.py with comprehensive fixtures if needed.

4. RUN: python -m pytest tests/ -v --tb=short
   Fix ANY failures. ALL tests must pass.

VALIDATION:
- python -m pytest tests/ -v --tb=short 2>&1 | tail -10 — should show all tests passed
- test -f .github/workflows/ci.yml && echo 'CI workflow exists'
Print 'TASK 5.2 COMPLETE — CI/CD and tests verified' when all pass.
"
```

---

### TASK 5.3 — **[PARALLEL-T3]** Makefile + Docker Polish

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)

Verify and polish the Makefile and Docker configuration.

1. Verify Makefile has ALL targets:
   setup, data, validate, features, features-fast, train, train-quick, train-baseline,
   evaluate, compare, explain, fairness, model-card, temporal, serve, serve-prod,
   dashboard, db-init, db-seed, submit, test, test-data, test-features, test-api,
   test-cov, lint, format, docker-build, docker-up, docker-down, clean, all

   Add any missing targets. All should be .PHONY.

2. Verify docker-compose.yml is valid.
3. Verify .env.example has all variables.
4. Create a LICENSE file (MIT).

VALIDATION:
- make -n all 2>&1 | head -15 && echo 'Makefile OK'
- test -f LICENSE && echo 'LICENSE exists'
- cat .env.example | wc -l — should be 15+
Print 'TASK 5.3 COMPLETE — Makefile and Docker verified' when all pass.
"
```

---

### TASK 5.4 — Hackathon Submission Notebook

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)
CONTEXT: Best model saved. Feature pipeline ready. Window detector ready. Test accounts: 16,015.

Create the final hackathon submission notebook.

Create notebooks/09_submission.ipynb using nbformat:

Cells:
1. Load best model from outputs/models/best_model.joblib
2. Load test_accounts.csv (16,015 accounts)
3. Load/compute features for test accounts
4. Generate probability predictions
5. Run SuspiciousWindowDetector for flagged accounts (probability > threshold)
6. Format submission: account_id, probability, suspicious_start, suspicious_end
7. Validate: exactly 16,015 rows, probabilities in [0,1], no missing IDs, timestamp format
8. Save to outputs/predictions/submission.csv
9. Print stats + first 10 rows

VALIDATION:
- python -c \"import nbformat; nb=nbformat.read('notebooks/09_submission.ipynb', as_version=4); print(f'Submission notebook: {len(nb.cells)} cells')\"
Print 'TASK 5.4 COMPLETE — Submission notebook verified' when all pass.
"
```

---

### TASK 5.5 — Final Validation Checklist

```bash
cd ~/rbi-mule-detection && claude --permission-mode bypass "
PROJECT: RBI Mule Account Detection System
WORKING DIRECTORY: $(pwd)

Run comprehensive validation of the ENTIRE project. Check everything and report status.

1. STRUCTURE CHECK:
   - Count .py files in src/ (should be 30+)
   - Count .py files in tests/ (should be 15+)
   - Verify all __init__.py files exist

2. IMPORTS CHECK:
   - python -c 'from src.data.loader import load_all'
   - python -c 'from src.features.pipeline import FeaturePipeline'
   - python -c 'from src.models.trainer import ModelTrainer'
   - python -c 'from src.explainability.shap_explainer import MuleExplainer'
   - python -c 'from src.temporal.window_detector import SuspiciousWindowDetector'
   - python -c 'from src.api.main import app'

3. FEATURE REGISTRY:
   - python -c 'from src.features.registry import FEATURE_REGISTRY; assert len(FEATURE_REGISTRY) == 57'

4. TEST SUITE:
   - python -m pytest tests/ --collect-only | grep 'test_' | wc -l (should be 30+)
   - python -m pytest tests/ -v --tb=short (all must pass)

5. FRONTEND:
   - Verify all 8 Streamlit pages exist in frontend/pages/
   - python -c 'import ast; [ast.parse(open(f\"frontend/pages/{p}\").read()) for p in __import__(\"os\").listdir(\"frontend/pages\") if p.endswith(\".py\")]'

6. DOCS:
   - wc -l README.md docs/*.md (each should be 50+ lines)

7. DEVOPS:
   - test -f docker-compose.yml
   - test -f Makefile
   - test -f .github/workflows/ci.yml
   - test -f .env.example

Print FINAL REPORT:
✅ or ❌ for each check
Total: X/Y checks passed
If all pass: '🚀 PROJECT COMPLETE — RBI Mule Detection System is production-ready!'
"
```

---

## PARALLEL EXECUTION GUIDE

You can run tasks in parallel across 3 terminal windows:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE │ TERMINAL 1 (Main)     │ TERMINAL 2            │ TERMINAL 3     │
│────────┼───────────────────────┼───────────────────────┼────────────────│
│  P1    │ 1.1→1.2→1.5→1.6→1.8  │ 1.3→1.4              │ 1.7→1.9       │
│────────┼───────────────────────┼───────────────────────┼────────────────│
│  P2    │ 2.1→2.4→2.8          │ 2.2→2.6              │ 2.3→2.5→2.7   │
│────────┼───────────────────────┼───────────────────────┼────────────────│
│  P3    │ 3.1→3.2→3.5→3.6      │ 3.3→3.7              │ 3.4→3.8       │
│────────┼───────────────────────┼───────────────────────┼────────────────│
│  P4    │ 4.1→4.2→4.3          │ 4.4                   │ 4.5           │
│────────┼───────────────────────┼───────────────────────┼────────────────│
│  P5    │ 5.1→5.4              │ 5.2                   │ 5.3           │
│────────┼───────────────────────┼───────────────────────┼────────────────│
│  FINAL │ 5.5                   │ (idle)                │ (idle)         │
└──────────────────────────────────────────────────────────────────────────┘

ARROWS (→) = paste AFTER previous task completes in that terminal
⛔ NEVER start next phase until ALL terminals finish current phase
After each phase: git add -A && git commit -m "Phase X complete"
```

**Total: 38 tasks across 5 phases — estimated 4-5 hours with 3 parallel agents.**
