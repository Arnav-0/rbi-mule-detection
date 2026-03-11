# Architecture

## Pipeline Overview

```
Raw CSVs → Data Pipeline → Feature Pipeline → Model Training → Evaluation → Submission
                                                     ↓
                                              Explainability
                                                     ↓
                                            API + Dashboard
```

## Components

### Data Pipeline (`src/data/`)
- **loader.py**: Loads 6 transaction CSVs + 6 static tables. Renames columns to internal schema (`transaction_timestamp` → `transaction_date`, `amount` → `transaction_amount`). Derives `is_credit` from `txn_type`.
- **preprocessor.py**: Adds derived columns (hour, day-of-week, weekend, night flags, log amount). Builds merged profile from accounts + linkage + customers + products.
- **validator.py**: Validates schema, date ranges, missing values, cross-table consistency.
- **merger.py**: Joins accounts with customer demographics and product holdings.
- **splitter.py**: Stratified train/val/test splits preserving class ratio.

### Feature Pipeline (`src/features/`)
8 generators producing 57 features total, orchestrated by `pipeline.py`:

| Generator | Features | Approach |
|-----------|----------|----------|
| Velocity | 10 | Windowed aggregations (1d/7d/30d/90d) |
| Amount Patterns | 8 | Statistical properties of transaction amounts |
| Temporal | 8 | Time-based patterns, dormancy, burstiness |
| Pass-Through | 7 | Credit-debit matching via merge_asof |
| Graph Network | 10 | NetworkX directed graph, PageRank, Louvain communities |
| Profile Mismatch | 5 | Transaction behavior vs account attributes |
| KYC Behavioral | 4 | Mobile change impact, KYC completeness |
| Interactions | 5 | Cross-group multiplicative features |

### Model Training (`src/models/`)
- **base.py**: Abstract `BaseModelWrapper` with Optuna interface
- 6 model wrappers: Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM, Neural Network
- **trainer.py**: Optuna hyperparameter optimization, trains all models, saves best
- **evaluator.py**: AUC-ROC, AUC-PR, F1, Brier score, confusion matrices
- **calibrator.py**: Platt scaling and isotonic regression
- **selector.py**: Statistical model comparison with Wilcoxon tests

### Explainability (`src/explainability/`)
- **shap_explainer.py**: TreeExplainer / LinearExplainer / KernelExplainer based on model type
- **natural_language.py**: Converts SHAP values to plain English explanations
- **fairness.py**: Fairlearn-based audit across age, geography, account type
- **model_card.py**: Google Model Card format documentation

### Temporal Detection (`src/temporal/`)
- **window_detector.py**: Z-score based anomaly detection on daily transaction volumes. 90-day rolling statistics, contiguous anomalous period identification, configurable extension.

### API (`src/api/`)
- FastAPI with 13 routes across 6 route modules
- SQLite database for caching predictions, features, explanations
- CORS enabled, startup model loading, error middleware

### Dashboard (`frontend/`)
- 8-page Streamlit app: Overview, Feature Explorer, Model Comparison, Explainability, Network Graph, Fairness Audit, Account Inspector, API Demo

## Technology Choices
- **CatBoost/LightGBM**: Best performers for tabular data with categorical features
- **Optuna**: Efficient Bayesian hyperparameter optimization
- **SHAP**: Model-agnostic, regulatory-compliant explanations
- **NetworkX**: Graph features for detecting mule networks
- **FastAPI**: Async-native, auto-documented API
- **Streamlit**: Rapid interactive dashboard development
- **SQLite**: Zero-config persistence for predictions and explanations
