<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/CatBoost-AUC_0.925-00C853?style=for-the-badge&logo=catboost&logoColor=white" alt="CatBoost">
  <img src="https://img.shields.io/badge/FastAPI-13_Endpoints-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-8_Pages-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</p>

<h1 align="center">RBI Mule Account Detection System</h1>

<p align="center">
  <strong>A production-grade machine learning system that identifies mule accounts used for money laundering in Indian banking — built for the Reserve Bank Innovation Hub x IIT Delhi Tryst Hackathon.</strong>
</p>

<p align="center">
  <em>Because every flagged mule account is a financial crime stopped before it reaches an innocent victim.</em>
</p>

---

## The Problem We're Solving

Money mules are people who — knowingly or unknowingly — allow their bank accounts to be used as conduits for laundered money. They're the human infrastructure of financial crime: drug money, fraud proceeds, and terrorist financing all flow through mule accounts before disappearing into the legitimate economy.

India's banking system processes **billions of transactions daily**. Manually reviewing accounts is impossible. Traditional rule-based systems generate too many false positives. We need something smarter.

**This system uses machine learning to automatically detect mule accounts** by analyzing transaction patterns, network behavior, and account profiles — achieving a **92.5% AUC-ROC** (validated with 5-fold stratified cross-validation) while maintaining interpretability through SHAP explanations and fairness auditing.

---

## What Makes This Different

This isn't just a model in a notebook. It's a **complete, deployable system**:

- **56 engineered features** derived from 12 known mule behavior patterns (rapid fund cycling, structuring, dormancy-burst, etc.)
- **6 models benchmarked** with Optuna + 5-fold stratified CV — CatBoost wins with 0.925 AUC-ROC
- **Temporal detection** — not just *who* is a mule, but *when* the suspicious activity happened
- **Real-time scoring** via FastAPI with SHAP explanations for every prediction
- **Fairness auditing** built in — we check for bias across demographics before deployment
- **Interactive dashboard** — 8-page Streamlit app with a neon futuristic theme for visual analysis
- **Network analysis** — PageRank, community detection, and interactive graph visualization

---

## Key Results

All metrics are **5-fold stratified cross-validation** averages (mean +/- std) — no data leakage, no lucky splits.

| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.9253 +/- 0.017** |
| **AUC-PR** | **0.4269 +/- 0.026** |
| **F1 Score** | **0.4741 +/- 0.025** |
| **Precision** | **0.4992 +/- 0.077** |
| **Recall** | **0.4791 +/- 0.101** |

> The model achieves strong class separation (AUC-ROC > 0.92) on a highly imbalanced dataset (1.09% mule rate) — critical for a system that affects real people's bank accounts.

### Model Comparison (5-Fold Stratified CV)

| Model | AUC-ROC | AUC-PR | F1 | Notes |
|-------|---------|--------|-----|-------|
| **CatBoost** | **0.9253 +/- 0.017** | **0.4269 +/- 0.026** | **0.4741 +/- 0.025** | **Best performer** |
| LightGBM | 0.9187 +/- 0.008 | 0.4269 +/- 0.120 | 0.4952 +/- 0.107 | Most consistent (lowest std) |
| Random Forest | 0.9181 +/- 0.024 | 0.4719 +/- 0.045 | 0.4983 +/- 0.043 | Solid ensemble |
| XGBoost | 0.8949 +/- 0.022 | 0.5088 +/- 0.057 | 0.5906 +/- 0.047 | Best F1 score |
| Neural Network | 0.8813 +/- 0.020 | 0.3195 +/- 0.089 | 0.4051 +/- 0.123 | PyTorch MLP |
| Logistic Regression | 0.8758 +/- 0.018 | 0.2895 +/- 0.069 | 0.3873 +/- 0.070 | Baseline, interpretable |

---

## Architecture

The system follows a clean pipeline architecture — each stage is independently testable and reproducible:

```
                         RBI Mule Detection System
                         ========================

  DATA LAYER                 FEATURE LAYER               MODEL LAYER
  ──────────                 ─────────────               ───────────
  ┌─────────────┐           ┌─────────────────┐         ┌─────────────────┐
  │  10 CSV     │           │  8 Feature      │         │  6 Models       │
  │  Files      │──Clean──▶ │  Groups         │──Fit──▶ │  + Optuna       │
  │  7.4M txns  │  Merge    │  56 Features    │  Tune   │  CatBoost best  │
  │  40K accts  │  Validate │  Registry-based │         │  0.925 AUC-ROC  │
  └─────────────┘           └─────────────────┘         └────────┬────────┘
                                                                 │
                                    ┌────────────────────────────┤
                                    ▼                            ▼
                         ┌──────────────────┐         ┌──────────────────┐
                         │  EXPLAINABILITY  │         │  TEMPORAL        │
                         │  ──────────────  │         │  ────────        │
                         │  SHAP values     │         │  Z-score anomaly │
                         │  Fairness audit  │         │  Suspicious      │
                         │  NL explanations │         │  window detect   │
                         │  Model cards     │         │  Temporal IoU    │
                         └────────┬─────────┘         └────────┬─────────┘
                                  │                            │
                                  ▼                            ▼
                         ┌──────────────────┐         ┌──────────────────┐
                         │  FASTAPI         │         │  STREAMLIT       │
                         │  ────────        │         │  ──────────      │
                         │  13 REST routes  │         │  8 pages         │
                         │  Real-time SHAP  │         │  Neon theme      │
                         │  SQLite cache    │         │  Interactive viz │
                         └──────────────────┘         └──────────────────┘
```

---

## Feature Engineering

The heart of the system. We engineered **56 features** across **8 groups**, each inspired by real mule account behaviors documented in AML research:

### Feature Groups

| Group | Count | What It Captures | Why It Matters |
|-------|-------|-----------------|----------------|
| **Velocity** | 10 | Transaction counts & amounts over 1d/7d/30d/90d windows, acceleration ratios | Mules show sudden spikes in activity — normal accounts don't go from 2 txns/month to 50 overnight |
| **Amount Patterns** | 8 | Round amounts, structuring scores, amount entropy, skewness | Laundered money often comes in suspiciously round numbers or just below reporting thresholds |
| **Temporal** | 8 | Dormancy periods, burst detection, night/weekend ratios, monthly CV | Mule accounts are often dormant for months, then explode with activity at odd hours |
| **Pass-Through** | 7 | Credit-debit matching, rapid turnover, net flow ratio | The hallmark of a mule: money in, money out, fast — the account is just a pipe |
| **Graph/Network** | 9 | PageRank, betweenness centrality, community detection, fan-in/out | Mules sit at the center of suspicious networks — graph features catch what transaction-level features miss |
| **Profile Mismatch** | 5 | Volume vs balance, account age vs activity, product mismatch score | A 19-year-old student account suddenly processing millions? Something's wrong |
| **KYC Behavioral** | 4 | Mobile change spikes, KYC completeness, linked accounts anomaly | Frequent contact info changes and incomplete KYC are red flags |
| **Interactions** | 5 | Cross-group multiplicative features (dormancy x burst, pass-through x velocity) | The most powerful signals come from combining: a dormant account that suddenly bursts AND shows pass-through behavior is almost certainly a mule |

### Real-Time Feature Computation

The system can compute all 56 features **on-the-fly** for any account — even ones not in the training set. Upload a CSV of transactions and get instant risk scores through the dashboard or API.

---

## The Dashboard

An 8-page interactive Streamlit dashboard with a **neon futuristic dark theme** — glassmorphism cards, gradient text, floating orbs, and the Orbitron font family. It's not just functional; it's designed to make data exploration enjoyable.

### Pages

| # | Page | What It Does |
|---|------|-------------|
| 🏠 | **Home** | Pipeline overview, quick-test predictions (existing accounts + CSV upload), system architecture |
| 📊 | **Overview** | Dataset statistics, class distribution, transaction volume trends, data quality audit |
| 🔬 | **Feature Explorer** | Feature distributions by class, correlation heatmaps, KS tests for discriminative power |
| 🏆 | **Model Comparison** | ROC/PR curves, confusion matrices, threshold analysis, side-by-side model metrics |
| 🧠 | **Explainability** | Global SHAP analysis, per-account waterfall plots, natural language explanations |
| 🕸 | **Network Graph** | Interactive pyvis graph, PageRank visualization, mule cluster detection |
| ⚖ | **Fairness Audit** | Demographic parity, equalized odds, 80% rule compliance, remediation recommendations |
| 🔎 | **Account Inspector** | Deep-dive into any account — risk score, transaction timeline, heatmaps, SHAP breakdown |
| 🚀 | **API Demo** | Live prediction interface, batch scoring, threshold sensitivity analysis |

---

## API Reference

The FastAPI backend serves predictions via REST, with automatic OpenAPI docs at `/docs`.

```
Base URL: http://localhost:8001
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single account prediction with SHAP explanation |
| `POST` | `/predict/batch` | Score multiple accounts at once |
| `GET` | `/account/{id}` | Full account analysis (features + prediction + history) |
| `GET` | `/model/info` | Current model metadata and performance metrics |
| `GET` | `/model/features` | Feature list grouped by category with importance scores |
| `GET` | `/dashboard/stats` | Aggregate statistics for the dashboard |
| `GET` | `/fairness/report` | Fairness audit results across sensitive features |
| `GET` | `/benchmark/results` | All model comparison metrics |
| `GET` | `/health` | Health check with version info |

### Example Request

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"account_id": "ACCT_000077", "threshold": 0.5}'
```

### Example Response

```json
{
  "account_id": "ACCT_000077",
  "probability": 0.0234,
  "label": "LEGITIMATE",
  "model_version": "v1",
  "top_features": [
    {"feature": "pass_through_score", "shap_value": -0.0412},
    {"feature": "velocity_7d_count", "shap_value": -0.0281},
    {"feature": "dormancy_ratio", "shap_value": 0.0156}
  ],
  "natural_language": "This account shows normal transaction patterns with low pass-through behavior..."
}
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- 4GB+ RAM (for model training)
- Data files from the hackathon placed in `data/raw/`

### Installation

```bash
git clone https://github.com/Arnav-0/rbi-mule-detection.git
cd rbi-mule-detection
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
make all
```

This runs everything: data validation → feature engineering → model training → evaluation → explainability → temporal detection → submission generation.

### Or Step by Step

```bash
# 1. Validate raw data files
make validate

# 2. Engineer 56 features for all 40K accounts
make features

# 3. Train 6 models with Optuna hyperparameter tuning
make train

# 4. Evaluate and compare all models
make evaluate
make compare

# 5. Generate SHAP explanations and fairness reports
make explain
make fairness

# 6. Detect suspicious time windows
make temporal

# 7. Launch the API and dashboard
make serve        # FastAPI on port 8001
make dashboard    # Streamlit on port 8501
```

### Docker

```bash
docker-compose up       # Start API + Dashboard
docker-compose down     # Stop all services
```

---

## Data

The system processes **10 CSV files** from the hackathon dataset:

| File | Records | Description |
|------|---------|-------------|
| `customers.csv` | ~39,988 | Customer demographics and KYC info |
| `accounts.csv` | ~40,038 | Account types, opening dates, balances |
| `product_details.csv` | ~39,988 | Banking products held by each customer |
| `linkage_table.csv` | ~40,038 | Customer-to-account mapping |
| `transactions_part_*.csv` (x6) | ~7.4M total | Full transaction history (Jul 2020 - Jun 2025) |
| `train_labels.csv` | ~24,023 | Ground truth labels (263 mules, 1.09% mule rate) |
| `test_accounts.csv` | ~16,015 | Accounts to predict |

### Submission Format

```csv
account_id,is_mule,suspicious_start,suspicious_end
ACCT_000003,0.87,2023-11-15T09:30:00,2024-02-20T16:45:00
ACCT_000077,0.02,,
```

- `is_mule`: Probability score [0, 1] — primary metric is **AUC-ROC**
- `suspicious_start/end`: ISO timestamps for the detected mule activity window — bonus metric is **Temporal IoU**

---

## Project Structure

```
rbi-mule-detection/
│
├── src/                          # Core source code
│   ├── data/                     # Data pipeline
│   │   ├── loader.py             #   Load & parse CSVs
│   │   ├── merger.py             #   Merge all tables
│   │   ├── preprocessor.py       #   Clean & transform
│   │   ├── splitter.py           #   Train/test split
│   │   └── validator.py          #   Schema validation
│   │
│   ├── features/                 # Feature engineering
│   │   ├── registry.py           #   Central feature registry
│   │   ├── pipeline.py           #   Feature computation pipeline
│   │   ├── realtime.py           #   Real-time feature computation
│   │   ├── velocity.py           #   Velocity features (10)
│   │   ├── amount_patterns.py    #   Amount pattern features (8)
│   │   ├── temporal.py           #   Temporal features (8)
│   │   ├── passthrough.py        #   Pass-through features (7)
│   │   ├── graph_network.py      #   Network/graph features (9)
│   │   ├── profile_mismatch.py   #   Profile mismatch features (5)
│   │   ├── kyc_behavioral.py     #   KYC behavioral features (4)
│   │   └── interactions.py       #   Interaction features (5)
│   │
│   ├── models/                   # Model training & evaluation
│   │   ├── trainer.py            #   Unified training loop
│   │   ├── evaluator.py          #   Metrics & comparison
│   │   ├── logistic.py           #   Logistic Regression
│   │   ├── random_forest.py      #   Random Forest
│   │   ├── xgboost_model.py      #   XGBoost
│   │   ├── lightgbm_model.py     #   LightGBM
│   │   ├── catboost_model.py     #   CatBoost
│   │   └── neural_net.py         #   PyTorch MLP
│   │
│   ├── explainability/           # Model interpretability
│   │   ├── shap_explainer.py     #   SHAP value computation
│   │   ├── fairness.py           #   Fairness auditing
│   │   ├── pdp.py                #   Partial dependence plots
│   │   ├── model_card.py         #   Model card generation
│   │   └── nl_explanations.py    #   Natural language explanations
│   │
│   ├── temporal/                 # Temporal analysis
│   │   └── window_detector.py    #   Suspicious window detection
│   │
│   ├── api/                      # FastAPI backend
│   │   ├── main.py               #   App setup & lifespan
│   │   ├── schemas.py            #   Pydantic models
│   │   ├── dependencies.py       #   Shared dependencies
│   │   ├── middleware.py          #   CORS & logging
│   │   └── routes/               #   6 route modules
│   │
│   ├── db/                       # Database layer
│   │   ├── schema.py             #   SQLite schema (6 tables)
│   │   └── crud.py               #   CRUD operations
│   │
│   └── utils/                    # Shared utilities
│       ├── config.py             #   Configuration
│       └── constants.py          #   Constants & paths
│
├── frontend/                     # Streamlit dashboard
│   ├── app.py                    #   Main entry point
│   ├── assets/
│   │   └── style.css             #   Neon futuristic theme (900+ lines)
│   ├── components/
│   │   ├── charts.py             #   Plotly chart builders
│   │   ├── layout.py             #   Layout components
│   │   └── tables.py             #   Table formatters
│   └── pages/                    #   8 dashboard pages
│       ├── 1_Overview.py
│       ├── 2_Feature_Explorer.py
│       ├── 3_Model_Comparison.py
│       ├── 4_Explainability.py
│       ├── 5_Network_Graph.py
│       ├── 6_Fairness_Audit.py
│       ├── 7_Account_Inspector.py
│       └── 8_API_Demo.py
│
├── tests/                        # Test suite
│   ├── test_api/                 #   API endpoint tests
│   ├── test_data/                #   Data pipeline tests
│   ├── test_features/            #   Feature computation tests
│   ├── test_models/              #   Model training tests
│   └── test_temporal/            #   Temporal detection tests
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda_data_quality.ipynb #   Exploratory data analysis
│   ├── 05_model_comparison.ipynb #   Model benchmarking
│   └── 09_submission.ipynb       #   Submission generation
│
├── docs/                         # Documentation
│   ├── architecture.md           #   System architecture
│   ├── feature_documentation.md  #   Feature descriptions
│   ├── api_reference.md          #   API documentation
│   └── interview_prep.md         #   Technical deep-dive notes
│
├── docker/                       # Docker configuration
│   ├── Dockerfile.api            #   API container
│   ├── Dockerfile.dashboard      #   Dashboard container
│   └── otel/                     #   OpenTelemetry config
│
├── scripts/                      # Utility scripts
│   └── generate_submission.py    #   Submission file generator
│
├── Makefile                      # All pipeline commands
├── docker-compose.yml            # Container orchestration
├── pyproject.toml                # Project metadata
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # You are here
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **ML Models** | LightGBM, XGBoost, CatBoost, scikit-learn, PyTorch |
| **Hyperparameter Tuning** | Optuna (50 trials per model) + 5-fold stratified CV |
| **Explainability** | SHAP, Fairlearn |
| **Feature Engineering** | pandas, NumPy, NetworkX |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly, pyvis |
| **Database** | SQLite |
| **Containerization** | Docker, Docker Compose |
| **Testing** | pytest |
| **Code Quality** | ruff, black |

---

## Testing

```bash
make test              # Run all tests
make lint              # Code quality checks with ruff
make format            # Auto-format with black
```

The test suite covers:
- Data pipeline integrity (schema validation, merge logic)
- Feature computation correctness (all 8 groups)
- Model training and prediction contracts
- API endpoint responses and error handling
- Temporal window detection accuracy

---

## How It Works (The Short Version)

1. **Load** 10 CSV files (customers, accounts, transactions, products, linkage)
2. **Merge** everything into a unified view per account
3. **Engineer** 56 features that capture mule behavior patterns
4. **Train** 6 models with Optuna + 5-fold stratified CV, pick the best (CatBoost)
5. **Explain** every prediction with SHAP values
6. **Audit** for fairness across demographic groups
7. **Detect** suspicious time windows using z-score anomaly detection
8. **Serve** predictions via REST API with natural language explanations
9. **Visualize** everything in an interactive dashboard

---

## Acknowledgments

- **Reserve Bank Innovation Hub (RBIH)** — for organizing the hackathon and providing the dataset
- **IIT Delhi Tryst** — for hosting the competition
- **The open-source community** — LightGBM, SHAP, FastAPI, Streamlit, and the dozens of libraries that made this possible

---

<p align="center">
  <strong>Built with purpose. Every mule account detected is a step toward a safer financial system.</strong>
</p>

<p align="center">
  <sub>Made by <a href="https://github.com/Arnav-0">Arnav</a></sub>
</p>
