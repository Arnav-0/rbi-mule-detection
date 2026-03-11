# RBI Mule Account Detection System

Production-grade machine learning system for detecting money mule accounts in banking transaction data, built for the [RBI Innovation Hub](https://rbihub.in/).

> **Money mules** are bank accounts used — knowingly or unknowingly — to launder proceeds of fraud. This system identifies them using 57 engineered features, 6 ML models, graph network analysis, and full explainability.

---

## Highlights

- **57 features** across 8 groups (velocity, amount patterns, temporal, passthrough, graph network, profile mismatch, KYC/behavioral, interactions)
- **6 model architectures** — Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, PyTorch Neural Net
- **Optuna** hyperparameter optimization with Bayesian search
- **SHAP + fairness** explainability with natural-language explanations
- **FastAPI** REST API for real-time scoring
- **Streamlit** dashboard for monitoring and account lookup
- **Docker Compose** for one-command deployment

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip or [uv](https://docs.astral.sh/uv/)

### Installation

```bash
git clone https://github.com/<your-username>/rbi-mule-detection.git
cd rbi-mule-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your settings (WandB key, paths, etc.)
```

### Data Setup

Place the raw CSV files in `data/raw/`:

```
data/raw/
├── customers.csv
├── accounts.csv
├── customer_account_linkage.csv
├── product_details.csv
├── transactions_part_0.csv  ... transactions_part_5.csv
├── train_labels.csv
└── test_accounts.csv
```

See [data/README.md](data/README.md) for full schema documentation.

---

## Pipeline

Run the full pipeline step-by-step or all at once:

```bash
# Full pipeline
make all

# Or step by step:
make data           # Load & validate raw data
make features       # Engineer 57 features (with graph network)
make features-fast  # Skip graph features for speed
make train          # Train all 6 models with Optuna
make train-quick    # Quick train (LightGBM, 20 trials)
make evaluate       # Evaluate & compare models
make explain        # Generate SHAP explanations
make fairness       # Run fairness audit
make temporal       # Detect suspicious temporal windows
make submit         # Generate submission predictions
```

---

## Project Structure

```
├── src/
│   ├── data/              # Loading, merging, preprocessing, splitting, validation
│   ├── features/          # 57 features in 8 generators + pipeline orchestration
│   ├── models/            # 6 model wrappers + trainer, evaluator, calibrator, selector
│   ├── explainability/    # SHAP, fairness (Fairlearn), model cards, NL explanations
│   ├── temporal/          # Suspicious window detection
│   ├── api/               # FastAPI REST API (predict, health)
│   ├── db/                # Database utilities
│   └── utils/             # Config, constants, logging, metrics
├── frontend/              # Streamlit dashboard
├── tests/                 # pytest test suite
├── notebooks/             # Exploratory data analysis
├── docker/                # Dockerfiles for API and dashboard
├── data/                  # Raw and processed data (gitignored)
├── outputs/               # Models, plots, predictions, reports (gitignored)
├── docker-compose.yml     # One-command deployment
├── Makefile               # Pipeline orchestration
├── pyproject.toml         # Project metadata & dependencies
└── requirements.txt       # Pip requirements
```

---

## Feature Groups

| Group | Count | Examples |
|-------|-------|---------|
| **Velocity** | 10 | `txn_count_7d`, `velocity_acceleration`, `frequency_change_ratio` |
| **Amount Patterns** | 8 | `structuring_score`, `round_amount_ratio`, `amount_entropy` |
| **Temporal** | 8 | `dormancy_days`, `burst_after_dormancy`, `unusual_hour_ratio` |
| **Passthrough** | 7 | `matched_amount_ratio`, `rapid_turnover_score`, `credit_debit_symmetry` |
| **Graph Network** | 10 | `pagerank`, `betweenness_centrality`, `community_mule_density` |
| **Profile Mismatch** | 5 | `txn_volume_vs_income`, `product_txn_mismatch` |
| **KYC Behavioral** | 4 | `mobile_change_flag`, `kyc_completeness` |
| **Interactions** | 5 | `dormancy_x_burst`, `fanin_x_passthrough_speed` |

---

## Models

| Model | Type | Key Strengths |
|-------|------|---------------|
| Logistic Regression | Linear | Interpretable baseline, fast inference |
| Random Forest | Ensemble | Robust to noise, feature importance |
| XGBoost | Boosting | High accuracy, handles imbalance |
| LightGBM | Boosting | Fast training, memory efficient |
| CatBoost | Boosting | Handles categorical features natively |
| Neural Network | Deep learning | Captures nonlinear interactions (focal loss) |

All models support Optuna tuning, class-weight balancing, and Platt/isotonic calibration.

---

## API

Start the API server:

```bash
make serve
# or: uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/predict` | Single account prediction |
| POST | `/api/v1/predict/batch` | Batch predictions |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"account_id": "ACC001", "features": {"txn_count_7d": 45, "structuring_score": 0.8, ...}}'
```

---

## Dashboard

```bash
make dashboard
# or: streamlit run frontend/app.py
```

Features: model performance plots, feature importance (SHAP), feature registry browser, account risk lookup.

---

## Docker Deployment

```bash
make docker-build    # Build images
make docker-up       # Start API + dashboard
make docker-down     # Stop services
```

Services:
- **API**: `http://localhost:8000`
- **Dashboard**: `http://localhost:8501`

---

## Testing

```bash
make test            # Run all tests
make test-data       # Data module tests only
make test-features   # Feature module tests only
make lint            # Ruff linter
```

---

## Explainability

Every prediction comes with:

1. **SHAP values** — per-feature contribution to the risk score
2. **Natural language explanations** — human-readable reasons (top-5)
3. **Fairness audit** — demographic parity & equalized odds via Fairlearn
4. **Model card** — standardized documentation following Google's framework
5. **Partial dependence plots** — feature effect visualization

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
