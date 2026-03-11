import os
from pathlib import Path

# Directory paths
DATA_RAW_DIR = Path(os.environ.get("DATA_RAW_DIR", "data/raw"))
DATA_PROCESSED_DIR = Path(os.environ.get("DATA_PROCESSED_DIR", "data/processed"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "outputs"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "outputs/models/best_model.joblib"))
DB_PATH = Path(os.environ.get("DB_PATH", "outputs/db/mule_detection.db"))

# API config
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8501"))

# ML config
RANDOM_SEED = 42
OPTUNA_N_TRIALS = int(os.environ.get("OPTUNA_N_TRIALS", "100"))
FEATURE_CUTOFF_DATE = os.environ.get("FEATURE_CUTOFF_DATE", "2025-06-30")

# Domain config
STRUCTURING_THRESHOLD = int(os.environ.get("STRUCTURING_THRESHOLD", "50000"))
DORMANCY_THRESHOLD_DAYS = int(os.environ.get("DORMANCY_THRESHOLD_DAYS", "90"))

# WandB
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rbi-mule-detection")

# Auto-create directories
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUTS_DIR,
             MODEL_PATH.parent, DB_PATH.parent,
             OUTPUTS_DIR / "models", OUTPUTS_DIR / "predictions",
             OUTPUTS_DIR / "reports", OUTPUTS_DIR / "plots",
             OUTPUTS_DIR / "shap_values", OUTPUTS_DIR / "logs"]:
    _dir.mkdir(parents=True, exist_ok=True)
