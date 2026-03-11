import os
from pathlib import Path

RANDOM_SEED = 42

DATA_RAW_DIR = Path(os.environ.get("DATA_RAW_DIR", "data/raw"))
DATA_PROCESSED_DIR = Path(os.environ.get("DATA_PROCESSED_DIR", "data/processed"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "outputs"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "outputs/models/best_model.joblib"))
DB_PATH = Path(os.environ.get("DB_PATH", "outputs/db/mule_detection.db"))

for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUTS_DIR,
             OUTPUTS_DIR / "models", OUTPUTS_DIR / "predictions",
             OUTPUTS_DIR / "reports", OUTPUTS_DIR / "plots",
             OUTPUTS_DIR / "shap_values", OUTPUTS_DIR / "db",
             OUTPUTS_DIR / "logs"]:
    _dir.mkdir(parents=True, exist_ok=True)
