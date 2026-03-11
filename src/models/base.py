from abc import ABC, abstractmethod
import numpy as np
import joblib
from pathlib import Path


class BaseModelWrapper(ABC):
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def get_optuna_params(self, trial) -> dict:
        ...

    @abstractmethod
    def build_model(self, params: dict):
        ...

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict_proba(self, X) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
        self.is_fitted = True

    def get_feature_importance(self) -> dict:
        return {}
