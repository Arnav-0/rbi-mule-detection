"""Logistic Regression wrapper."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModelWrapper


class LogisticRegressionWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__(name="logistic_regression", model_type="linear")

    def get_optuna_params(self, trial) -> dict:
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            # l1_ratio=0 → pure L2, l1_ratio=1 → pure L1 (sklearn 1.8+ API)
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    def build_model(self, params: dict):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=params["C"],
                l1_ratio=params["l1_ratio"],
                # sklearn 1.8+: don't pass penalty; use l1_ratio to control sparsity
                # l1_ratio=0 → L2, l1_ratio=1 → L1 (saga solver supports both)
                solver="saga",
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )),
        ])

    def get_feature_importance(self) -> dict:
        if self.model is None or not self.is_fitted:
            return {}
        coef = self.model.named_steps["classifier"].coef_[0]
        return {"importances": np.abs(coef)}
