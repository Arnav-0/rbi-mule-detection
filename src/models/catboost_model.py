"""CatBoost wrapper."""
from __future__ import annotations

import numpy as np

from src.models.base import BaseModelWrapper


class CatBoostWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__(name="catboost", model_type="gradient_boosting")

    def get_optuna_params(self, trial) -> dict:
        return {
            "iterations": trial.suggest_int("iterations", 100, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }

    def build_model(self, params: dict):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(
            **params,
            auto_class_weights="Balanced",
            eval_metric="AUC",
            verbose=0,
            random_seed=42,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            from catboost import Pool
            fit_kwargs["eval_set"] = Pool(X_val, y_val)
            fit_kwargs["early_stopping_rounds"] = 50

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict:
        if self.model is None or not self.is_fitted:
            return {}
        return {"importances": self.model.get_feature_importance()}
