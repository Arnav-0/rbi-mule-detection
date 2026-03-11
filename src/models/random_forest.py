"""Random Forest wrapper."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModelWrapper


class RandomForestWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__(name="random_forest", model_type="ensemble")

    def get_optuna_params(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    def build_model(self, params: dict):
        self.model = RandomForestClassifier(
            **params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    def get_feature_importance(self) -> dict:
        if self.model is None or not self.is_fitted:
            return {}
        return {"importances": self.model.feature_importances_}
