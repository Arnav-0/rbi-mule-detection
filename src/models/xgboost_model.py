"""XGBoost wrapper."""
from __future__ import annotations

import numpy as np

from src.models.base import BaseModelWrapper


class XGBoostWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__(name="xgboost", model_type="gradient_boosting")
        self._scale_pos_weight = 1.0

    def get_optuna_params(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def build_model(self, params: dict):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            **params,
            scale_pos_weight=self._scale_pos_weight,
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        y_arr = np.asarray(y_train)
        n_pos = y_arr.sum()
        n_neg = len(y_arr) - n_pos
        self._scale_pos_weight = n_neg / max(n_pos, 1)
        if self.model is not None:
            self.model.set_params(scale_pos_weight=self._scale_pos_weight)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
            self.model.set_params(early_stopping_rounds=50)

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.is_fitted = True

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict:
        if self.model is None or not self.is_fitted:
            return {}
        return {"importances": self.model.feature_importances_}
