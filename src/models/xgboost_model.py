import numpy as np
from xgboost import XGBClassifier

from src.models.base import BaseModelWrapper


class XGBoostWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__('xgboost', 'boosting')
        self.scale_pos_weight = 1.0

    def get_optuna_params(self, trial) -> dict:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        }

    def build_model(self, params: dict):
        # Remove keys that we set explicitly to avoid duplicates
        clean = {k: v for k, v in params.items()
                 if k not in ('scale_pos_weight', 'eval_metric', 'tree_method',
                              'random_state', 'use_label_encoder', 'early_stopping_rounds')}
        self.model = XGBClassifier(
            **clean,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric='auc',
            tree_method='hist',
            random_state=42,
            early_stopping_rounds=50,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        self.build_model({})

        fit_params = {'verbose': False}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

    def get_feature_importance(self) -> dict:
        return dict(enumerate(self.model.feature_importances_))
