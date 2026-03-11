import lightgbm
from lightgbm import LGBMClassifier

from src.models.base import BaseModelWrapper


class LightGBMWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__('lightgbm', 'boosting')

    def get_optuna_params(self, trial) -> dict:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

    def build_model(self, params: dict):
        self.model = LGBMClassifier(
            **params,
            is_unbalance=True,
            metric='auc',
            verbose=-1,
            random_state=42,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['callbacks'] = [
                lightgbm.early_stopping(50, verbose=False),
                lightgbm.log_evaluation(0),
            ]
        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

    def get_feature_importance(self) -> dict:
        return dict(enumerate(self.model.feature_importances_))
