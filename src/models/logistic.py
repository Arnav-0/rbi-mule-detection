from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModelWrapper


class LogisticRegressionWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__('logistic_regression', 'linear')

    def get_optuna_params(self, trial) -> dict:
        return {
            'C': trial.suggest_float('C', 0.001, 10.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'saga',
        }

    def build_model(self, params: dict):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                **params,
                class_weight='balanced',
                max_iter=1000,
            )),
        ])

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict:
        coefs = abs(self.model.named_steps['classifier'].coef_[0])
        return dict(enumerate(coefs))
