"""Model training with Optuna hyperparameter optimization."""

import argparse
import logging
from typing import Dict, Any, Optional

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from src.models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains models using Optuna hyperparameter tuning."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_with_optuna(
        self,
        model_wrapper: BaseModelWrapper,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
    ) -> Dict[str, Any]:
        """Train a single model with Optuna hyperparameter optimization.

        Returns dict with keys: model, best_params, best_score, study.
        """
        best_model = None
        best_score = -1.0
        best_params = {}

        def objective(trial):
            nonlocal best_model, best_score, best_params

            params = model_wrapper.get_optuna_params(trial)
            model_wrapper.build_model(params)
            model_wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            y_prob = model_wrapper.predict_proba(X_val)
            score = roc_auc_score(y_val, y_prob)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_model = model_wrapper.model

            return score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Rebuild the best model so the wrapper holds it
        model_wrapper.build_model(best_params)
        model_wrapper.model = best_model
        model_wrapper.is_fitted = True

        logger.info(
            "Model %s best AUC-ROC: %.4f", model_wrapper.name, best_score
        )

        return {
            "model": model_wrapper,
            "best_params": best_params,
            "best_score": best_score,
            "study": study,
        }

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """Train all registered model wrappers and return results dict."""
        from src.models.logistic import LogisticRegressionWrapper
        from src.models.random_forest import RandomForestWrapper
        from src.models.xgboost_model import XGBoostWrapper
        from src.models.catboost_model import CatBoostWrapper
        from src.models.lightgbm_model import LightGBMWrapper
        from src.models.neural_net import NeuralNetWrapper

        wrappers = [
            LogisticRegressionWrapper(),
            RandomForestWrapper(),
            XGBoostWrapper(),
            CatBoostWrapper(),
            LightGBMWrapper(),
            NeuralNetWrapper(),
        ]

        results: Dict[str, Dict[str, Any]] = {}
        for wrapper in wrappers:
            logger.info("Training %s ...", wrapper.name)
            try:
                result = self.train_with_optuna(
                    wrapper, X_train, y_train, X_val, y_val, n_trials=n_trials
                )
                results[wrapper.name] = result
            except Exception as exc:
                logger.error("Failed to train %s: %s", wrapper.name, exc)
                results[wrapper.name] = {"error": str(exc)}

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mule-detection models")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Name of a single model to train (e.g. xgboost)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Train all registered models",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model",
    )
    parser.add_argument(
        "--predict-test",
        action="store_true",
        help="Generate predictions on the test set after training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import pandas as pd
    import joblib
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from src.models.evaluator import ModelEvaluator

    # Load data
    features = pd.read_parquet("data/processed/features_matrix.parquet")
    labels = pd.read_csv("data/raw/train_labels.csv").set_index("account_id")["is_mule"]
    test_ids = pd.read_csv("data/raw/test_accounts.csv")["account_id"].values

    train_features = features.loc[features.index.intersection(labels.index)]
    y = labels.reindex(train_features.index)

    X_train, X_val, y_train, y_val = train_test_split(
        train_features.values, y.values,
        test_size=0.2, stratify=y.values, random_state=42,
    )
    feature_names = list(train_features.columns)

    logger.info("Train: %d, Val: %d, Features: %d", len(X_train), len(X_val), len(feature_names))

    trainer = ModelTrainer()

    if args.all_models:
        results = trainer.train_all_models(X_train, y_train, X_val, y_val, n_trials=args.optuna_trials)
    elif args.model:
        from src.models.logistic import LogisticRegressionWrapper
        from src.models.random_forest import RandomForestWrapper
        from src.models.xgboost_model import XGBoostWrapper
        from src.models.catboost_model import CatBoostWrapper
        from src.models.lightgbm_model import LightGBMWrapper
        from src.models.neural_net import NeuralNetWrapper

        model_map = {
            "logistic": LogisticRegressionWrapper,
            "random_forest": RandomForestWrapper,
            "xgboost": XGBoostWrapper,
            "catboost": CatBoostWrapper,
            "lightgbm": LightGBMWrapper,
            "neural_net": NeuralNetWrapper,
        }
        wrapper = model_map[args.model]()
        result = trainer.train_with_optuna(wrapper, X_train, y_train, X_val, y_val, n_trials=args.optuna_trials)
        results = {args.model: result}
    else:
        parser.error("Specify --model NAME or --all-models")

    # Evaluate and save
    evaluator = ModelEvaluator()
    best_name, best_score = None, -1
    for name, res in results.items():
        if "error" in res:
            logger.error("Skipping %s: %s", name, res["error"])
            continue
        y_prob = res["model"].predict_proba(X_val)
        metrics = evaluator.evaluate(y_val, y_prob)
        logger.info("%s — AUC-ROC: %.4f, AUC-PR: %.4f, F1: %.4f",
                     name, metrics["auc_roc"], metrics["auc_pr"], metrics["f1"])
        if metrics["auc_roc"] > best_score:
            best_score = metrics["auc_roc"]
            best_name = name

    if best_name:
        best_wrapper = results[best_name]["model"]
        model_dir = Path("outputs/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        best_wrapper.save(model_dir / "best_model.joblib")
        logger.info("Best model: %s (AUC-ROC=%.4f) saved to outputs/models/best_model.joblib", best_name, best_score)

        if args.predict_test:
            test_features = features.loc[features.index.intersection(test_ids)]
            test_probs = best_wrapper.predict_proba(test_features.values)
            submission = pd.DataFrame({
                "account_id": test_features.index,
                "is_mule": test_probs,
            })
            submission.to_csv("outputs/predictions/submission.csv", index=False)
            logger.info("Test predictions saved to outputs/predictions/submission.csv")
