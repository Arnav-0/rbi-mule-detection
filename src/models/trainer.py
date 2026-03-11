"""Model trainer with Optuna hyperparameter optimization."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, output_dir: str = "outputs/models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_with_optuna(
        self,
        model_wrapper,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials: int = 100,
    ) -> dict[str, Any]:
        best_score = -np.inf
        best_params = {}
        best_model = None

        def objective(trial):
            nonlocal best_score, best_params, best_model
            params = model_wrapper.get_optuna_params(trial)
            model_wrapper.build_model(params)
            model_wrapper.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            y_prob = model_wrapper.predict_proba(X_val)
            score = roc_auc_score(y_val, y_prob)
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model_wrapper.model
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        model_wrapper.model = best_model
        model_wrapper.is_fitted = True

        return {
            "model": model_wrapper,
            "best_params": best_params,
            "best_score": best_score,
            "study": study,
        }

    def train_all_models(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials: int = 50,
    ) -> dict[str, dict]:
        from src.models.logistic import LogisticRegressionWrapper
        from src.models.random_forest import RandomForestWrapper
        from src.models.xgboost_model import XGBoostWrapper
        from src.models.catboost_model import CatBoostWrapper
        from src.models.lightgbm_model import LightGBMWrapper
        from src.models.neural_net import NeuralNetWrapper, _TORCH_AVAILABLE

        wrappers = [
            LogisticRegressionWrapper(),
            RandomForestWrapper(),
            XGBoostWrapper(),
            CatBoostWrapper(),
            LightGBMWrapper(),
        ]
        if _TORCH_AVAILABLE:
            wrappers.append(NeuralNetWrapper())
        else:
            logger.warning("PyTorch not installed — skipping NeuralNetWrapper")

        results = {}
        for wrapper in wrappers:
            logger.info("Training %s with %d Optuna trials", wrapper.name, n_trials)
            try:
                result = self.train_with_optuna(
                    wrapper, X_train, y_train, X_val, y_val, n_trials=n_trials
                )
                results[wrapper.name] = result
                save_path = self.output_dir / f"{wrapper.name}.joblib"
                wrapper.save(save_path)
                logger.info("%s val AUC: %.4f", wrapper.name, result["best_score"])
            except Exception as exc:
                logger.error("Failed to train %s: %s", wrapper.name, exc)
        return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Train mule detection models")
    parser.add_argument("--model", type=str, default=None, help="Single model name to train")
    parser.add_argument("--all-models", action="store_true", help="Train all 6 models")
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--predict-test", action="store_true", help="Save test predictions after training")
    parser.add_argument("--features", type=str, default="outputs/features_matrix.parquet")
    parser.add_argument("--labels", type=str, default="data/train_labels.csv")
    return parser.parse_args()


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    args = _parse_args()

    features_path = Path(args.features)
    labels_path = Path(args.labels)

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    X = pd.read_parquet(features_path)
    y = pd.read_csv(labels_path, index_col="account_id").loc[X.index, "is_mule"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    trainer = ModelTrainer()

    if args.all_models:
        results = trainer.train_all_models(X_train, y_train, X_val, y_val, n_trials=args.optuna_trials)
        for name, res in results.items():
            print(f"{name}: val AUC = {res['best_score']:.4f}")
    elif args.model:
        model_map = {
            "logistic": "src.models.logistic.LogisticRegressionWrapper",
            "random_forest": "src.models.random_forest.RandomForestWrapper",
            "xgboost": "src.models.xgboost_model.XGBoostWrapper",
            "catboost": "src.models.catboost_model.CatBoostWrapper",
            "lightgbm": "src.models.lightgbm_model.LightGBMWrapper",
            "neural_net": "src.models.neural_net.NeuralNetWrapper",
        }
        if args.model not in model_map:
            raise ValueError(f"Unknown model: {args.model}. Choose from {list(model_map)}")
        import importlib
        module_path, cls_name = model_map[args.model].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), cls_name)
        wrapper = cls()
        result = trainer.train_with_optuna(
            wrapper, X_train, y_train, X_val, y_val, n_trials=args.optuna_trials
        )
        print(f"{wrapper.name}: val AUC = {result['best_score']:.4f}")

        if args.predict_test:
            probs = wrapper.predict_proba(X_val)
            out = pd.DataFrame({"account_id": X_val.index, "mule_probability": probs})
            out.to_csv(f"outputs/{wrapper.name}_val_preds.csv", index=False)
            print(f"Saved predictions to outputs/{wrapper.name}_val_preds.csv")
    else:
        print("Specify --model NAME or --all-models")
