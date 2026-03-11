"""Model training with Optuna hyperparameter optimization and stratified k-fold CV."""

import argparse
import copy
import json
import logging
from typing import Dict, Any, Optional, List

import numpy as np
import optuna
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

from src.models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


def _get_all_wrappers():
    from src.models.logistic import LogisticRegressionWrapper
    from src.models.random_forest import RandomForestWrapper
    from src.models.xgboost_model import XGBoostWrapper
    from src.models.catboost_model import CatBoostWrapper
    from src.models.lightgbm_model import LightGBMWrapper
    from src.models.neural_net import NeuralNetWrapper

    return {
        "logistic": LogisticRegressionWrapper,
        "random_forest": RandomForestWrapper,
        "xgboost": XGBoostWrapper,
        "catboost": CatBoostWrapper,
        "lightgbm": LightGBMWrapper,
        "neural_net": NeuralNetWrapper,
    }


class ModelTrainer:
    """Trains models using Optuna hyperparameter tuning with optional k-fold CV."""

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

    def train_with_cv(
        self,
        wrapper_cls,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Train with stratified k-fold CV. Tunes on each fold, reports mean +/- std.

        Returns dict with cv_metrics, best_params (from best fold), and per-fold details.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        fold_metrics: List[Dict[str, float]] = []
        best_fold_score = -1.0
        best_fold_params = {}
        best_fold_wrapper = None

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            wrapper = wrapper_cls()
            logger.info("  Fold %d/%d — Train: %d, Val: %d",
                        fold_idx + 1, n_folds, len(y_tr), len(y_vl))

            result = self.train_with_optuna(wrapper, X_tr, y_tr, X_vl, y_vl, n_trials=n_trials)
            fold_wrapper = result["model"]

            y_prob = fold_wrapper.predict_proba(X_vl)
            y_pred = (y_prob >= 0.5).astype(int)

            # Find optimal threshold for F1
            prec_vals, rec_vals, thresholds_pr = precision_recall_curve(y_vl, y_prob)
            f1_vals = np.where(
                (prec_vals + rec_vals) > 0,
                2 * prec_vals * rec_vals / (prec_vals + rec_vals),
                0.0,
            )
            best_thr_idx = np.argmax(f1_vals[:-1])
            opt_threshold = float(thresholds_pr[best_thr_idx])
            y_pred_opt = (y_prob >= opt_threshold).astype(int)

            metrics = {
                "auc_roc": roc_auc_score(y_vl, y_prob),
                "auc_pr": average_precision_score(y_vl, y_prob),
                "f1": f1_score(y_vl, y_pred_opt),
                "precision": precision_score(y_vl, y_pred_opt, zero_division=0),
                "recall": recall_score(y_vl, y_pred_opt, zero_division=0),
                "threshold": opt_threshold,
            }
            fold_metrics.append(metrics)

            logger.info("  Fold %d — AUC-ROC: %.4f, AUC-PR: %.4f, F1: %.4f",
                        fold_idx + 1, metrics["auc_roc"], metrics["auc_pr"], metrics["f1"])

            if metrics["auc_roc"] > best_fold_score:
                best_fold_score = metrics["auc_roc"]
                best_fold_params = result["best_params"].copy()
                best_fold_wrapper = fold_wrapper

        # Aggregate CV metrics
        metric_keys = ["auc_roc", "auc_pr", "f1", "precision", "recall"]
        cv_summary = {}
        for key in metric_keys:
            vals = [m[key] for m in fold_metrics]
            cv_summary[f"{key}_mean"] = float(np.mean(vals))
            cv_summary[f"{key}_std"] = float(np.std(vals))

        logger.info(
            "%s CV results — AUC-ROC: %.4f +/- %.4f, AUC-PR: %.4f +/- %.4f, F1: %.4f +/- %.4f",
            wrapper_cls().name,
            cv_summary["auc_roc_mean"], cv_summary["auc_roc_std"],
            cv_summary["auc_pr_mean"], cv_summary["auc_pr_std"],
            cv_summary["f1_mean"], cv_summary["f1_std"],
        )

        # Retrain final model on ALL data with best hyperparams
        logger.info("Retraining %s on full data with best params from CV ...", wrapper_cls().name)
        final_wrapper = wrapper_cls()
        final_wrapper.build_model(best_fold_params)
        # Use 90/10 split for early stopping during final retrain
        from sklearn.model_selection import train_test_split
        X_final_tr, X_final_vl, y_final_tr, y_final_vl = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=self.random_state
        )
        final_wrapper.fit(X_final_tr, y_final_tr, X_val=X_final_vl, y_val=y_final_vl)

        return {
            "model": final_wrapper,
            "best_params": best_fold_params,
            "cv_metrics": cv_summary,
            "fold_metrics": fold_metrics,
            "n_folds": n_folds,
        }

    def train_all_models_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        n_trials: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """Train all models with stratified k-fold CV."""
        model_map = _get_all_wrappers()

        results: Dict[str, Dict[str, Any]] = {}
        for name, wrapper_cls in model_map.items():
            logger.info("=" * 60)
            logger.info("Training %s with %d-fold CV ...", name, n_folds)
            logger.info("=" * 60)
            try:
                result = self.train_with_cv(wrapper_cls, X, y, n_folds=n_folds, n_trials=n_trials)
                results[name] = result
            except Exception as exc:
                logger.error("Failed to train %s: %s", name, exc, exc_info=True)
                results[name] = {"error": str(exc)}

        return results

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """Train all registered model wrappers (single split, no CV)."""
        model_map = _get_all_wrappers()

        results: Dict[str, Dict[str, Any]] = {}
        for name, wrapper_cls in model_map.items():
            wrapper = wrapper_cls()
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
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
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
    from src.models.evaluator import ModelEvaluator

    # Load data
    features = pd.read_parquet("data/processed/features_matrix.parquet")
    labels = pd.read_csv("data/raw/train_labels.csv").set_index("account_id")["is_mule"]
    test_ids = pd.read_csv("data/raw/test_accounts.csv")["account_id"].values

    train_features = features.loc[features.index.intersection(labels.index)]
    y = labels.reindex(train_features.index)
    X = train_features.values
    feature_names = list(train_features.columns)

    logger.info("Data: %d accounts, %d features", len(X), len(feature_names))

    trainer = ModelTrainer()
    model_map = _get_all_wrappers()

    if args.all_models:
        results = trainer.train_all_models_cv(X, y.values, n_folds=args.n_folds, n_trials=args.optuna_trials)
    elif args.model:
        if args.model not in model_map:
            parser.error(f"Unknown model '{args.model}'. Choose from: {list(model_map.keys())}")
        wrapper_cls = model_map[args.model]
        logger.info("Training %s with %d-fold CV ...", args.model, args.n_folds)
        result = trainer.train_with_cv(wrapper_cls, X, y.values, n_folds=args.n_folds, n_trials=args.optuna_trials)
        results = {args.model: result}
    else:
        parser.error("Specify --model NAME or --all-models")

    # Build benchmark results with CV metrics and save models
    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    benchmark = {}
    best_name, best_score = None, -1.0

    for name, res in results.items():
        if "error" in res:
            logger.error("Skipping %s: %s", name, res["error"])
            continue

        wrapper = res["model"]
        cv = res["cv_metrics"]

        # Save individual model
        wrapper.save(model_dir / f"{name}.joblib")

        # Compute final-model predictions on a holdout for curves/CM
        # Use last CV fold's metrics as representative
        fold_m = res["fold_metrics"]

        entry = {
            "model_type": name,
            "n_folds": res["n_folds"],
            "metrics": {
                "auc_roc": cv["auc_roc_mean"],
                "auc_pr": cv["auc_pr_mean"],
                "f1_score": cv["f1_mean"],
                "precision": cv["precision_mean"],
                "recall": cv["recall_mean"],
            },
            "cv_std": {
                "auc_roc": cv["auc_roc_std"],
                "auc_pr": cv["auc_pr_std"],
                "f1_score": cv["f1_std"],
                "precision": cv["precision_std"],
                "recall": cv["recall_std"],
            },
            "fold_metrics": fold_m,
            "params": res["best_params"],
            "n_train": len(X),
            "n_features": len(feature_names),
            "validation": f"{res['n_folds']}-fold stratified CV",
        }
        benchmark[name] = entry

        mean_auc = cv["auc_roc_mean"]
        logger.info(
            "%s — CV AUC-ROC: %.4f +/- %.4f, AUC-PR: %.4f +/- %.4f, F1: %.4f +/- %.4f",
            name, cv["auc_roc_mean"], cv["auc_roc_std"],
            cv["auc_pr_mean"], cv["auc_pr_std"],
            cv["f1_mean"], cv["f1_std"],
        )

        if mean_auc > best_score:
            best_score = mean_auc
            best_name = name

    # Save benchmark
    with open(reports_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, default=str)
    logger.info("Saved benchmark to outputs/reports/benchmark_results.json")

    if best_name:
        best_wrapper = results[best_name]["model"]
        best_wrapper.save(model_dir / "best_model.joblib")
        logger.info("Best model: %s (mean CV AUC-ROC=%.4f) saved to outputs/models/best_model.joblib",
                     best_name, best_score)

        if args.predict_test:
            test_features = features.loc[features.index.intersection(test_ids)]
            test_probs = best_wrapper.predict_proba(test_features.values)
            pred_dir = Path("outputs/predictions")
            pred_dir.mkdir(parents=True, exist_ok=True)
            submission = pd.DataFrame({
                "account_id": test_features.index,
                "is_mule": test_probs,
            })
            submission.to_csv(pred_dir / "submission.csv", index=False)
            logger.info("Test predictions saved to outputs/predictions/submission.csv")
