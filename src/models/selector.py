"""Model selection and statistical comparison utilities."""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

from src.models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


class ModelSelector:
    """Select the best model and perform pairwise statistical comparisons."""

    @staticmethod
    def select_best(
        results: Dict[str, Dict[str, Any]],
        metric: str = "auc_roc",
    ) -> str:
        """Return the name of the model with the highest *metric*.

        *results* maps model names to dicts containing at least a
        ``metrics`` sub-dict (output of :class:`ModelEvaluator.evaluate`)
        or a top-level ``best_score`` key when coming straight from the
        trainer.
        """
        best_name: Optional[str] = None
        best_value = -np.inf

        for name, res in results.items():
            # Support both trainer output (best_score) and evaluator output
            if "metrics" in res and metric in res["metrics"]:
                value = res["metrics"][metric]
            elif metric == "auc_roc" and "best_score" in res:
                value = res["best_score"]
            elif metric in res:
                value = res[metric]
            else:
                logger.warning(
                    "Metric '%s' not found for model '%s'; skipping.",
                    metric,
                    name,
                )
                continue

            if value > best_value:
                best_value = value
                best_name = name

        logger.info(
            "Best model by %s: %s (%.4f)", metric, best_name, best_value
        )
        return best_name

    @staticmethod
    def statistical_comparison(
        results: Dict[str, Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> pd.DataFrame:
        """Pairwise Wilcoxon signed-rank tests on cross-validated AUC-ROC.

        Parameters
        ----------
        results : dict
            Maps model names to dicts with a ``model`` key holding a
            :class:`BaseModelWrapper` instance.
        X, y : array-like
            Full dataset used for cross-validation scoring.
        cv : int
            Number of stratified folds.

        Returns
        -------
        pd.DataFrame
            Symmetric matrix of p-values from pairwise Wilcoxon tests.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Collect per-fold AUC-ROC for each model ----------------------------
        model_scores: Dict[str, np.ndarray] = {}
        model_names = [
            name
            for name, res in results.items()
            if "model" in res and isinstance(res["model"], BaseModelWrapper)
        ]

        for name in model_names:
            wrapper: BaseModelWrapper = results[name]["model"]
            fold_scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]

                # Clone and retrain for unbiased CV estimate
                params = results[name].get("best_params", {})
                wrapper.build_model(params)
                wrapper.fit(X_tr, y_tr)
                y_prob = wrapper.predict_proba(X_va)
                fold_scores.append(roc_auc_score(y_va, y_prob))

            model_scores[name] = np.array(fold_scores)
            logger.info(
                "%s CV AUC-ROC: %.4f +/- %.4f",
                name,
                np.mean(fold_scores),
                np.std(fold_scores),
            )

        # Pairwise Wilcoxon tests --------------------------------------------
        n = len(model_names)
        pvalue_matrix = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                scores_i = model_scores[model_names[i]]
                scores_j = model_scores[model_names[j]]
                try:
                    _, pval = wilcoxon(scores_i, scores_j)
                except ValueError:
                    # Identical distributions or too few samples
                    pval = 1.0
                pvalue_matrix[i, j] = pval
                pvalue_matrix[j, i] = pval

        df = pd.DataFrame(
            pvalue_matrix, index=model_names, columns=model_names
        )
        return df
