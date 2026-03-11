"""Model evaluation: metrics, plots, comparisons."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from scipy.stats import wilcoxon

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, plots_dir: str = "outputs/plots"):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, y_true, y_prob, threshold: Optional[float] = None) -> dict:
        if threshold is None:
            # Youden's J optimal threshold
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            j_scores = tpr - fpr
            threshold = float(thresholds[np.argmax(j_scores)])

        y_pred = (y_prob >= threshold).astype(int)
        return {
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "auc_pr": float(average_precision_score(y_true, y_prob)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "brier": float(brier_score_loss(y_true, y_prob)),
            "threshold": threshold,
            "n_pos": int(y_true.sum()),
            "n_neg": int((1 - y_true).sum()),
        }

    def compare_models(self, results: dict) -> pd.DataFrame:
        """Build comparison DataFrame; results[name] = {y_prob, y_true, metrics}."""
        rows = []
        for name, res in results.items():
            metrics = res.get("metrics", {})
            row = {"model": name, **metrics}
            rows.append(row)
        df = pd.DataFrame(rows).set_index("model")

        # Wilcoxon pairwise on AUC-ROC if CV scores available
        model_names = list(results.keys())
        if all("cv_scores" in results[n] for n in model_names):
            pval_matrix = pd.DataFrame(
                np.ones((len(model_names), len(model_names))),
                index=model_names,
                columns=model_names,
            )
            for i, n1 in enumerate(model_names):
                for j, n2 in enumerate(model_names):
                    if i < j:
                        s1 = results[n1]["cv_scores"]
                        s2 = results[n2]["cv_scores"]
                        if len(s1) == len(s2) and not np.allclose(s1, s2):
                            _, pval = wilcoxon(s1, s2)
                            pval_matrix.loc[n1, n2] = pval
                            pval_matrix.loc[n2, n1] = pval
            logger.info("Wilcoxon p-value matrix:\n%s", pval_matrix.round(4))

        return df.sort_values("auc_roc", ascending=False)

    def plot_roc_curves(self, results: dict, save_name: str = "roc_curves.png"):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, res in results.items():
            y_true = res["y_true"]
            y_prob = res["y_prob"]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curves — All Models")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(self.plots_dir / save_name, dpi=150)
        plt.close(fig)

    def plot_pr_curves(self, results: dict, save_name: str = "pr_curves.png"):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, res in results.items():
            y_true = res["y_true"]
            y_prob = res["y_prob"]
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves — All Models")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(self.plots_dir / save_name, dpi=150)
        plt.close(fig)

    def plot_calibration(self, results: dict, top_n: int = 3, save_name: str = "calibration.png"):
        try:
            import matplotlib.pyplot as plt
            from sklearn.calibration import calibration_curve
        except ImportError:
            return

        names = list(results.keys())[:top_n]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        for name in names:
            y_true = results[name]["y_true"]
            y_prob = results[name]["y_prob"]
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            ax.plot(prob_pred, prob_true, marker="o", label=name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plots_dir / save_name, dpi=150)
        plt.close(fig)

    def plot_confusion_matrices(self, results: dict, save_name: str = "confusion_matrices.png"):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        names = list(results.keys())
        n = len(names)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for idx, name in enumerate(names):
            res = results[name]
            y_true = res["y_true"]
            y_prob = res["y_prob"]
            threshold = res.get("threshold", 0.5)
            y_pred = (y_prob >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            ax = axes[idx]
            ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, str(cm[r, c]), ha="center", va="center", color="black")

        for idx in range(len(names), len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
        fig.savefig(self.plots_dir / save_name, dpi=150)
        plt.close(fig)
