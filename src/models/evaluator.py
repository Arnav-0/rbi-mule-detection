"""Model evaluation utilities: metrics, comparison tables, and diagnostic plots."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare trained models."""

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive set of binary-classification metrics.

        If *threshold* is ``None`` the optimal threshold is chosen as the
        point on the precision-recall curve that maximises F1.
        """
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, y_prob)

        # Determine threshold ------------------------------------------------
        if threshold is None:
            prec_vals, rec_vals, thresholds_pr = precision_recall_curve(
                y_true, y_prob
            )
            f1_vals = np.where(
                (prec_vals + rec_vals) > 0,
                2 * prec_vals * rec_vals / (prec_vals + rec_vals),
                0.0,
            )
            # precision_recall_curve returns len(thresholds) == len(prec) - 1
            best_idx = np.argmax(f1_vals[:-1])
            threshold = float(thresholds_pr[best_idx])

        y_pred = (y_prob >= threshold).astype(int)

        return {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "brier_score": brier,
            "log_loss": logloss,
            "threshold": threshold,
        }

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    @staticmethod
    def compare_models(results: Dict[str, dict]) -> pd.DataFrame:
        """Build a DataFrame comparing evaluation metrics across models.

        *results* maps model names to dicts that contain at least a
        ``metrics`` key (output of :meth:`evaluate`).
        """
        rows = []
        for name, res in results.items():
            metrics = res.get("metrics", res)
            row = {"model": name}
            row.update(metrics)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("model")
        return df.sort_values("auc_roc", ascending=False)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def plot_roc_curves(
        results: Dict[str, dict], save_dir: Path
    ) -> None:
        """Plot overlaid ROC curves for every model and save to *save_dir*."""
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, res in results.items():
            y_true = res["y_true"]
            y_prob = res["y_prob"]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(save_dir / "roc_curves.png", dpi=150)
        plt.close(fig)
        logger.info("Saved ROC curves to %s", save_dir / "roc_curves.png")

    @staticmethod
    def plot_pr_curves(
        results: Dict[str, dict], save_dir: Path
    ) -> None:
        """Plot overlaid Precision-Recall curves."""
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, res in results.items():
            y_true = res["y_true"]
            y_prob = res["y_prob"]
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(save_dir / "pr_curves.png", dpi=150)
        plt.close(fig)
        logger.info("Saved PR curves to %s", save_dir / "pr_curves.png")

    @staticmethod
    def plot_calibration(
        results: Dict[str, dict], save_dir: Path, n_bins: int = 10
    ) -> None:
        """Plot calibration (reliability) curves."""
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, res in results.items():
            y_true = res["y_true"]
            y_prob = res["y_prob"]
            frac_pos, mean_pred = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
            ax.plot(mean_pred, frac_pos, "s-", label=name)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curves")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(save_dir / "calibration_curves.png", dpi=150)
        plt.close(fig)
        logger.info(
            "Saved calibration curves to %s",
            save_dir / "calibration_curves.png",
        )

    @staticmethod
    def plot_confusion_matrices(
        results: Dict[str, dict], save_dir: Path
    ) -> None:
        """Plot confusion matrices for all models in a grid."""
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        n = len(results)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, (name, res) in enumerate(results.items()):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            y_true = res["y_true"]
            y_prob = res["y_prob"]
            threshold = res.get("threshold", 0.5)
            y_pred = (y_prob >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=12,
                    )
            fig.colorbar(im, ax=ax, fraction=0.046)

        # Hide unused axes
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].set_visible(False)

        fig.tight_layout()
        fig.savefig(save_dir / "confusion_matrices.png", dpi=150)
        plt.close(fig)
        logger.info(
            "Saved confusion matrices to %s",
            save_dir / "confusion_matrices.png",
        )
