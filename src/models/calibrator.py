"""Probability calibration for trained classifiers."""

import logging
from typing import Dict

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from src.models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Wrap a fitted classifier with Platt scaling or isotonic regression."""

    def __init__(self):
        self.calibrated_model = None

    def calibrate(
        self,
        model: BaseModelWrapper,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        method: str = "isotonic",
    ) -> CalibratedClassifierCV:
        """Calibrate *model* on the calibration set.

        Parameters
        ----------
        model : BaseModelWrapper
            A fitted model wrapper whose ``.model`` attribute is a
            scikit-learn-compatible estimator.
        X_cal, y_cal : array-like
            Calibration features and labels.
        method : str
            ``'isotonic'`` for isotonic regression or ``'sigmoid'`` for
            Platt scaling.

        Returns
        -------
        CalibratedClassifierCV
            The calibrated classifier (also stored on ``self.calibrated_model``).
        """
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(
                f"method must be 'isotonic' or 'sigmoid', got '{method}'"
            )

        self.calibrated_model = CalibratedClassifierCV(
            estimator=model.model,
            method=method,
            cv="prefit",
        )
        self.calibrated_model.fit(X_cal, y_cal)
        logger.info(
            "Calibrated %s using %s regression.", model.name, method
        )
        return self.calibrated_model

    @staticmethod
    def evaluate_calibration(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute calibration quality metrics.

        Returns
        -------
        dict with ``brier_score`` and ``expected_calibration_error`` (ECE).
        """
        brier = brier_score_loss(y_true, y_prob)

        # Expected Calibration Error (ECE) -----------------------------------
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total = len(y_true)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_prob > lo) & (y_prob <= hi)
            count = mask.sum()
            if count == 0:
                continue
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            ece += (count / total) * abs(avg_true - avg_pred)

        return {
            "brier_score": float(brier),
            "expected_calibration_error": float(ece),
        }
