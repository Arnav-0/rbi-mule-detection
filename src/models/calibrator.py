"""Probability calibration via Platt scaling and isotonic regression."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


class ProbabilityCalibrator:
    def __init__(self, method: str = "sigmoid", cv: int = 5):
        """
        method: 'sigmoid' (Platt scaling) or 'isotonic'
        cv: number of cross-validation folds
        """
        if method not in ("sigmoid", "isotonic"):
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        self.method = method
        self.cv = cv
        self.calibrated_model = None

    def fit(self, model_wrapper, X_train, y_train):
        """Wrap the base estimator with CalibratedClassifierCV."""
        self.calibrated_model = CalibratedClassifierCV(
            estimator=model_wrapper.model,
            method=self.method,
            cv=self.cv,
        )
        self.calibrated_model.fit(X_train, y_train)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.calibrated_model is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return self.calibrated_model.predict_proba(X)[:, 1]

    def calibration_error(self, y_true, y_prob, n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE)."""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)
        return float(ece)
