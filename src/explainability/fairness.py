"""Fairness audit for mule detection model."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENSITIVE_FEATURES = ["age_group", "geography_tier", "account_type"]


class FairnessAuditor:
    SENSITIVE_FEATURES = SENSITIVE_FEATURES

    def prepare_sensitive_features(self, profile: pd.DataFrame) -> pd.DataFrame:
        """Derive fairness-relevant groupings from profile data."""
        result = pd.DataFrame(index=profile.index)

        if "age" in profile.columns:
            age = profile["age"]
            result["age_group"] = pd.cut(
                age,
                bins=[0, 25, 45, 65, 200],
                labels=["<25", "25-45", "45-65", ">65"],
                right=True,
            ).astype(str)
        elif "age_group" in profile.columns:
            result["age_group"] = profile["age_group"].astype(str)
        else:
            result["age_group"] = "unknown"

        if "geography_tier" in profile.columns:
            result["geography_tier"] = profile["geography_tier"].astype(str)
        elif "city_tier" in profile.columns:
            result["geography_tier"] = profile["city_tier"].astype(str)
        else:
            result["geography_tier"] = "unknown"

        if "account_type" in profile.columns:
            result["account_type"] = profile["account_type"].astype(str)
        else:
            result["account_type"] = "unknown"

        return result

    def audit(self, y_true, y_pred, sensitive_df: pd.DataFrame) -> dict:
        """Run fairness audit using MetricFrame-style analysis.

        Returns per-group metrics and disparity measures.
        """
        try:
            from fairlearn.metrics import (
                MetricFrame,
                demographic_parity_difference,
                equalized_odds_difference,
            )
            from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

            metric_fns = {
                "accuracy": accuracy_score,
                "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
                "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
                "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
                "selection_rate": lambda yt, yp: yp.mean() if hasattr(yp, "mean") else np.mean(yp),
            }

            results = {}
            for sf_name in SENSITIVE_FEATURES:
                if sf_name not in sensitive_df.columns:
                    continue
                sf = sensitive_df[sf_name]
                mf = MetricFrame(
                    metrics=metric_fns,
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sf,
                )
                dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
                eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sf)

                results[sf_name] = {
                    "overall": mf.overall.to_dict(),
                    "by_group": mf.by_group.to_dict(),
                    "demographic_parity_difference": float(dp_diff),
                    "equalized_odds_difference": float(eo_diff),
                    "passes_80pct_rule": self._check_80pct_rule(y_pred, sf),
                }
            return results

        except ImportError:
            logger.warning("fairlearn not installed; running manual fairness audit")
            return self._manual_audit(y_true, y_pred, sensitive_df)

    def _manual_audit(self, y_true, y_pred, sensitive_df: pd.DataFrame) -> dict:
        from sklearn.metrics import recall_score, precision_score

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        results = {}

        for sf_name in SENSITIVE_FEATURES:
            if sf_name not in sensitive_df.columns:
                continue
            sf = sensitive_df[sf_name].values
            groups = np.unique(sf)
            by_group = {}
            selection_rates = []

            for g in groups:
                mask = sf == g
                if mask.sum() == 0:
                    continue
                yt_g = y_true[mask]
                yp_g = y_pred[mask]
                sel_rate = yp_g.mean()
                selection_rates.append(sel_rate)
                by_group[str(g)] = {
                    "n": int(mask.sum()),
                    "selection_rate": float(sel_rate),
                    "recall": float(recall_score(yt_g, yp_g, zero_division=0)),
                    "precision": float(precision_score(yt_g, yp_g, zero_division=0)),
                }

            dp_diff = max(selection_rates) - min(selection_rates) if selection_rates else 0.0
            results[sf_name] = {
                "by_group": by_group,
                "demographic_parity_difference": float(dp_diff),
                "passes_80pct_rule": self._check_80pct_rule(y_pred, sensitive_df[sf_name].values),
            }
        return results

    def _check_80pct_rule(self, y_pred, sensitive_feature) -> bool:
        """Check if all groups meet the 80% rule (adverse impact ratio >= 0.8)."""
        y_pred = np.asarray(y_pred)
        groups = np.unique(sensitive_feature)
        rates = []
        for g in groups:
            mask = sensitive_feature == g
            if mask.sum() == 0:
                continue
            rates.append(y_pred[mask].mean())
        if not rates:
            return True
        return min(rates) / max(rates) >= 0.8

    def generate_report(self, results: dict, save_path: str = "outputs/fairness_report.json") -> str:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        lines = ["=" * 60, "FAIRNESS AUDIT REPORT", "=" * 60]
        for sf_name, res in results.items():
            lines.append(f"\nSensitive Feature: {sf_name}")
            lines.append(f"  Demographic Parity Difference: {res.get('demographic_parity_difference', 'N/A'):.4f}")
            lines.append(f"  Equalized Odds Difference: {res.get('equalized_odds_difference', 'N/A'):.4f}" if "equalized_odds_difference" in res else "  Equalized Odds: N/A")
            passes = res.get("passes_80pct_rule", False)
            lines.append(f"  Passes 80% Rule: {'YES' if passes else 'NO (BIAS DETECTED)'}")
        lines.append("=" * 60)

        summary = "\n".join(lines)
        logger.info(summary)
        return summary

    def mitigate_if_needed(self, model, X_train, y_train, sensitive_train, audit_results: dict):
        """Apply ThresholdOptimizer if bias detected in any group."""
        bias_detected = any(
            not res.get("passes_80pct_rule", True)
            for res in audit_results.values()
        )
        if not bias_detected:
            logger.info("No bias detected; mitigation not needed.")
            return model

        try:
            from fairlearn.postprocessing import ThresholdOptimizer

            mitigated = ThresholdOptimizer(
                estimator=model,
                constraints="demographic_parity",
                objective="balanced_accuracy_score",
                predict_method="predict_proba",
            )
            mitigated.fit(X_train, y_train, sensitive_features=sensitive_train)
            logger.info("ThresholdOptimizer applied for bias mitigation.")
            return mitigated
        except ImportError:
            logger.warning("fairlearn not available; skipping mitigation.")
            return model
