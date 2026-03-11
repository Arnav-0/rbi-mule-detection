"""SHAP-based global and local explainability for mule detection models."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MuleExplainer:
    def __init__(self, model, model_type: str, X_background=None):
        """
        model: fitted sklearn/xgb/lgbm/catboost/nn model or wrapper
        model_type: 'tree', 'linear', or 'neural_net'
        X_background: background dataset for KernelExplainer
        """
        import shap

        self.model = model
        self.model_type = model_type
        self.explainer = None

        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            background = X_background if X_background is not None else shap.sample(
                pd.DataFrame(np.zeros((100, 1))), 50
            )
            self.explainer = shap.LinearExplainer(model, background)
        else:
            background = X_background[:100] if X_background is not None else None
            def predict_fn(x):
                return model.predict_proba(x) if hasattr(model, "predict_proba") else model(x)
            self.explainer = shap.KernelExplainer(predict_fn, background)

    def compute_shap_values(self, X) -> np.ndarray:
        vals = self.explainer.shap_values(X)
        if isinstance(vals, list):
            return vals[1] if len(vals) == 2 else vals[0]
        return vals

    def explain_global(self, X, save_dir: str = "outputs/plots/shap"):
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("shap or matplotlib not available")
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        shap_values = self.compute_shap_values(X)
        np.save(save_dir / "shap_values.npy", shap_values)

        # Beeswarm
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(save_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Bar
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(save_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Global SHAP plots saved to %s", save_dir)

    def explain_local(self, X_single, account_id: str, save_dir: str = "outputs/plots/shap"):
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("shap or matplotlib not available")
            return {}

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        shap_values = self.compute_shap_values(X_single)

        # Waterfall
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                base_values=self.explainer.expected_value if hasattr(self.explainer, "expected_value") else 0,
                data=X_single.iloc[0] if hasattr(X_single, "iloc") else X_single[0],
                feature_names=list(X_single.columns) if hasattr(X_single, "columns") else None,
            ),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_dir / f"shap_waterfall_{account_id}.png", dpi=150, bbox_inches="tight")
        plt.close()

        vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        feature_names = list(X_single.columns) if hasattr(X_single, "columns") else [f"f{i}" for i in range(len(vals))]
        top_idx = np.argsort(np.abs(vals))[::-1][:10]
        top_features = {feature_names[i]: float(vals[i]) for i in top_idx}

        return {
            "account_id": account_id,
            "top_features": top_features,
            "shap_values": vals.tolist(),
        }
