"""Partial Dependence Plot analyzer."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PDPAnalyzer:
    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names

    def compute_pdp(self, X, feature: str, grid_resolution: int = 50) -> dict:
        """Compute PDP for a single feature."""
        from sklearn.inspection import partial_dependence

        feature_idx = self.feature_names.index(feature)
        result = partial_dependence(
            self.model, X, features=[feature_idx], grid_resolution=grid_resolution, kind="average"
        )
        return {
            "feature": feature,
            "grid_values": result["grid_values"][0].tolist(),
            "average": result["average"][0].tolist(),
        }

    def plot_top_features(
        self,
        X,
        top_features: list[str],
        save_path: str = "outputs/plots/pdp_top_features.png",
        n_cols: int = 5,
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        n = len(top_features)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()

        for idx, feature in enumerate(top_features):
            try:
                pdp = self.compute_pdp(X, feature)
                axes[idx].plot(pdp["grid_values"], pdp["average"])
                axes[idx].set_title(feature, fontsize=9)
                axes[idx].set_xlabel("Feature value", fontsize=8)
                axes[idx].set_ylabel("Predicted probability", fontsize=8)
            except Exception as exc:
                logger.warning("PDP failed for %s: %s", feature, exc)
                axes[idx].set_title(f"{feature} (error)", fontsize=9)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("PDP plot saved to %s", save_path)
