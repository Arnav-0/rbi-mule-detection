from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class PDPAnalyzer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def compute_pdp(self, X, feature_idx, grid_resolution=50):
        result = partial_dependence(
            self.model, X, features=[feature_idx], grid_resolution=grid_resolution
        )
        return {'grid': result['grid_values'][0], 'pdp': result['average'][0]}

    def plot_top_features(self, X, top_n=10, save_dir='outputs/plots'):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get feature importances to find top features
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            importances = np.ones(X.shape[1])
        top_indices = np.argsort(importances)[::-1][:top_n]

        rows = 2
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
        axes = axes.flatten()

        for i, feat_idx in enumerate(top_indices):
            if i >= rows * cols:
                break
            result = self.compute_pdp(X, feat_idx)
            axes[i].plot(result['grid'], result['pdp'])
            axes[i].set_title(
                self.feature_names[feat_idx]
                if feat_idx < len(self.feature_names)
                else f'Feature {feat_idx}'
            )
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Partial Dependence')

        plt.tight_layout()
        plt.savefig(save_dir / 'pdp_top_features.png', dpi=150, bbox_inches='tight')
        plt.close()
