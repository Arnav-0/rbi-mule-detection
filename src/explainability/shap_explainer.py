import shap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class MuleExplainer:
    def __init__(self, model, model_type, X_background=None):
        self.model = model
        self.model_type = model_type
        if model_type in ('boosting', 'ensemble'):
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(model, X_background)
        else:  # neural_network or other
            self.explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, 'predict_proba') else model,
                X_background[:100] if X_background is not None else X_background
            )

    def compute_shap_values(self, X) -> np.ndarray:
        values = self.explainer.shap_values(X)
        # Handle different return types (list for tree, array for others)
        if isinstance(values, list):
            return values[1]  # positive class
        return values

    def explain_global(self, X, feature_names=None, save_dir='outputs/plots'):
        shap_values = self.compute_shap_values(X)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Bar plot (mean absolute SHAP)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_bar.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Save raw values
        np.save(save_dir / 'shap_values.npy', shap_values)
        return shap_values

    def explain_local(self, X_single, feature_names=None, account_id=None, save_dir='outputs/plots'):
        shap_values = self.compute_shap_values(
            X_single.reshape(1, -1) if X_single.ndim == 1 else X_single
        )
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Waterfall plot
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=(
                self.explainer.expected_value
                if not isinstance(self.explainer.expected_value, list)
                else self.explainer.expected_value[1]
            ),
            data=X_single.flatten(),
            feature_names=feature_names
        )
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)
        fname = f'shap_waterfall_{account_id}.png' if account_id else 'shap_waterfall.png'
        plt.savefig(save_dir / fname, dpi=150, bbox_inches='tight')
        plt.close()

        # Top features
        top_idx = np.argsort(np.abs(shap_values[0]))[::-1]
        top_features = {}
        for i in top_idx[:10]:
            name = feature_names[i] if feature_names else f'feature_{i}'
            top_features[name] = {
                'shap_value': float(shap_values[0][i]),
                'feature_value': float(X_single.flatten()[i])
            }
        return top_features

    def generate_natural_language(self, top_features: dict) -> str:
        """Convert top SHAP features to a plain-English explanation."""
        parts = []
        feature_descriptions = {
            'rapid_turnover_score': 'rapid fund pass-through behavior',
            'velocity_acceleration': 'sudden increase in transaction frequency',
            'burst_after_dormancy': 'burst of activity after a dormant period',
            'matched_amount_ratio': 'credits quickly matched by similar debits',
            'credit_debit_time_delta_median': 'very short time between credits and debits',
            'structuring_score': 'transactions just below reporting thresholds',
            'round_amount_ratio': 'high proportion of round-amount transactions',
            'dormancy_days': 'long period of account inactivity',
            'txn_volume_vs_income': 'transaction volume disproportionate to account profile',
            'pagerank': 'high importance in the transaction network',
            'betweenness_centrality': 'acting as an intermediary in the network',
            'fan_in_ratio': 'receiving funds from many sources',
            'fan_out_ratio': 'distributing funds to many destinations',
            'unusual_hour_ratio': 'high proportion of after-hours transactions',
            'pct_above_10k': 'frequent high-value transactions',
        }
        for name, info in list(top_features.items())[:5]:
            desc = feature_descriptions.get(name, name.replace('_', ' '))
            direction = 'high' if info['shap_value'] > 0 else 'low'
            if info['shap_value'] > 0:
                parts.append(desc)
        if parts:
            return f"This account was flagged due to: {', '.join(parts)}."
        return "No strong contributing factors identified."


if __name__ == '__main__':
    import logging
    import json
    import joblib
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    # Load model and features
    model = joblib.load('outputs/models/best_model.joblib')
    features = pd.read_parquet('data/processed/features_matrix.parquet')
    labels = pd.read_csv('data/raw/train_labels.csv').set_index('account_id')['is_mule']

    train_features = features.loc[features.index.intersection(labels.index)]
    feature_names = list(train_features.columns)

    logger.info("Computing SHAP values for %d training accounts with %d features...",
                len(train_features), len(feature_names))

    # Use TreeExplainer for CatBoost/XGBoost/LightGBM
    explainer = MuleExplainer(model, model_type='boosting')

    # Global explanations
    logger.info("Generating global SHAP plots...")
    shap_values = explainer.explain_global(
        train_features.values, feature_names=feature_names, save_dir='outputs/plots'
    )
    logger.info("Saved: outputs/plots/shap_beeswarm.png, shap_bar.png, shap_values.npy")

    # Local explanations for top mule accounts
    mule_accounts = labels[labels == 1].index
    mule_in_features = train_features.index.intersection(mule_accounts)
    n_local = min(5, len(mule_in_features))
    logger.info("Generating %d local waterfall plots for mule accounts...", n_local)

    all_explanations = {}
    for acct_id in mule_in_features[:n_local]:
        X_single = train_features.loc[acct_id].values
        top = explainer.explain_local(
            X_single, feature_names=feature_names,
            account_id=acct_id, save_dir='outputs/plots'
        )
        nl = explainer.generate_natural_language(top)
        all_explanations[acct_id] = {'top_features': top, 'natural_language': nl}
        logger.info("  %s: %s", acct_id, nl)

    # Save explanations JSON
    with open('outputs/shap_values/explanations.json', 'w') as f:
        json.dump(all_explanations, f, indent=2, default=str)

    logger.info("SHAP analysis complete. Outputs in outputs/plots/ and outputs/shap_values/")
