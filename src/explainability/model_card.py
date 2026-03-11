from pathlib import Path
from datetime import datetime


class ModelCardGenerator:
    def generate(self, model_info, eval_results, fairness_results=None, save_path='docs/model_card.md'):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        card = f"""# Model Card: {model_info.get('name', 'Mule Account Detection Model')}

## Model Details

- **Model Name:** {model_info.get('name', 'N/A')}
- **Model Type:** {model_info.get('type', 'N/A')}
- **Version:** {model_info.get('version', '1.0')}
- **Date:** {datetime.now().strftime('%Y-%m-%d')}
- **Framework:** {model_info.get('framework', 'scikit-learn / XGBoost / LightGBM / CatBoost / PyTorch')}
- **License:** Proprietary - RBI Compliance Use Only

## Intended Use

- **Primary Use:** Detection of mule (money laundering intermediary) bank accounts in the Indian banking system
- **Intended Users:** RBI compliance teams, bank fraud investigation units
- **Out-of-Scope Uses:**
  - Not intended for direct customer-facing decisions without human review
  - Not intended for use outside the Indian banking regulatory context
  - Should not be used as the sole basis for account closure or legal action

## Training Data

- **Source:** Synthetic banking transaction data representative of Indian banking patterns
- **Size:** {model_info.get('training_samples', 'N/A')} samples
- **Features:** {model_info.get('n_features', 57)} engineered features across transaction, temporal, network, and behavioral categories
- **Label Distribution:** Binary (mule=1, legitimate=0), class imbalance handled via {model_info.get('imbalance_strategy', 'class weighting and focal loss')}

## Performance Metrics

| Metric | Value |
|--------|-------|
"""
        if eval_results:
            for metric, value in eval_results.items():
                if isinstance(value, float):
                    card += f"| {metric} | {value:.4f} |\n"
                else:
                    card += f"| {metric} | {value} |\n"

        card += """
## Fairness Analysis

"""
        if fairness_results:
            for feature, data in fairness_results.items():
                passes = data.get('passes_80_percent_rule', 'N/A')
                card += f"### {feature}\n"
                card += f"- **Passes 80% Rule:** {passes}\n"
                if 'demographic_parity_difference' in data:
                    card += f"- **Demographic Parity Difference:** {data['demographic_parity_difference']:.4f}\n"
                if 'equalized_odds_difference' in data and data['equalized_odds_difference'] is not None:
                    card += f"- **Equalized Odds Difference:** {data['equalized_odds_difference']:.4f}\n"
                card += "\n"
        else:
            card += "Fairness analysis not yet performed.\n\n"

        card += """## Limitations

- Model performance may degrade on transaction patterns significantly different from training data
- Temporal patterns may shift as mule account operators adapt their strategies
- Model assumes feature engineering pipeline produces consistent outputs
- Performance metrics are based on synthetic/labeled data; real-world performance may vary

## Ethical Considerations

- **False Positives:** May incorrectly flag legitimate accounts, causing inconvenience to customers
- **Bias Risk:** Model should be regularly audited for disparate impact across demographic groups
- **Human Oversight:** All model predictions should be reviewed by trained investigators before action
- **Privacy:** Model operates on aggregated transaction features, not raw transaction details
- **Regulatory Compliance:** Designed to align with RBI guidelines on fraud detection and AML

## Monitoring and Updates

- Model should be retrained quarterly or when performance metrics drop below thresholds
- Feature drift should be monitored continuously
- Fairness metrics should be recalculated with each model update
"""

        with open(save_path, 'w') as f:
            f.write(card)

        return card
