"""Model Card generator following Google's Model Card framework."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    def generate(
        self,
        model_info: dict,
        eval_results: dict,
        fairness_results: dict,
        save_path: str = "docs/model_card.md",
    ) -> str:
        """Generate a Model Card in Markdown format.

        model_info: {name, version, type, description, training_date, features}
        eval_results: {auc_roc, auc_pr, f1, precision, recall, brier, threshold}
        fairness_results: output from FairnessAuditor.audit()
        """
        today = date.today().isoformat()
        name = model_info.get("name", "Mule Account Detection Model")
        version = model_info.get("version", "1.0.0")
        model_type = model_info.get("type", "Gradient Boosting")
        description = model_info.get("description", "Binary classifier to detect mule accounts in banking transactions.")
        training_date = model_info.get("training_date", today)
        n_features = model_info.get("n_features", 57)

        auc_roc = eval_results.get("auc_roc", "N/A")
        auc_pr = eval_results.get("auc_pr", "N/A")
        f1 = eval_results.get("f1", "N/A")
        precision = eval_results.get("precision", "N/A")
        recall = eval_results.get("recall", "N/A")
        brier = eval_results.get("brier", "N/A")
        threshold = eval_results.get("threshold", 0.5)

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        fairness_section = self._format_fairness(fairness_results)

        card = f"""# Model Card: {name}

**Version:** {version}
**Date:** {training_date}
**Model Type:** {model_type}

---

## Model Details

{description}

- **Architecture:** {model_type}
- **Input features:** {n_features} engineered features (velocity, amount patterns, temporal, profile mismatch, KYC/behavioral, network graph, interactions)
- **Output:** Probability score [0, 1] that an account is a money mule
- **Decision threshold:** {fmt(threshold)}
- **Training framework:** scikit-learn / XGBoost / LightGBM / CatBoost / PyTorch
- **Hyperparameter tuning:** Optuna (Bayesian optimization)

---

## Intended Use

### Primary Use Cases
- Automated flagging of potential mule accounts for human review
- Prioritization of investigation queues for compliance teams
- Risk scoring for new account onboarding (with caution)

### Out-of-Scope Uses
- Automated account closure without human review
- Credit scoring or loan eligibility decisions
- Use on populations significantly different from training distribution

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | {fmt(auc_roc)} |
| AUC-PR | {fmt(auc_pr)} |
| F1 Score | {fmt(f1)} |
| Precision | {fmt(precision)} |
| Recall | {fmt(recall)} |
| Brier Score | {fmt(brier)} |

*Evaluation performed on held-out 20% stratified split.*

---

## Fairness Analysis

{fairness_section}

---

## Limitations

1. **Temporal drift:** Model performance may degrade as mule behavior patterns evolve. Recommend quarterly retraining.
2. **Class imbalance:** Trained on imbalanced data (~2–5% mule prevalence). Metrics may not reflect real-world performance if prevalence shifts.
3. **Feature availability:** Some features (graph-network) require significant computational resources and may not be available in real-time inference.
4. **Cold-start problem:** Newly opened accounts have limited transaction history, reducing model reliability.
5. **Label quality:** Ground truth labels are derived from confirmed cases; unreported mule accounts contribute noise.

---

## Ethical Considerations

- **Human oversight:** All model flags must be reviewed by a trained compliance officer before any account action.
- **Explainability:** Every prediction is accompanied by top-5 SHAP-based natural language reasons.
- **Appeals process:** Account holders should have a clear mechanism to dispute flagging decisions.
- **Data minimization:** Only transaction and KYC data necessary for fraud detection is used; no sensitive protected attributes are used as model inputs.
- **Regular audits:** Fairness audit should be repeated quarterly and after any model update.

---

## Training Data

- **Source:** Synthetic transaction data representative of Indian retail banking (RBI hackathon dataset)
- **Time period:** Covers multiple months of transaction history
- **Label source:** Confirmed mule account labels from financial intelligence unit
- **Preprocessing:** Outlier capping at 99th percentile, missing value imputation, feature normalization

---

## Caveats

This model was developed for the RBI Mule Account Detection Hackathon. Production deployment requires additional validation, legal review, and RBI regulatory approval.
"""

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(card, encoding="utf-8")
        logger.info("Model card saved to %s", save_path)
        return card

    def _format_fairness(self, fairness_results: dict) -> str:
        if not fairness_results:
            return "_Fairness audit not yet performed._"

        lines = []
        for sf_name, res in fairness_results.items():
            dp_diff = res.get("demographic_parity_difference", "N/A")
            eo_diff = res.get("equalized_odds_difference", "N/A")
            passes = res.get("passes_80pct_rule", False)
            lines.append(f"### {sf_name.replace('_', ' ').title()}")
            lines.append(f"- Demographic Parity Difference: {dp_diff:.4f}" if isinstance(dp_diff, float) else f"- Demographic Parity Difference: {dp_diff}")
            lines.append(f"- Equalized Odds Difference: {eo_diff:.4f}" if isinstance(eo_diff, float) else f"- Equalized Odds Difference: {eo_diff}")
            lines.append(f"- 80% Rule: {'PASS' if passes else 'FAIL — bias mitigation applied'}")
            lines.append("")
        return "\n".join(lines)
