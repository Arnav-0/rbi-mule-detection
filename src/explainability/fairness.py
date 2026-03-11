import numpy as np
import pandas as pd
from pathlib import Path
import json


class FairnessAuditor:
    SENSITIVE_FEATURES = ['age_group', 'geography_tier', 'account_type']

    def prepare_sensitive_features(self, profile_df):
        sensitive_df = pd.DataFrame(index=profile_df.index)

        if 'age' in profile_df.columns:
            bins = [0, 25, 45, 65, 200]
            labels = ['<25', '25-45', '45-65', '>65']
            sensitive_df['age_group'] = pd.cut(profile_df['age'], bins=bins, labels=labels)
        elif 'age_group' in profile_df.columns:
            sensitive_df['age_group'] = profile_df['age_group']

        if 'geography_tier' in profile_df.columns:
            sensitive_df['geography_tier'] = profile_df['geography_tier']
        elif 'geo_tier' in profile_df.columns:
            sensitive_df['geography_tier'] = profile_df['geo_tier']

        if 'account_type' in profile_df.columns:
            sensitive_df['account_type'] = profile_df['account_type']

        return sensitive_df

    def audit(self, y_true, y_pred, sensitive_df):
        try:
            from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
            from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
        except ImportError:
            return self._audit_manual(y_true, y_pred, sensitive_df)

        metrics = {
            'accuracy': accuracy_score,
            'recall': recall_score,
            'precision': precision_score,
            'f1': f1_score,
            'selection_rate': lambda y_t, y_p: np.mean(y_p),
        }

        results = {}
        for feature in self.SENSITIVE_FEATURES:
            if feature not in sensitive_df.columns:
                continue

            mf = MetricFrame(
                metrics=metrics,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_df[feature]
            )

            dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_df[feature])

            try:
                eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_df[feature])
            except Exception:
                eod = None

            # 80% rule check
            group_rates = mf.by_group['selection_rate']
            min_rate = group_rates.min()
            max_rate = group_rates.max()
            passes_80_rule = (min_rate / max_rate >= 0.8) if max_rate > 0 else True

            results[feature] = {
                'overall': mf.overall.to_dict(),
                'by_group': mf.by_group.to_dict(),
                'demographic_parity_difference': float(dpd),
                'equalized_odds_difference': float(eod) if eod is not None else None,
                'passes_80_percent_rule': bool(passes_80_rule),
                'min_selection_rate': float(min_rate),
                'max_selection_rate': float(max_rate),
            }

        return results

    def _audit_manual(self, y_true, y_pred, sensitive_df):
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

        results = {}
        for feature in self.SENSITIVE_FEATURES:
            if feature not in sensitive_df.columns:
                continue

            groups = sensitive_df[feature].unique()
            by_group = {}
            for group in groups:
                mask = sensitive_df[feature] == group
                if mask.sum() == 0:
                    continue
                yt = y_true[mask]
                yp = y_pred[mask]
                by_group[str(group)] = {
                    'accuracy': float(accuracy_score(yt, yp)),
                    'recall': float(recall_score(yt, yp, zero_division=0)),
                    'precision': float(precision_score(yt, yp, zero_division=0)),
                    'f1': float(f1_score(yt, yp, zero_division=0)),
                    'selection_rate': float(np.mean(yp)),
                }

            rates = [v['selection_rate'] for v in by_group.values()]
            min_rate = min(rates) if rates else 0
            max_rate = max(rates) if rates else 0
            passes_80_rule = (min_rate / max_rate >= 0.8) if max_rate > 0 else True

            results[feature] = {
                'by_group': by_group,
                'passes_80_percent_rule': bool(passes_80_rule),
                'min_selection_rate': float(min_rate),
                'max_selection_rate': float(max_rate),
            }

        return results

    def generate_report(self, results, save_path='outputs/reports/fairness_report.json'):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            return obj

        report = json.loads(json.dumps(results, default=convert))

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Formatted summary
        summary = "FAIRNESS AUDIT REPORT\n" + "=" * 50 + "\n\n"
        for feature, data in results.items():
            summary += f"Sensitive Feature: {feature}\n"
            summary += f"  Passes 80% Rule: {data.get('passes_80_percent_rule', 'N/A')}\n"
            summary += f"  Min Selection Rate: {data.get('min_selection_rate', 'N/A'):.4f}\n"
            summary += f"  Max Selection Rate: {data.get('max_selection_rate', 'N/A'):.4f}\n"
            if 'demographic_parity_difference' in data:
                summary += f"  Demographic Parity Diff: {data['demographic_parity_difference']:.4f}\n"
            summary += "\n"

        summary_path = save_path.with_suffix('.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)

        return report

    def mitigate_if_needed(self, model, X, y, sensitive_features, threshold=0.8):
        try:
            from fairlearn.postprocessing import ThresholdOptimizer
        except ImportError:
            return None

        mitigated = ThresholdOptimizer(
            estimator=model,
            constraints='demographic_parity',
            objective='balanced_accuracy_score',
        )
        mitigated.fit(X, y, sensitive_features=sensitive_features)
        return mitigated
