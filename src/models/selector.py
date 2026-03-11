"""Model selection utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


class ModelSelector:
    def select_best(self, results: dict, metric: str = "auc_roc") -> tuple[str, dict]:
        """Return (name, result_dict) for the model with highest metric."""
        best_name = max(
            results,
            key=lambda n: results[n].get("metrics", {}).get(metric, -np.inf),
        )
        return best_name, results[best_name]

    def statistical_comparison(self, results: dict) -> pd.DataFrame:
        """Pairwise Wilcoxon signed-rank test on CV AUC scores.

        Returns a DataFrame of p-values (rows vs. columns).
        """
        names = [n for n in results if "cv_scores" in results[n]]
        if len(names) < 2:
            return pd.DataFrame()

        pvals = pd.DataFrame(
            np.ones((len(names), len(names))), index=names, columns=names
        )
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                if i >= j:
                    continue
                s1 = np.asarray(results[n1]["cv_scores"])
                s2 = np.asarray(results[n2]["cv_scores"])
                if np.allclose(s1, s2):
                    pval = 1.0
                else:
                    _, pval = wilcoxon(s1, s2)
                pvals.loc[n1, n2] = pval
                pvals.loc[n2, n1] = pval
        return pvals

    def rank_models(self, results: dict, metrics: list[str] | None = None) -> pd.DataFrame:
        """Rank models by multiple metrics; lower rank = better."""
        if metrics is None:
            metrics = ["auc_roc", "auc_pr", "f1"]

        rows = {}
        for name, res in results.items():
            m = res.get("metrics", {})
            rows[name] = {k: m.get(k, np.nan) for k in metrics}

        df = pd.DataFrame(rows).T
        rank_df = df.rank(ascending=False)
        rank_df["mean_rank"] = rank_df.mean(axis=1)
        return rank_df.sort_values("mean_rank")
