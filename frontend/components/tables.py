"""Reusable table formatting components — Neon Futuristic Theme."""

import pandas as pd


def format_model_comparison(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        metrics = r.get("metrics", {})
        rows.append({
            "Model": r.get("model_type", "Unknown"),
            "AUC-ROC": metrics.get("auc_roc", 0),
            "AUC-PR": metrics.get("auc_pr", 0),
            "F1 Score": metrics.get("f1_score", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "Accuracy": metrics.get("accuracy", 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Rank"
    return df


def format_fairness_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append({
            "Sensitive Feature": r.get("sensitive_feature", ""),
            "Demographic Parity Diff": round(r.get("demographic_parity_diff", 0), 4),
            "Equalized Odds Diff": round(r.get("equalized_odds_diff", 0), 4),
            "80% Rule Pass": "PASS" if r.get("pass_80_rule") else "FAIL",
        })
    return pd.DataFrame(rows)


def highlight_best_model(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns

    def _highlight_max(s):
        is_max = s == s.max()
        return [
            "background: rgba(123,97,255,0.2); font-weight: bold; color: #f0ecff"
            if v else ""
            for v in is_max
        ]

    styled = df.style.apply(_highlight_max, subset=numeric_cols)
    styled = styled.format({col: "{:.4f}" for col in numeric_cols})
    styled = styled.set_properties(**{
        "background-color": "rgba(12,10,36,0.8)",
        "color": "#b8aae8",
        "border-color": "rgba(123,97,255,0.1)",
    })
    styled = styled.set_properties(subset=["Model"], **{
        "font-weight": "600",
        "color": "#f0ecff",
    })
    return styled


def highlight_fairness(df: pd.DataFrame):
    if "80% Rule Pass" not in df.columns:
        return df.style

    def _color_pass(val):
        if val == "PASS":
            return "color: #00f5d4; font-weight: bold"
        elif val == "FAIL":
            return "color: #ff4d6d; font-weight: bold"
        return ""

    styled = df.style.map(_color_pass, subset=["80% Rule Pass"])
    styled = styled.set_properties(**{
        "background-color": "rgba(12,10,36,0.8)",
        "color": "#b8aae8",
        "border-color": "rgba(123,97,255,0.1)",
    })
    return styled
