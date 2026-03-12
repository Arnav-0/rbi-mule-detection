"""Page 3: Model Comparison — ROC/PR curves, confusion matrices, benchmarks."""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.components.charts import (
    _apply_theme, plot_roc_curves, plot_pr_curves, plot_confusion_matrix,
    plot_calibration, MULE_COLOR, LEGIT_COLOR, ACCENT_COLOR,
    NEON_PURPLE, NEON_PINK, NEON_CYAN, NEON_MAGENTA, NEON_YELLOW, NEON_BLUE, PALETTE,
    _hex_to_rgba,
)
from frontend.components.tables import format_model_comparison, highlight_best_model
from frontend.components.layout import page_header, section, empty_state, info_callout, pipeline_flow, neon_legend

st.set_page_config(page_title="Model Comparison | RBI Mule Detection", page_icon="🏆", layout="wide")
page_header(
    "Model Comparison",
    "Compare 6 models head-to-head — see which algorithm best detects mule accounts and why"
)

pipeline_flow([
    ("⚙️", "56 Features", "Engineered", "cyan"),
    ("🧠", "6 Models", "LR, RF, XGB, LGBM...", "purple"),
    ("🎯", "Optuna + 5-Fold CV", "Hyperparameters", "magenta"),
    ("🏆", "Evaluation", "You are here", "yellow"),
], highlight=3)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


@st.cache_data
def load_all_reports():
    """Load all model evaluation reports."""
    reports = {}
    for f in REPORTS_DIR.glob("*_report.json"):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                continue
            model_name = f.stem.replace("_report", "")
            reports[model_name] = data
        except (json.JSONDecodeError, KeyError):
            continue
    bench_path = REPORTS_DIR / "benchmark_results.json"
    if bench_path.exists():
        try:
            bench = json.loads(bench_path.read_text(encoding='utf-8'))
            if isinstance(bench, dict):
                for name, data in bench.items():
                    if not isinstance(data, dict):
                        continue
                    if name not in reports and name.lower() not in {k.lower() for k in reports}:
                        reports[name] = data
                    elif name.lower() in {k.lower() for k in reports}:
                        # Merge — keep existing but add missing keys
                        existing_key = next(k for k in reports if k.lower() == name.lower())
                        for key, val in data.items():
                            if key not in reports[existing_key]:
                                reports[existing_key][key] = val
            elif isinstance(bench, list):
                for item in bench:
                    name = item.get("model_type", item.get("name", "unknown"))
                    if name not in reports:
                        reports[name] = item
        except json.JSONDecodeError:
            pass
    return reports


def get_metric(data, key, fallback_key=None):
    """Extract metric from report data."""
    metrics = data.get("metrics", data)
    val = metrics.get(key, 0)
    if val == 0 and fallback_key:
        val = metrics.get(fallback_key, 0)
    return val


reports = load_all_reports()

if reports:
    model_names = list(reports.keys())
    # Canonical display names
    display_names = {m: reports[m].get("model_type", m) for m in model_names}

    # Check if CV validation was used
    any_cv = any(reports[m].get("n_folds") or reports[m].get("cv_std") for m in model_names)
    if any_cv:
        info_callout(
            "How to read this page",
            "Each model was trained with stratified 5-fold cross-validation and Optuna tuning. "
            "Metrics shown are mean values across folds. AUC-ROC measures overall discrimination, "
            "AUC-PR is better for rare events like mule accounts, "
            "and F1 balances precision (avoiding false alarms) with recall (catching real mules)."
        )
    else:
        info_callout(
            "How to read this page",
            "Each model was trained on the same data and evaluated on a held-out set. "
            "AUC-ROC measures overall discrimination, AUC-PR is better for rare events like mule accounts, "
            "and F1 balances precision (avoiding false alarms) with recall (catching real mules)."
        )

    # --- Summary Metrics Row ---
    best_model_name = max(model_names, key=lambda m: get_metric(reports[m], "auc_roc", "roc_auc"))
    best_auc = get_metric(reports[best_model_name], "auc_roc", "roc_auc")
    best_f1_name = max(model_names, key=lambda m: get_metric(reports[m], "f1_score", "f1"))
    best_recall_name = max(model_names, key=lambda m: get_metric(reports[m], "recall"))

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.metric("Models Evaluated", len(model_names))
    with mc2:
        st.metric("Best AUC-ROC", f"{best_auc:.4f}")
    with mc3:
        st.metric("Best Model", display_names[best_model_name])
    with mc4:
        st.metric("Best F1", f"{get_metric(reports[best_f1_name], 'f1_score', 'f1'):.4f}")
    with mc5:
        st.metric("Best Recall", f"{get_metric(reports[best_recall_name], 'recall'):.1%}")

    st.markdown("")

    # --- Leaderboard Table ---
    section("Performance Leaderboard")
    table_data = []
    for name, data in reports.items():
        table_data.append({
            "model_type": display_names[name],
            "metrics": {
                "auc_roc": get_metric(data, "auc_roc", "roc_auc"),
                "auc_pr": get_metric(data, "auc_pr", "avg_precision"),
                "f1_score": get_metric(data, "f1_score", "f1"),
                "precision": get_metric(data, "precision"),
                "recall": get_metric(data, "recall"),
                "accuracy": get_metric(data, "accuracy"),
            }
        })

    comp_df = format_model_comparison(table_data)
    if not comp_df.empty:
        st.dataframe(highlight_best_model(comp_df), use_container_width=True, height=280)
        best = comp_df.iloc[0]
        best_key = max(model_names, key=lambda m: get_metric(reports[m], "auc_roc", "roc_auc"))
        cv_std = reports[best_key].get("cv_std", {})
        if cv_std:
            auc_std = cv_std.get("auc_roc", 0)
            st.success(
                f"**Best Model: {best['Model']}** — AUC-ROC: {best['AUC-ROC']:.4f} +/- {auc_std:.4f}, "
                f"AUC-PR: {best['AUC-PR']:.4f} +/- {cv_std.get('auc_pr', 0):.4f}, "
                f"F1: {best['F1 Score']:.4f} +/- {cv_std.get('f1_score', 0):.4f}  "
                f"(5-fold stratified CV)"
            )
        else:
            st.success(
                f"**Best Model: {best['Model']}** — AUC-ROC: {best['AUC-ROC']:.4f}, "
                f"AUC-PR: {best['AUC-PR']:.4f}, F1: {best['F1 Score']:.4f}, "
                f"Recall: {best['Recall']:.1%}"
            )

    st.markdown("")

    # --- Tabs ---
    tab_detail, tab_auc, tab_curves, tab_cm, tab_hyper, tab_raw = st.tabs([
        "Detailed View", "AUC Comparison", "ROC & PR Curves",
        "Confusion Matrices", "Hyperparameters", "Raw Data"
    ])

    with tab_detail:
        # Per-model detail cards
        section("Individual Model Performance")
        info_callout(
            "Model cards",
            "Each card shows a model's key metrics, training details, and confusion matrix. "
            "Green highlights indicate the best score for that metric across all models."
        )

        # Find best values for highlighting
        all_metrics_keys = ["auc_roc", "auc_pr", "f1_score", "precision", "recall", "accuracy"]
        best_vals = {}
        for mk in all_metrics_keys:
            fk = "roc_auc" if mk == "auc_roc" else ("avg_precision" if mk == "auc_pr" else ("f1" if mk == "f1_score" else None))
            best_vals[mk] = max(get_metric(reports[m], mk, fk) for m in model_names)

        # Display in rows of 2
        sorted_models = sorted(model_names, key=lambda m: get_metric(reports[m], "auc_roc", "roc_auc"), reverse=True)
        for row_start in range(0, len(sorted_models), 2):
            cols = st.columns(2)
            for col_idx in range(2):
                idx = row_start + col_idx
                if idx >= len(sorted_models):
                    break
                m = sorted_models[idx]
                data = reports[m]
                dname = display_names[m]
                with cols[col_idx]:
                    rank = idx + 1
                    rank_emoji = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"

                    auc_roc = get_metric(data, "auc_roc", "roc_auc")
                    auc_pr = get_metric(data, "auc_pr", "avg_precision")
                    f1 = get_metric(data, "f1_score", "f1")
                    prec = get_metric(data, "precision")
                    rec = get_metric(data, "recall")
                    acc = get_metric(data, "accuracy")

                    # Card header
                    st.markdown(
                        "<div style='background:linear-gradient(135deg,rgba(14,11,40,0.9),rgba(20,16,52,0.8));"
                        "border:1px solid rgba(123,97,255,0.25);border-radius:16px;padding:24px;margin-bottom:16px;'>"
                        + f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;'>"
                        + f"<span style='font-size:1.8rem;'>{rank_emoji}</span>"
                        + f"<span style='font-size:1.3rem;font-weight:800;color:#f0ecff;'>{dname}</span>"
                        + f"</div>"
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                    # Metrics grid
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("AUC-ROC", f"{auc_roc:.4f}")
                    with m2:
                        st.metric("AUC-PR", f"{auc_pr:.4f}")
                    with m3:
                        st.metric("F1 Score", f"{f1:.4f}")

                    m4, m5, m6 = st.columns(3)
                    with m4:
                        st.metric("Precision", f"{prec:.4f}")
                    with m5:
                        st.metric("Recall", f"{rec:.1%}")
                    with m6:
                        st.metric("Accuracy", f"{acc:.1%}")

                    # Confusion matrix if available
                    cm = data.get("confusion_matrix", data.get("metrics", {}).get("confusion_matrix"))
                    if cm:
                        cm_arr = np.array(cm)
                        tn, fp, fn, tp = cm_arr[0][0], cm_arr[0][1], cm_arr[1][0], cm_arr[1][1]
                        st.markdown(
                            "<div style='background:rgba(12,10,36,0.6);border:1px solid rgba(123,97,255,0.15);"
                            "border-radius:12px;padding:16px;margin-top:8px;'>"
                            + "<div style='color:#7b6cbf;font-size:0.7rem;font-weight:700;text-transform:uppercase;"
                            + "letter-spacing:0.12em;margin-bottom:10px;'>Confusion Matrix</div>"
                            + "<table style='width:100%;text-align:center;color:#b8aae8;font-size:0.85rem;'>"
                            + "<tr><td></td>"
                            + "<td style='color:#7b6cbf;font-weight:600;padding:6px;'>Pred Legit</td>"
                            + "<td style='color:#7b6cbf;font-weight:600;padding:6px;'>Pred Mule</td></tr>"
                            + f"<tr><td style='color:#7b6cbf;font-weight:600;padding:6px;'>True Legit</td>"
                            + f"<td style='background:rgba(0,245,212,0.1);padding:8px;border-radius:8px;"
                            + f"font-weight:700;color:#00f5d4;'>{tn:,}</td>"
                            + f"<td style='background:rgba(255,77,109,0.08);padding:8px;border-radius:8px;"
                            + f"color:#ff4d6d;'>{fp:,}</td></tr>"
                            + f"<tr><td style='color:#7b6cbf;font-weight:600;padding:6px;'>True Mule</td>"
                            + f"<td style='background:rgba(255,77,109,0.08);padding:8px;border-radius:8px;"
                            + f"color:#ff4d6d;'>{fn:,}</td>"
                            + f"<td style='background:rgba(0,245,212,0.1);padding:8px;border-radius:8px;"
                            + f"font-weight:700;color:#00f5d4;'>{tp:,}</td></tr>"
                            + "</table></div>",
                            unsafe_allow_html=True,
                        )

                    # Training info
                    n_train = data.get("n_train", 0)
                    n_folds = data.get("n_folds", 0)
                    validation_str = data.get("validation", "")
                    if n_train and validation_str:
                        st.caption(f"Train: {n_train:,} | Validation: {validation_str}")
                    elif n_train:
                        n_val = data.get("n_val", 0)
                        st.caption(f"Train: {n_train:,} | Val: {n_val:,}")

                    st.markdown("")

        # Radar chart comparing all models
        st.markdown("")
        section("Multi-Metric Radar")
        info_callout(
            "Reading the radar chart",
            "Each axis represents a metric (0 to 1). The model with the largest filled area "
            "is the strongest overall performer. Hover over traces for exact values."
        )
        radar_metrics = ["AUC-ROC", "AUC-PR", "F1 Score", "Precision", "Recall", "Accuracy"]
        colors_radar = [NEON_PURPLE, NEON_PINK, NEON_CYAN, NEON_YELLOW, NEON_MAGENTA, NEON_BLUE]

        fig = go.Figure()
        for i, (_, row) in enumerate(comp_df.iterrows()):
            vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
            color = colors_radar[i % len(colors_radar)]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.08),
                line=dict(color=color, width=2.5),
                name=row["Model"],
                hovertemplate="%{theta}: %{r:.4f}<extra>%{fullData.name}</extra>",
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(123,97,255,0.12)",
                                tickfont=dict(color="#7b6cbf", size=10)),
                bgcolor="rgba(12,10,36,0.5)",
                angularaxis=dict(gridcolor="rgba(123,97,255,0.12)",
                                 tickfont=dict(color="#b8aae8", size=12)),
            ),
            title="<b>Model Performance Radar — All Models</b>",
            height=550,
            legend=dict(font=dict(color="#b8aae8")),
        )
        st.plotly_chart(_apply_theme(fig), use_container_width=True)

    with tab_auc:
        col_roc_bar, col_pr_bar = st.columns(2)

        auc_rocs = [get_metric(reports[m], "auc_roc", "roc_auc") for m in model_names]
        auc_prs = [get_metric(reports[m], "auc_pr", "avg_precision") for m in model_names]
        d_names = [display_names[m] for m in model_names]

        with col_roc_bar:
            sorted_idx = np.argsort(auc_rocs)[::-1]
            s_names = [d_names[i] for i in sorted_idx]
            s_rocs = [auc_rocs[i] for i in sorted_idx]

            fig = go.Figure(go.Bar(
                x=s_names, y=s_rocs,
                marker=dict(
                    color=s_rocs,
                    colorscale=[[0, "#9b85ff"], [1, NEON_MAGENTA]],
                    line=dict(color="rgba(123,97,255,0.3)", width=1.5),
                ),
                text=[f"{v:.4f}" for v in s_rocs],
                textposition="auto", textfont=dict(color="#f0ecff", size=13, family="Inter"),
                hovertemplate="%{x}<br>AUC-ROC: %{y:.4f}<extra></extra>",
            ))
            fig.update_layout(
                title="<b>AUC-ROC by Model</b>",
                yaxis_title="AUC-ROC",
                yaxis=dict(range=[max(0.5, min(s_rocs) - 0.05), 1.0]),
                height=420,
            )
            st.plotly_chart(_apply_theme(fig), use_container_width=True)

        with col_pr_bar:
            sorted_idx = np.argsort(auc_prs)[::-1]
            s_names = [d_names[i] for i in sorted_idx]
            s_prs = [auc_prs[i] for i in sorted_idx]

            fig = go.Figure(go.Bar(
                x=s_names, y=s_prs,
                marker=dict(
                    color=s_prs,
                    colorscale=[[0, "#9b85ff"], [1, NEON_MAGENTA]],
                    line=dict(color="rgba(123,97,255,0.3)", width=1.5),
                ),
                text=[f"{v:.4f}" for v in s_prs],
                textposition="auto", textfont=dict(color="#f0ecff", size=13, family="Inter"),
                hovertemplate="%{x}<br>AUC-PR: %{y:.4f}<extra></extra>",
            ))
            fig.update_layout(
                title="<b>AUC-PR by Model</b>",
                yaxis_title="AUC-PR",
                yaxis=dict(range=[0, 1.0]),
                height=420,
            )
            st.plotly_chart(_apply_theme(fig), use_container_width=True)

        # Multi-metric grouped bar
        st.markdown("")
        section("All Metrics Side-by-Side")
        metric_keys = ["auc_roc", "auc_pr", "f1_score", "precision", "recall", "accuracy"]
        metric_labels = ["AUC-ROC", "AUC-PR", "F1", "Precision", "Recall", "Accuracy"]
        sorted_models_display = sorted(model_names, key=lambda m: get_metric(reports[m], "auc_roc", "roc_auc"), reverse=True)

        fig = go.Figure()
        for i, m in enumerate(sorted_models_display):
            dname = display_names[m]
            vals = [get_metric(reports[m], mk, "roc_auc" if mk == "auc_roc" else None) for mk in metric_keys]
            fig.add_trace(go.Bar(
                name=dname,
                x=metric_labels,
                y=vals,
                marker_color=colors_radar[i % len(colors_radar)],
                text=[f"{v:.3f}" for v in vals],
                textposition="auto",
                textfont=dict(color="#f0ecff", size=10),
                hovertemplate=f"{dname}<br>" + "%{x}: %{y:.4f}<extra></extra>",
            ))
        fig.update_layout(
            barmode="group",
            title="<b>All Metrics — All Models</b>",
            yaxis_title="Score", yaxis=dict(range=[0, 1.05]),
            height=480,
            legend=dict(font=dict(color="#b8aae8")),
        )
        st.plotly_chart(_apply_theme(fig), use_container_width=True)

    with tab_curves:
        info_callout(
            "ROC vs PR curves",
            "The ROC curve (left) shows trade-off between catching mules and false alarms. "
            "The PR curve (right) is better for imbalanced data — it shows how precise the model is "
            "at each level of recall. Only models with stored curve data are shown."
        )
        col_roc, col_pr = st.columns(2)

        roc_data = {}
        pr_data = {}
        for name, data in reports.items():
            curves = data.get("curves", {})
            if "fpr" in curves and "tpr" in curves:
                roc_data[display_names[name]] = {
                    "fpr": curves["fpr"], "tpr": curves["tpr"],
                    "auc": get_metric(data, "auc_roc", "roc_auc")
                }
            if "recall_curve" in curves and "precision_curve" in curves:
                pr_data[display_names[name]] = {
                    "recall": curves["recall_curve"], "precision": curves["precision_curve"],
                    "auc_pr": get_metric(data, "auc_pr", "avg_precision")
                }

        with col_roc:
            if roc_data:
                st.plotly_chart(plot_roc_curves(roc_data), use_container_width=True)
            else:
                empty_state("ROC curve data not in reports")

        with col_pr:
            if pr_data:
                st.plotly_chart(plot_pr_curves(pr_data), use_container_width=True)
            else:
                empty_state("PR curve data not in reports")

        if not roc_data and not pr_data:
            st.info("Curve data is only available for models with stored FPR/TPR and precision/recall arrays. "
                     "Currently only LightGBM has curve data. Other models show metrics in the other tabs.")

    with tab_cm:
        info_callout(
            "Confusion matrices",
            "Each cell shows how many accounts fell into that category. Top-left = correctly identified legitimate, "
            "bottom-right = correctly caught mules. Off-diagonal cells are errors."
        )
        cm_models = [m for m in model_names if "confusion_matrix" in reports[m]
                     or "confusion_matrix" in reports[m].get("metrics", {})]

        if cm_models:
            # Sort by AUC-ROC
            cm_models = sorted(cm_models, key=lambda m: get_metric(reports[m], "auc_roc", "roc_auc"), reverse=True)
            n_cols = min(3, len(cm_models))
            rows_needed = (len(cm_models) + n_cols - 1) // n_cols
            for row_i in range(rows_needed):
                cols = st.columns(n_cols)
                for col_i in range(n_cols):
                    idx = row_i * n_cols + col_i
                    if idx < len(cm_models):
                        m = cm_models[idx]
                        cm = reports[m].get("confusion_matrix",
                             reports[m].get("metrics", {}).get("confusion_matrix"))
                        if cm:
                            with cols[col_i]:
                                fig = plot_confusion_matrix(np.array(cm), title=display_names[m])
                                st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("No confusion matrix data in reports")

    with tab_hyper:
        section("Hyperparameters")
        info_callout(
            "Tuning details",
            "Each model was tuned with Optuna across 5-fold stratified CV. "
            "Below are the best hyperparameters found (from the best-scoring fold)."
        )

        sorted_models_hp = sorted(model_names, key=lambda m: get_metric(reports[m], "auc_roc", "roc_auc"), reverse=True)
        for m in sorted_models_hp:
            data = reports[m]
            params = data.get("params", {})
            if params:
                with st.expander(f"{display_names[m]} — AUC-ROC: {get_metric(data, 'auc_roc', 'roc_auc'):.4f}", expanded=(m == sorted_models_hp[0])):
                    param_df = pd.DataFrame([
                        {"Parameter": k, "Value": str(v)}
                        for k, v in params.items()
                    ])
                    st.dataframe(param_df, use_container_width=True, hide_index=True, height=min(400, len(params) * 40 + 40))
            else:
                with st.expander(f"{display_names[m]} — no params stored"):
                    st.info("Hyperparameters not saved for this model.")

    with tab_raw:
        for name in sorted(model_names, key=lambda m: get_metric(reports[m], "auc_roc", "roc_auc"), reverse=True):
            with st.expander(f"{display_names[name]}"):
                st.json(reports[name])

else:
    empty_state(
        "No model reports found",
        "Run the training pipeline first. Expected: <code>outputs/reports/*_report.json</code>"
    )
