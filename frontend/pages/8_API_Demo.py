"""Page 8: API Demo — live prediction interface (works without FastAPI server)."""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.components.charts import (
    _apply_theme, plot_gauge,
    MULE_COLOR, LEGIT_COLOR, ACCENT_COLOR,
    NEON_PURPLE, NEON_PINK, NEON_CYAN, NEON_BLUE, NEON_YELLOW,
    _hex_to_rgba,
)
from frontend.components.layout import page_header, section, empty_state, info_callout, neon_legend, pipeline_flow

st.set_page_config(page_title="API Demo | RBI Mule Detection", page_icon="🚀", layout="wide")
page_header(
    "API Demo",
    "Test predictions live — submit account IDs and get real-time mule risk scores with explanations"
)

pipeline_flow([
    ("📡", "Input", "Account ID", "cyan"),
    ("🧠", "Model", "CatBoost", "purple"),
    ("🔬", "Explain", "SHAP + features", "magenta"),
    ("📋", "Response", "You are here", "yellow"),
], highlight=3)


@st.cache_resource
def load_model():
    try:
        import joblib
        return joblib.load("outputs/models/best_model.joblib")
    except Exception:
        return None


@st.cache_data
def load_feature_matrix():
    for name in ["features_matrix.parquet", "feature_matrix.parquet"]:
        path = Path("data/processed") / name
        if path.exists():
            return pd.read_parquet(path)
    return None


@st.cache_data
def load_shap_data():
    shap_path = Path("outputs/plots/shap_values.npy")
    names_path = Path("outputs/shap_values/feature_names.json")
    ids_path = Path("outputs/shap_values/account_ids.npy")
    vals = np.load(shap_path, allow_pickle=True) if shap_path.exists() else None
    names = json.loads(names_path.read_text(encoding='utf-8')) if names_path.exists() else None
    ids = list(np.load(ids_path, allow_pickle=True)) if ids_path.exists() else None
    return vals, names, ids


def predict_account(model, feat_df, account_id, threshold=0.5):
    """Predict mule probability for an account."""
    if account_id not in feat_df.index:
        return None
    x = feat_df.loc[[account_id]]
    prob = float(model.predict_proba(x)[:, 1][0])
    label = "MULE" if prob >= threshold else "LEGITIMATE"
    risk = "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.4 else "LOW")
    # Top features by importance
    importances = model.feature_importances_
    feat_vals = x.iloc[0]
    top_idx = importances.argsort()[::-1][:5]
    top_features = []
    for i in top_idx:
        top_features.append({
            "feature": feat_df.columns[i],
            "importance": float(importances[i]),
            "value": float(feat_vals.iloc[i]),
        })
    return {"probability": prob, "label": label, "risk": risk, "top_features": top_features}


model = load_model()
feat_df = load_feature_matrix()
shap_vals, shap_names, shap_ids = load_shap_data()

# Sidebar
with st.sidebar:
    st.markdown("### Prediction Engine")
    if model is not None:
        st.success("Model loaded")
    else:
        st.error("Model not found")
    if feat_df is not None:
        st.success(f"{len(feat_df):,} accounts available")
    else:
        st.error("Feature matrix not found")
    st.markdown("")
    st.markdown(
        "<div style='color:#7b6cbf; font-size:0.78rem; line-height:1.6;'>"
        "Predictions run directly using the trained CatBoost model. "
        "No API server required."
        "</div>",
        unsafe_allow_html=True,
    )

if model is None or feat_df is None:
    empty_state("Model or feature data not available", "Ensure model and features are committed to the repo.")
    st.stop()

info_callout(
    "How this works",
    "Submit account IDs for real-time scoring using the trained CatBoost model. "
    "Each prediction includes a probability, classification, risk level, and the top features driving the decision."
)

# --- Tabs ---
tab_single, tab_batch, tab_info = st.tabs(["Single Prediction", "Batch Prediction", "Model Info"])

with tab_single:
    col_input, col_result = st.columns([1, 2])

    with col_input:
        section("Input")
        account_id = st.text_input("Account ID", value="", placeholder="e.g., ACCT_000077")
        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
        predict_btn = st.button("Predict", type="primary", use_container_width=True)
        st.markdown(
            "<div style='color:#7b6cbf; font-size:0.78rem; margin-top:8px; line-height:1.6;'>"
            "The threshold controls when an account is flagged as a mule. "
            "Lower = more sensitive (more flags), higher = more conservative."
            "</div>",
            unsafe_allow_html=True,
        )

    with col_result:
        if predict_btn and account_id:
            result = predict_account(model, feat_df, account_id, threshold)
            if result:
                prob = result["probability"]

                fig = plot_gauge(prob, title=f"Mule Probability: {account_id}")
                st.plotly_chart(fig, use_container_width=True)

                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric("Probability", f"{prob:.4f}")
                with rc2:
                    st.metric("Prediction", result["label"])
                with rc3:
                    st.metric("Risk Level", result["risk"])

                section("Top Contributing Features")
                for feat in result["top_features"]:
                    st.markdown(
                        f"**{feat['feature']}** — importance: {feat['importance']:.4f}, "
                        f"value: {feat['value']:.4f}"
                    )

                # SHAP explanation if available
                if shap_vals is not None and shap_ids and account_id in shap_ids:
                    acc_idx = shap_ids.index(account_id)
                    acc_shap = shap_vals[acc_idx]
                    sorted_idx = np.argsort(np.abs(acc_shap))[::-1][:5]
                    st.markdown("")
                    section("SHAP Explanation")
                    for i in sorted_idx:
                        val = acc_shap[i]
                        name = shap_names[i] if shap_names else f"feature_{i}"
                        direction = "increases" if val > 0 else "decreases"
                        color = NEON_PINK if val > 0 else NEON_BLUE
                        st.markdown(
                            f"<span style='color:{color};font-weight:bold;'>"
                            f"{'+'if val > 0 else ''}{val:.4f}</span> "
                            f"**{name}** {direction} risk",
                            unsafe_allow_html=True,
                        )

                # NL explanation
                expl_path = Path("outputs/shap_values/explanations.json")
                if expl_path.exists():
                    explanations = json.loads(expl_path.read_text(encoding='utf-8'))
                    if account_id in explanations:
                        nl = explanations[account_id]
                        if isinstance(nl, dict):
                            nl = nl.get("natural_language", "")
                        if nl:
                            st.markdown("")
                            section("Natural Language Explanation")
                            st.info(nl)
            else:
                st.warning(f"Account `{account_id}` not found in feature matrix.")
        elif predict_btn:
            st.warning("Please enter an account ID.")

    # Threshold sensitivity
    if account_id and account_id in feat_df.index:
        st.markdown("")
        section("Threshold Sensitivity Analysis")
        info_callout(
            "What is this?",
            "This chart shows how the account's classification changes at different thresholds. "
            "Red bars = classified as MULE at that threshold. Cyan = LEGITIMATE."
        )
        x = feat_df.loc[[account_id]]
        prob = float(model.predict_proba(x)[:, 1][0])
        thresholds = np.arange(0.1, 0.95, 0.05)

        fig = go.Figure()
        fig.add_hline(y=prob, line_dash="solid", line_color=NEON_PURPLE,
                      annotation_text=f"Probability: {prob:.4f}",
                      annotation_font_color=NEON_PURPLE)

        bar_colors = [NEON_PINK if prob >= t else NEON_CYAN for t in thresholds]
        fig.add_trace(go.Bar(
            x=[f"{t:.2f}" for t in thresholds],
            y=[prob] * len(thresholds),
            marker=dict(color=bar_colors, opacity=0.7,
                        line=dict(color="rgba(123,97,255,0.12)", width=0.5)),
            showlegend=False,
            hovertemplate="Threshold: %{x}<br>Probability: %{y:.4f}<extra></extra>",
        ))

        fig.update_layout(
            title=f"<b>Threshold Sensitivity — {account_id}</b>",
            xaxis_title="Threshold", yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=380,
        )
        st.plotly_chart(_apply_theme(fig), use_container_width=True)

        neon_legend([
            (NEON_PINK, "Classified as MULE"),
            (NEON_CYAN, "Classified as LEGITIMATE"),
        ])

with tab_batch:
    section("Batch Prediction")
    info_callout(
        "Batch scoring",
        "Submit multiple account IDs at once for scoring. "
        "Results include probability, classification, and a distribution chart."
    )

    col_batch_input, col_batch_result = st.columns([1, 2])

    with col_batch_input:
        batch_input = st.text_area(
            "Account IDs (one per line)",
            placeholder="ACCT_000077\nACCT_000123\nACCT_000456",
            height=200,
        )
        batch_threshold = st.slider("Batch Threshold", 0.0, 1.0, 0.5, 0.01, key="batch_thresh")
        batch_btn = st.button("Run Batch Prediction", type="primary", use_container_width=True)

    with col_batch_result:
        if batch_btn:
            ids = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]
            if ids:
                results = []
                for aid in ids:
                    r = predict_account(model, feat_df, aid, batch_threshold)
                    if r:
                        results.append({"account_id": aid, "probability": r["probability"],
                                       "prediction": r["label"], "risk": r["risk"]})
                    else:
                        results.append({"account_id": aid, "probability": None,
                                       "prediction": "NOT FOUND", "risk": "N/A"})

                df = pd.DataFrame(results)
                flagged = sum(1 for r in results if r["prediction"] == "MULE")
                total = len(results)

                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    st.metric("Total", total)
                with bc2:
                    st.metric("Flagged", flagged)
                with bc3:
                    st.metric("Flag Rate", f"{flagged/total:.1%}" if total > 0 else "N/A")

                st.dataframe(df, use_container_width=True, hide_index=True)

                valid_probs = df["probability"].dropna()
                if len(valid_probs) > 0:
                    fig = go.Figure(go.Histogram(
                        x=valid_probs,
                        nbinsx=20,
                        marker=dict(color=NEON_PURPLE, opacity=0.85,
                                    line=dict(color="rgba(123,97,255,0.12)", width=0.5)),
                    ))
                    fig.add_vline(x=batch_threshold, line_dash="dash",
                                  line_color=NEON_YELLOW,
                                  annotation_text=f"Threshold: {batch_threshold}")
                    fig.update_layout(
                        title="<b>Prediction Distribution</b>",
                        xaxis_title="Probability", yaxis_title="Count",
                        height=300,
                    )
                    st.plotly_chart(_apply_theme(fig), use_container_width=True)
            else:
                st.warning("Enter at least one account ID.")

with tab_info:
    col_model, col_features = st.columns(2)

    with col_model:
        section("Model Information")
        bench_path = Path("outputs/reports/benchmark_results.json")
        if bench_path.exists():
            bench = json.loads(bench_path.read_text(encoding='utf-8'))
            if isinstance(bench, dict) and bench:
                best_name = max(bench.keys(),
                               key=lambda k: bench[k].get("metrics", {}).get("auc_roc", 0))
                best = bench[best_name]
                st.metric("Best Model", best.get("model_type", best_name))
                st.metric("AUC-ROC", f"{best.get('metrics', {}).get('auc_roc', 0):.4f}")
                st.metric("Features", best.get("n_features", "?"))
                st.metric("Training Samples", f"{best.get('n_train', 0):,}")
                st.metric("Validation", best.get("validation", "N/A"))
                with st.expander("Full Benchmark Data"):
                    st.json(bench)

    with col_features:
        section("Feature Information")
        from src.features.registry import FEATURE_REGISTRY, get_features_by_group
        groups = {}
        for name, meta in FEATURE_REGISTRY.items():
            g = meta["group"]
            if g not in groups:
                groups[g] = []
            groups[g].append(name)
        st.metric("Total Features", len(FEATURE_REGISTRY))
        for group, feats in sorted(groups.items()):
            with st.expander(f"{group} ({len(feats)} features)"):
                for f in feats:
                    st.markdown(f"- `{f}` — {FEATURE_REGISTRY[f]['description']}")
