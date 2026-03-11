"""Page 8: API Demo — live prediction interface."""

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
    "Test the prediction API live — submit account IDs and get real-time mule risk scores with explanations"
)

pipeline_flow([
    ("📡", "REST Request", "Account ID", "cyan"),
    ("🚀", "FastAPI", "13 endpoints", "purple"),
    ("🧠", "Model", "LightGBM", "magenta"),
    ("🔬", "Explain", "SHAP values", "pink"),
    ("📋", "Response", "You are here", "yellow"),
], highlight=4)

API_BASE = "http://localhost:8001"


def try_api_call(endpoint: str, method: str = "GET", json_data: dict = None):
    """Make API call with error handling."""
    try:
        import requests
        if method == "GET":
            resp = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        else:
            resp = requests.post(f"{API_BASE}{endpoint}", json=json_data, timeout=10)
        return resp.json(), resp.status_code
    except ImportError:
        return {"error": "requests library not installed. Run: pip install requests"}, 500
    except Exception as e:
        return {"error": str(e)}, 500


# --- Sidebar: Health Check & Info ---
with st.sidebar:
    st.markdown("### API Status")
    if st.button("Check Health", use_container_width=True):
        data, status = try_api_call("/health")
        if status == 200:
            st.success(f"API Healthy: v{data.get('version', '?')}")
        else:
            st.error("API unreachable")
    st.markdown("")
    st.markdown(f"**Base URL:** `{API_BASE}`")
    st.markdown(f"**Docs:** `{API_BASE}/docs`")
    st.markdown(f"**Redoc:** `{API_BASE}/redoc`")
    st.markdown("")
    st.markdown(
        "<div style='color:#7b6cbf; font-size:0.78rem; line-height:1.6;'>"
        "The FastAPI backend serves predictions via REST endpoints. "
        "Start it with: <code>make api</code>"
        "</div>",
        unsafe_allow_html=True,
    )

info_callout(
    "How this works",
    "This page connects to the FastAPI backend running on port 8001. "
    "You can submit individual account IDs or batches for real-time scoring. "
    "Each prediction includes a probability, classification, and the top features driving the decision."
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
            with st.spinner("Calling prediction API..."):
                data, status = try_api_call("/predict", method="POST",
                                            json_data={"account_id": account_id, "threshold": threshold})

            if status == 200:
                prob = data.get("probability", 0)
                label = data.get("label", "UNKNOWN")
                top_features = data.get("top_features", [])

                # Gauge
                fig = plot_gauge(prob, title=f"Mule Probability: {account_id}")
                st.plotly_chart(fig, use_container_width=True)

                # Result metrics
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.metric("Probability", f"{prob:.4f}")
                with rc2:
                    st.metric("Prediction", label)
                with rc3:
                    st.metric("Model Version", data.get("model_version", "v1"))

                # Top SHAP features
                if top_features:
                    section("Top Contributing Features")
                    for feat in top_features[:5]:
                        name = feat.get("feature", feat.get("name", "?"))
                        value = feat.get("shap_value", feat.get("value", 0))
                        direction = "increases" if value > 0 else "decreases"
                        color = NEON_PINK if value > 0 else NEON_BLUE
                        st.markdown(
                            f"<span style='color:{color};font-weight:bold;'>"
                            f"{'+'if value > 0 else ''}{value:.4f}</span> "
                            f"**{name}** {direction} risk",
                            unsafe_allow_html=True,
                        )

                # Natural language
                nl = data.get("natural_language", "")
                if nl:
                    st.markdown("")
                    section("Explanation")
                    st.info(nl)

                with st.expander("Raw API Response"):
                    st.json(data)
            else:
                st.error(f"API Error (status {status})")
                st.json(data)

        elif predict_btn:
            st.warning("Please enter an account ID.")

    # --- Threshold Sensitivity ---
    if account_id:
        st.markdown("")
        section("Threshold Sensitivity Analysis")
        info_callout(
            "What is this?",
            "This chart shows how the account's classification changes at different thresholds. "
            "Red bars = classified as MULE at that threshold. Cyan = LEGITIMATE. "
            "This helps you pick the right operating point."
        )
        data, status = try_api_call("/predict", method="POST",
                                    json_data={"account_id": account_id, "threshold": 0.5})
        if status == 200:
            prob = data.get("probability", 0)
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
        "Submit multiple account IDs at once. The API will score all of them and return "
        "results in a table, along with a distribution chart showing how many were flagged."
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
                with st.spinner(f"Predicting {len(ids)} accounts..."):
                    data, status = try_api_call("/predict/batch", method="POST",
                                                json_data={"account_ids": ids, "threshold": batch_threshold})
                if status == 200:
                    bc1, bc2, bc3 = st.columns(3)
                    with bc1:
                        st.metric("Total", data.get("total", 0))
                    with bc2:
                        st.metric("Flagged", data.get("flagged", 0))
                    with bc3:
                        total = data.get("total", 1)
                        flagged = data.get("flagged", 0)
                        st.metric("Flag Rate", f"{flagged/total:.1%}" if total > 0 else "N/A")

                    preds = data.get("predictions", [])
                    if preds:
                        df = pd.DataFrame(preds)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        # Distribution of predictions
                        if "probability" in df.columns:
                            fig = go.Figure(go.Histogram(
                                x=df["probability"],
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
                    st.error(f"API Error: {status}")
                    st.json(data)
            else:
                st.warning("Enter at least one account ID.")

with tab_info:
    col_model, col_features = st.columns(2)

    with col_model:
        section("Model Information")
        if st.button("Fetch Model Info", use_container_width=True):
            data, status = try_api_call("/model/info")
            if status == 200:
                for key, val in data.items():
                    if key != "features":
                        st.metric(key.replace("_", " ").title(), str(val))
                with st.expander("Raw Response"):
                    st.json(data)
            else:
                st.error(f"Error: {status}")
                st.json(data)

    with col_features:
        section("Feature Information")
        if st.button("Fetch Feature List", use_container_width=True):
            data, status = try_api_call("/model/features")
            if status == 200:
                st.metric("Total Features", data.get("total", 0))
                groups = data.get("groups", {})
                if groups:
                    for group, feats in groups.items():
                        with st.expander(f"{group} ({len(feats)} features)"):
                            for f in feats:
                                st.markdown(f"- `{f}`")
            else:
                st.error(f"Error: {status}")

    # Dashboard stats
    st.markdown("")
    section("Dashboard Statistics")
    if st.button("Fetch Dashboard Stats", use_container_width=True):
        data, status = try_api_call("/dashboard/stats")
        if status == 200:
            ds_cols = st.columns(4)
            items = list(data.items())
            for i, (key, val) in enumerate(items[:4]):
                with ds_cols[i]:
                    st.metric(key.replace("_", " ").title(), str(val))
            with st.expander("Full Response"):
                st.json(data)
        else:
            st.error(f"Error: {status}")
