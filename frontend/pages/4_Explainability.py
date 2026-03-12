"""Page 4: Explainability — SHAP analysis and per-account explanations."""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.components.charts import (
    _apply_theme, plot_feature_importance,
    MULE_COLOR, LEGIT_COLOR, ACCENT_COLOR,
    NEON_PINK, NEON_CYAN, NEON_BLUE, NEON_MAGENTA,
    _hex_to_rgba,
)
from frontend.components.layout import page_header, section, empty_state, info_callout, neon_legend, pipeline_flow

st.set_page_config(page_title="Explainability | RBI Mule Detection", page_icon="🧠", layout="wide")
page_header(
    "Model Explainability",
    "Understand WHY the model flags accounts as mules — global patterns and per-account breakdowns using SHAP"
)

pipeline_flow([
    ("🧠", "Trained Model", "CatBoost", "purple"),
    ("🔬", "SHAP Explainer", "TreeExplainer", "magenta"),
    ("📊", "Feature Impact", "Per-account SHAP", "cyan"),
    ("💬", "Explanation", "You are here", "yellow"),
], highlight=3)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHAP_DIR = PROJECT_ROOT / "outputs" / "shap_values"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@st.cache_data
def load_shap_values():
    for d in [PLOTS_DIR, SHAP_DIR]:
        for ext in [".npy", ".npz"]:
            path = d / f"shap_values{ext}"
            if path.exists():
                data = np.load(path, allow_pickle=True)
                if ext == ".npz":
                    return data[data.files[0]]
                return data
    return None


@st.cache_data
def load_feature_names():
    path = SHAP_DIR / "feature_names.json"
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    path = SHAP_DIR / "feature_names.npy"
    if path.exists():
        return list(np.load(path, allow_pickle=True))
    from src.features.registry import get_all_feature_names
    return get_all_feature_names()


@st.cache_data
def load_explanations_json():
    path = SHAP_DIR / "explanations.json"
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return None


@st.cache_data
def load_account_ids():
    path = SHAP_DIR / "account_ids.npy"
    if path.exists():
        return list(np.load(path, allow_pickle=True))
    path = SHAP_DIR / "account_ids.json"
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return None


shap_values = load_shap_values()
feature_names = load_feature_names()
account_ids = load_account_ids()

info_callout(
    "What is SHAP?",
    "SHAP (SHapley Additive exPlanations) uses game theory to explain each prediction. "
    "Every feature gets a score showing how much it pushed the prediction toward mule (+) or "
    "legitimate (-). This makes the model's reasoning transparent and auditable."
)

# Summary metrics
if shap_values is not None:
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Accounts Explained", f"{shap_values.shape[0]:,}")
    with mc2:
        st.metric("Features", f"{shap_values.shape[1]}")
    with mc3:
        mean_abs = np.abs(shap_values).mean()
        st.metric("Mean |SHAP|", f"{mean_abs:.4f}")
    with mc4:
        max_shap = np.abs(shap_values).max()
        st.metric("Max |SHAP|", f"{max_shap:.4f}")

st.markdown("")

# --- Tabs ---
tab_bar, tab_beeswarm, tab_detail, tab_pdp = st.tabs([
    "Global Importance", "Beeswarm Plot", "Per-Account", "Partial Dependence"
])

with tab_bar:
    if shap_values is not None and feature_names:
        info_callout(
            "Global feature importance",
            "These bars show the average absolute SHAP value across all accounts. "
            "The top features are the ones the model relies on most. A feature with a high "
            "mean |SHAP| strongly influences most predictions."
        )
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        fig = plot_feature_importance(
            feature_names, mean_abs_shap.tolist(),
            top_n=min(25, len(feature_names)),
            title="Mean |SHAP| — Top 25 Features"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("SHAP Statistics Table"):
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean |SHAP|": mean_abs_shap,
                "Std SHAP": np.std(shap_values, axis=0),
                "Max |SHAP|": np.max(np.abs(shap_values), axis=0),
                "Min SHAP": np.min(shap_values, axis=0),
                "Max SHAP": np.max(shap_values, axis=0),
            }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
            st.dataframe(shap_df, use_container_width=True, height=400)
    else:
        empty_state("No SHAP values found", "Run: <code>python -m src.explainability.shap_explainer</code>")

with tab_beeswarm:
    beeswarm_path = PLOTS_DIR / "shap_beeswarm.png"
    if beeswarm_path.exists():
        st.image(str(beeswarm_path), caption="SHAP Beeswarm Plot", use_column_width=True)
    elif shap_values is not None and feature_names:
        info_callout(
            "How to read the beeswarm",
            "Each dot is one account. Horizontal position shows the SHAP value (left = lowers mule risk, "
            "right = raises it). Dot color shows the feature's actual value (blue = low, red = high). "
            "Look for patterns — e.g., if high values of a feature always push right, that feature is a strong mule indicator."
        )
        section("SHAP Value Distribution (Top 20)")
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-20:][::-1]
        top_names = [feature_names[i] for i in top_idx]

        fig = go.Figure()
        sample_n = min(500, shap_values.shape[0])
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(shap_values.shape[0], sample_n, replace=False)

        for rank, fi in enumerate(top_idx):
            vals = shap_values[sample_idx, fi]
            jitter = rng.normal(0, 0.15, len(vals))
            fig.add_trace(go.Scatter(
                x=vals, y=[rank + j for j in jitter],
                mode="markers",
                marker=dict(
                    size=3, opacity=0.6,
                    color=vals,
                    colorscale=[[0, NEON_BLUE], [0.5, "#0a081e"], [1, NEON_PINK]],
                    showscale=False,
                ),
                name=feature_names[fi],
                showlegend=False,
                hovertemplate=f"{feature_names[fi]}<br>SHAP: %{{x:.4f}}<extra></extra>",
            ))

        fig.update_layout(
            title="<b>SHAP Value Distribution (Beeswarm Style)</b>",
            xaxis_title="SHAP Value (impact on prediction)",
            yaxis=dict(
                tickvals=list(range(len(top_idx))),
                ticktext=top_names,
            ),
            height=650,
        )
        st.plotly_chart(_apply_theme(fig), use_container_width=True)

        neon_legend([
            (NEON_BLUE, "Decreases mule risk"),
            (NEON_PINK, "Increases mule risk"),
        ])
    else:
        empty_state("No SHAP data available")

with tab_detail:
    section("Per-Account SHAP Explanation")
    info_callout(
        "Individual explanations",
        "Select an account to see exactly which features drove its prediction. "
        "Positive SHAP values (red bars) push toward mule classification, "
        "negative values (blue bars) push toward legitimate."
    )

    if shap_values is not None and feature_names and account_ids:
        col_select, col_info = st.columns([2, 1])
        with col_select:
            selected_account = st.selectbox(
                "Select Account ID",
                options=account_ids[:500],
                index=0,
            )
        with col_info:
            if selected_account:
                acc_idx = account_ids.index(selected_account)
                total_shap = np.sum(shap_values[acc_idx])
                st.metric("Net SHAP Sum", f"{total_shap:+.4f}")

        if selected_account:
            acc_idx = account_ids.index(selected_account)
            acc_shap = shap_values[acc_idx]

            # Waterfall-style chart
            sorted_idx = np.argsort(np.abs(acc_shap))[::-1][:15]
            feat_names_sorted = [feature_names[i] for i in sorted_idx]
            shap_sorted = [acc_shap[i] for i in sorted_idx]

            colors = [NEON_PINK if v > 0 else NEON_BLUE for v in shap_sorted]

            fig = go.Figure(go.Bar(
                x=shap_sorted,
                y=feat_names_sorted,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in shap_sorted],
                textposition="auto",
                textfont=dict(color="#f0ecff", size=11),
                hovertemplate="%{y}<br>SHAP: %{x:+.4f}<extra></extra>",
            ))
            fig.update_layout(
                title=f"<b>SHAP Waterfall — Account {selected_account}</b>",
                xaxis_title="SHAP Value (positive = higher mule risk)",
                height=500,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(_apply_theme(fig), use_container_width=True)

            # Natural language explanation
            explanations = load_explanations_json()
            if explanations and selected_account in explanations:
                st.markdown("#### Natural Language Explanation")
                st.info(explanations[selected_account])
            else:
                top3 = sorted_idx[:3]
                st.markdown("#### Key Drivers")
                for i in top3:
                    direction = "increases" if acc_shap[i] > 0 else "decreases"
                    st.markdown(f"- **{feature_names[i]}** {direction} mule risk (SHAP: {acc_shap[i]:+.4f})")
    else:
        empty_state("No per-account SHAP data available", "Run the explainability pipeline first")

with tab_pdp:
    section("Partial Dependence Plots")
    info_callout(
        "What are PDPs?",
        "Partial Dependence Plots show how changing one feature's value affects the model's "
        "prediction, averaging over all other features. They reveal whether the relationship "
        "is linear, threshold-based, or more complex."
    )

    pdp_dir = OUTPUTS_DIR / "plots" / "pdp"
    if pdp_dir.exists():
        pdp_files = sorted(pdp_dir.glob("*.png"))
        if pdp_files:
            selected_pdp = st.selectbox(
                "Select Feature for PDP",
                options=[f.stem for f in pdp_files],
            )
            if selected_pdp:
                img_path = pdp_dir / f"{selected_pdp}.png"
                if img_path.exists():
                    st.image(str(img_path), caption=f"PDP: {selected_pdp}", use_column_width=True)
        else:
            empty_state("No PDP plots found", "Expected in <code>outputs/plots/pdp/</code>")
    else:
        if feature_names:
            selected_feat = st.selectbox(
                "Select Feature for PDP",
                options=feature_names[:20],
            )
            st.info(f"PDP for `{selected_feat}` not yet generated. Run: `python -m src.explainability.pdp`")
        else:
            empty_state("No PDP data available")
