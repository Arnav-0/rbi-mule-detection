"""Page 2: Feature Explorer — distributions, correlations, importance."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats as sp_stats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frontend.components.charts import (
    _apply_theme, plot_feature_importance, plot_distribution_comparison,
    plot_correlation_heatmap, MULE_COLOR, LEGIT_COLOR, ACCENT_COLOR,
    NEON_CYAN, NEON_PINK, NEON_BLUE, PALETTE,
)
from frontend.components.layout import page_header, section, empty_state, info_callout, pipeline_flow
from src.features.registry import FEATURE_REGISTRY, get_features_by_group

st.set_page_config(page_title="Features | RBI Mule Detection", page_icon="🔬", layout="wide")
page_header(
    "Feature Explorer",
    "Analyze the 56 engineered features that power mule detection — see how each separates mule from legitimate accounts"
)

pipeline_flow([
    ("📁", "Raw Data", "10 CSV files", "cyan"),
    ("🔄", "8 Generators", "Txn, balance, time...", "purple"),
    ("⚙️", "56 Features", "You are here", "yellow"),
    ("🧠", "Model Input", "Best model", "magenta"),
], highlight=2)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
PLOTS_DIR = Path("outputs/plots")


@st.cache_data
def load_feature_matrix():
    for name in ["features_matrix.parquet", "feature_matrix.parquet"]:
        path = PROCESSED_DIR / name
        if path.exists():
            return pd.read_parquet(path)
    return None


@st.cache_data
def load_labels():
    path = RAW_DIR / "train_labels.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_shap_importance():
    for d in [PLOTS_DIR, Path("outputs/shap_values")]:
        for ext in [".npy", ".npz"]:
            path = d / f"shap_values{ext}"
            if path.exists():
                data = np.load(path, allow_pickle=True)
                vals = data[data.files[0]] if ext == ".npz" else data
                return np.abs(vals).mean(axis=0)
    return None


feat_df = load_feature_matrix()
labels_df = load_labels()
shap_imp = load_shap_importance()

# Sidebar
groups = sorted(set(m["group"] for m in FEATURE_REGISTRY.values()))
with st.sidebar:
    st.markdown("### Filters")
    selected_group = st.selectbox("Feature Group", ["All"] + groups, index=0)
    power_filter = st.multiselect("Power Level", ["High", "Medium"], default=["High", "Medium"])
    st.markdown("")
    st.markdown(
        "<div style='color:#7b6cbf; font-size:0.78rem; line-height:1.6;'>"
        "<b>High power</b> features strongly separate mules from legitimate accounts. "
        "<b>Medium power</b> features add supporting signal."
        "</div>",
        unsafe_allow_html=True,
    )

feature_list = list(FEATURE_REGISTRY.keys()) if selected_group == "All" else get_features_by_group(selected_group)
feature_list = [f for f in feature_list if FEATURE_REGISTRY[f]["power"] in power_filter]

gc1, gc2, gc3 = st.columns(3)
with gc1:
    st.metric("Showing", f"{len(feature_list)} features")
with gc2:
    st.metric("Group", selected_group)
with gc3:
    high_count = sum(1 for f in feature_list if FEATURE_REGISTRY[f]["power"] == "High")
    st.metric("High Power", f"{high_count}")

st.markdown("")

if feat_df is not None and labels_df is not None:
    available_features = [f for f in feature_list if f in feat_df.columns]
    if not available_features:
        empty_state(f"No features from '{selected_group}' in matrix")
        st.stop()

    # Merge labels
    train_ids = set(labels_df["account_id"])
    if "account_id" in feat_df.columns:
        train_feat = feat_df[feat_df["account_id"].isin(train_ids)].copy()
        label_map = labels_df.set_index("account_id")["is_mule"]
        train_feat["is_mule"] = train_feat["account_id"].map(label_map).fillna(0).astype(int)
    else:
        train_mask = feat_df.index.isin(train_ids)
        train_feat = feat_df[train_mask].copy()
        label_map = labels_df.set_index("account_id")["is_mule"]
        train_feat["is_mule"] = train_feat.index.map(label_map).fillna(0).astype(int)

    tab_dist, tab_corr, tab_imp, tab_reg = st.tabs(["Distributions", "Correlations", "Importance", "Registry"])

    with tab_dist:
        info_callout(
            "How to read this",
            "Pick a feature below. The histogram shows how its values differ between mule and "
            "legitimate accounts. A high KS statistic (> 0.3) means the feature strongly separates "
            "the two classes — the model can learn from it."
        )
        selected_feat = st.selectbox("Select Feature", available_features, index=0)
        if selected_feat and len(train_feat) > 0:
            legit = train_feat[train_feat["is_mule"] == 0][selected_feat].dropna()
            mule = train_feat[train_feat["is_mule"] == 1][selected_feat].dropna()

            col1, col2 = st.columns([3, 1])
            with col1:
                fig = plot_distribution_comparison(legit, mule, selected_feat)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if len(legit) > 0 and len(mule) > 0:
                    ks_stat, ks_pval = sp_stats.ks_2samp(legit, mule)
                    st.metric("KS Statistic", f"{ks_stat:.4f}")
                    st.metric("p-value", f"{ks_pval:.2e}")
                    st.metric("Legit Mean", f"{legit.mean():.4f}")
                    st.metric("Mule Mean", f"{mule.mean():.4f}")
                    effect = abs(legit.mean() - mule.mean()) / max(legit.std(), 1e-8)
                    st.metric("Effect Size", f"{effect:.3f}")

            # Box plot
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=legit.sample(min(2000, len(legit)), random_state=42),
                                     name="Legitimate", marker_color=NEON_CYAN, boxmean=True))
            fig_box.add_trace(go.Box(y=mule.sample(min(2000, len(mule)), random_state=42),
                                     name="Mule", marker_color=NEON_PINK, boxmean=True))
            fig_box.update_layout(
                title=f"<b>Box Plot: {selected_feat}</b>",
                height=350, yaxis_title=selected_feat,
            )
            st.plotly_chart(_apply_theme(fig_box), use_container_width=True)

    with tab_corr:
        info_callout(
            "Feature correlations",
            "Highly correlated features (blue/cyan cells) carry redundant information. "
            "The model works best when features are diverse and independent."
        )
        top_n = st.slider("Features to show", 8, min(30, len(available_features)), min(20, len(available_features)))
        if shap_imp is not None:
            feat_names_all = list(feat_df.columns)
            imp_map = {name: float(shap_imp[i]) for i, name in enumerate(feat_names_all) if i < len(shap_imp)}
            top_feats = sorted([f for f in available_features if f in imp_map],
                               key=lambda f: imp_map.get(f, 0), reverse=True)[:top_n]
        else:
            top_feats = available_features[:top_n]
        corr = train_feat[top_feats].corr()
        fig = plot_correlation_heatmap(corr.values, top_feats, f"Top {len(top_feats)} Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)

    with tab_imp:
        info_callout(
            "SHAP importance",
            "Each bar shows the average absolute SHAP value — how much that feature moves "
            "the model's prediction on average. Higher = more influential."
        )
        if shap_imp is not None:
            feat_names_all = list(feat_df.columns)
            avail_imp = [(f, float(shap_imp[i])) for i, f in enumerate(feat_names_all)
                         if f in available_features and i < len(shap_imp)]
            if avail_imp:
                names, vals = zip(*avail_imp)
                fig = plot_feature_importance(list(names), list(vals), top_n=min(25, len(names)),
                                             title="Feature Importance (Mean |SHAP|)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            variances = train_feat[available_features].var().sort_values(ascending=False)
            fig = plot_feature_importance(variances.index.tolist()[:20], variances.values.tolist()[:20],
                                         top_n=20, title="Feature Variance (SHAP not available)")
            st.plotly_chart(fig, use_container_width=True)

    with tab_reg:
        info_callout(
            "Feature registry",
            "Every feature is documented with its group, discriminative power, and a plain-English description."
        )
        reg_data = [{"Feature": n, "Group": m["group"], "Power": m["power"], "Description": m["description"]}
                    for n, m in FEATURE_REGISTRY.items() if n in available_features]
        st.dataframe(pd.DataFrame(reg_data), use_container_width=True, height=500)
else:
    empty_state("Feature matrix not found", "Run: <code>make features</code>")
    section("Feature Registry")
    reg_data = [{"Feature": n, "Group": m["group"], "Power": m["power"], "Description": m["description"]}
                for n, m in FEATURE_REGISTRY.items()]
    st.dataframe(pd.DataFrame(reg_data), use_container_width=True, height=600)
