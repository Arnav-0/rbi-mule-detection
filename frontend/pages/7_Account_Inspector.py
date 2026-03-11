"""Page 7: Account Inspector — deep-dive into individual accounts."""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.components.charts import (
    _apply_theme, plot_gauge, plot_feature_importance,
    MULE_COLOR, LEGIT_COLOR, ACCENT_COLOR,
    NEON_PURPLE, NEON_PINK, NEON_CYAN, NEON_BLUE, NEON_MAGENTA, NEON_YELLOW,
    _hex_to_rgba,
)
from frontend.components.layout import page_header, section, empty_state, info_callout, neon_legend, pipeline_flow

st.set_page_config(page_title="Account Inspector | RBI Mule Detection", page_icon="🔎", layout="wide")
page_header(
    "Account Inspector",
    "Deep-dive into any account — see its risk score, transaction history, and the exact features driving its prediction"
)

pipeline_flow([
    ("🔎", "Account ID", "User input", "cyan"),
    ("⚙️", "Features", "57 values", "purple"),
    ("🧠", "Predict", "Mule probability", "magenta"),
    ("📊", "SHAP", "Feature impact", "pink"),
    ("📋", "Risk Report", "You are here", "yellow"),
], highlight=4)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
PLOTS_DIR = Path("outputs/plots")
SHAP_DIR = Path("outputs/shap_values")


@st.cache_data
def load_feature_matrix():
    for name in ["features_matrix.parquet", "feature_matrix.parquet"]:
        path = PROCESSED_DIR / name
        if path.exists():
            return pd.read_parquet(path)
    return None


@st.cache_data
def load_predictions():
    path = Path("outputs/predictions/submission.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_labels():
    path = RAW_DIR / "train_labels.csv"
    if path.exists():
        return pd.read_csv(path).set_index("account_id")
    return None


@st.cache_data
def load_shap_data():
    vals = None
    path = PLOTS_DIR / "shap_values.npy"
    if path.exists():
        vals = np.load(path, allow_pickle=True)
    names = None
    from src.features.registry import get_all_feature_names
    names = get_all_feature_names()
    return vals, names


@st.cache_data
def load_transactions_for_account(account_id: str):
    """Load transactions for a specific account from raw CSVs."""
    parts = sorted(RAW_DIR.glob("transactions_part_*.csv"))
    txn_list = []
    for p in parts:
        df = pd.read_csv(p, low_memory=False)
        rename = {"transaction_timestamp": "transaction_date", "amount": "transaction_amount", "txn_type": "transaction_type"}
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        if "account_id" in df.columns:
            acc_txn = df[df["account_id"] == account_id]
            if len(acc_txn) > 0:
                txn_list.append(acc_txn)
    if txn_list:
        result = pd.concat(txn_list)
        if "transaction_date" in result.columns:
            result["transaction_date"] = pd.to_datetime(result["transaction_date"], errors="coerce")
        return result
    return None


feat_df = load_feature_matrix()
pred_df = load_predictions()
labels_df = load_labels()
shap_vals, shap_names = load_shap_data()

# Account ID input
all_ids = list(feat_df.index) if feat_df is not None else []

with st.sidebar:
    st.markdown("### Account Lookup")
    account_id = st.text_input("Account ID", value="", placeholder="e.g., ACCT_000077")
    if all_ids:
        st.caption(f"{len(all_ids):,} accounts available")
    st.markdown("")
    st.markdown(
        "<div style='color:#7b6cbf; font-size:0.78rem; line-height:1.6;'>"
        "Enter an account ID to see its full risk profile, transaction history, "
        "and SHAP explanation."
        "</div>",
        unsafe_allow_html=True,
    )

if account_id:
    # --- Profile Row ---
    col_info = st.columns(4)
    with col_info[0]:
        st.metric("Account ID", account_id)
    with col_info[1]:
        if labels_df is not None and account_id in labels_df.index:
            is_mule = int(labels_df.loc[account_id, "is_mule"])
            label_text = "MULE" if is_mule else "LEGITIMATE"
            st.metric("True Label", label_text)
        else:
            st.metric("True Label", "TEST (unknown)")
    with col_info[2]:
        in_features = account_id in feat_df.index if feat_df is not None else False
        st.metric("Features Available", "Yes" if in_features else "No")
    with col_info[3]:
        probability = None
        if pred_df is not None:
            pred_row = pred_df[pred_df["account_id"] == account_id]
            if len(pred_row) > 0:
                probability = float(pred_row.iloc[0]["is_mule"])
                st.metric("Mule Probability", f"{probability:.4f}")
            else:
                st.metric("Mule Probability", "N/A")

    st.markdown("")

    # --- Tabs ---
    tab_risk, tab_txn, tab_shap, tab_feats = st.tabs([
        "Risk Assessment", "Transaction Timeline", "SHAP Explanation", "Feature Values"
    ])

    with tab_risk:
        col_gauge, col_pred_info = st.columns([1, 1])

        with col_gauge:
            section("Risk Score")
            if probability is not None:
                fig = plot_gauge(probability, title=f"Mule Probability: {account_id}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No prediction available for this account.")

        with col_pred_info:
            section("Prediction Details")
            if probability is not None:
                threshold = 0.5
                prediction = "MULE" if probability >= threshold else "LEGITIMATE"
                risk_level = "HIGH" if probability > 0.7 else ("MEDIUM" if probability > 0.4 else "LOW")

                st.metric("Probability", f"{probability:.4f}")
                st.metric("Prediction", prediction)
                st.metric("Risk Level", risk_level)

                info_callout(
                    "What this means",
                    f"This account has a {probability:.1%} probability of being a mule. "
                    f"At the default 0.5 threshold, it is classified as <b>{prediction}</b>. "
                    f"Risk level: <b>{risk_level}</b>."
                )

                # Suspicious window
                if pred_df is not None:
                    row = pred_df[pred_df["account_id"] == account_id].iloc[0]
                    s_start = row.get("suspicious_start")
                    s_end = row.get("suspicious_end")
                    if pd.notna(s_start) and pd.notna(s_end):
                        st.metric("Suspicious Window", f"{s_start} to {s_end}")
            else:
                st.info("No prediction data available.")

    with tab_txn:
        section("Transaction Timeline")
        txn = load_transactions_for_account(account_id)

        if txn is not None and len(txn) > 0:
            st.markdown(f"**{len(txn):,} transactions found**")

            if "transaction_date" in txn.columns and "transaction_amount" in txn.columns:
                txn_sorted = txn.sort_values("transaction_date")

                colors = []
                if "transaction_type" in txn_sorted.columns:
                    for t in txn_sorted["transaction_type"]:
                        t_str = str(t).upper()
                        if t_str == "C":
                            colors.append(NEON_CYAN)
                        elif t_str == "D":
                            colors.append(NEON_PINK)
                        else:
                            colors.append(NEON_PURPLE)
                else:
                    colors = [NEON_PURPLE] * len(txn_sorted)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=txn_sorted["transaction_date"],
                    y=txn_sorted["transaction_amount"],
                    mode="markers",
                    marker=dict(size=7, color=colors, opacity=0.8,
                                line=dict(width=0.5, color="rgba(123,97,255,0.12)")),
                    text=[f"Amount: {a:,.2f}" for a in txn_sorted["transaction_amount"]],
                    hovertemplate="%{x}<br>%{text}<extra></extra>",
                ))

                # Add suspicious window highlight
                if pred_df is not None:
                    pred_row = pred_df[pred_df["account_id"] == account_id]
                    if len(pred_row) > 0:
                        s_start = pred_row.iloc[0].get("suspicious_start")
                        s_end = pred_row.iloc[0].get("suspicious_end")
                        if pd.notna(s_start) and pd.notna(s_end):
                            fig.add_vrect(
                                x0=s_start, x1=s_end,
                                fillcolor=_hex_to_rgba(NEON_PINK, 0.15),
                                line=dict(color=NEON_PINK, width=2, dash="dash"),
                                annotation_text="Suspicious Window",
                                annotation_font_color=NEON_PINK,
                            )

                fig.update_layout(
                    title=f"<b>Transaction Timeline — {account_id}</b>",
                    xaxis_title="Date", yaxis_title="Amount",
                    height=420,
                )
                st.plotly_chart(_apply_theme(fig), use_container_width=True)

                neon_legend([
                    (NEON_CYAN, "Credit"),
                    (NEON_PINK, "Debit"),
                    (NEON_PURPLE, "Other"),
                ])

                # Amount x Hour heatmap
                st.markdown("")
                section("Transaction Pattern: Amount x Hour")
                info_callout(
                    "Why this matters",
                    "Mule accounts often show unusual patterns — transactions at odd hours or in "
                    "specific amount ranges. This heatmap reveals the account's behavioral fingerprint."
                )
                txn_copy = txn.copy()
                txn_copy["hour"] = txn_copy["transaction_date"].dt.hour
                amount_bins = [0, 100, 500, 1000, 5000, 10000, 50000, float("inf")]
                bin_labels = ["0-100", "100-500", "500-1K", "1K-5K", "5K-10K", "10K-50K", "50K+"]
                txn_copy["amount_bin"] = pd.cut(txn_copy["transaction_amount"], bins=amount_bins,
                                                labels=bin_labels, right=False)
                heatmap_data = txn_copy.groupby(["amount_bin", "hour"], observed=True).size().unstack(fill_value=0)

                if not heatmap_data.empty:
                    fig = go.Figure(go.Heatmap(
                        z=heatmap_data.values,
                        x=[str(h) for h in heatmap_data.columns],
                        y=heatmap_data.index.astype(str).tolist(),
                        colorscale=[[0, "#020209"], [0.3, "#312E81"], [0.6, NEON_PURPLE], [1, NEON_MAGENTA]],
                        hovertemplate="Hour: %{x}<br>Amount: %{y}<br>Count: %{z}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=f"<b>Transaction Heatmap — {account_id}</b>",
                        xaxis_title="Hour of Day", yaxis_title="Amount Range",
                        height=380,
                    )
                    st.plotly_chart(_apply_theme(fig), use_container_width=True)

                # Transaction summary stats
                st.markdown("")
                section("Transaction Summary")
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.metric("Total Txns", f"{len(txn):,}")
                with sc2:
                    st.metric("Total Volume", f"{txn['transaction_amount'].sum():,.0f}")
                with sc3:
                    st.metric("Avg Amount", f"{txn['transaction_amount'].mean():,.0f}")
                with sc4:
                    st.metric("Max Amount", f"{txn['transaction_amount'].max():,.0f}")
        else:
            empty_state("No transaction data found for this account")

    with tab_shap:
        section("SHAP Explanation")
        info_callout(
            "Per-account SHAP",
            "These bars show which features pushed this specific account toward or away from "
            "a mule classification. Red bars increase mule risk, blue bars decrease it."
        )
        if shap_vals is not None and shap_names and feat_df is not None and account_id in feat_df.index:
            train_ids = list(feat_df.loc[feat_df.index.isin(
                pd.read_csv(RAW_DIR / "train_labels.csv")["account_id"]
            )].index)
            if account_id in train_ids:
                acc_idx = train_ids.index(account_id)
                if acc_idx < shap_vals.shape[0]:
                    acc_shap = shap_vals[acc_idx]
                    sorted_idx = np.argsort(np.abs(acc_shap))[::-1][:15]
                    feat_names = [shap_names[i] for i in sorted_idx]
                    shap_sorted = [acc_shap[i] for i in sorted_idx]
                    colors_shap = [NEON_PINK if v > 0 else NEON_BLUE for v in shap_sorted]

                    fig = go.Figure(go.Bar(
                        x=shap_sorted, y=feat_names, orientation="h",
                        marker_color=colors_shap,
                        text=[f"{v:+.4f}" for v in shap_sorted],
                        textposition="auto", textfont=dict(color="#f0ecff", size=11),
                        hovertemplate="%{y}<br>SHAP: %{x:+.4f}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=f"<b>SHAP Values — {account_id}</b>",
                        xaxis_title="SHAP Value (+ = higher mule risk)",
                        height=500, yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(_apply_theme(fig), use_container_width=True)

                    # Key drivers
                    st.markdown("#### Key Risk Drivers")
                    for i in sorted_idx[:3]:
                        direction = "increases" if acc_shap[i] > 0 else "decreases"
                        st.markdown(
                            f"- **{shap_names[i]}** {direction} mule risk "
                            f"(SHAP: {acc_shap[i]:+.4f})"
                        )
                else:
                    st.info("SHAP values not available for this account index.")
            else:
                st.info("SHAP values only available for training accounts.")
        else:
            empty_state("No SHAP explanation available", "Run the explainability pipeline first")

    with tab_feats:
        section("Feature Values")
        info_callout(
            "Raw feature values",
            "These are the 57 engineered features computed for this account. "
            "Each value feeds into the model alongside the SHAP explanation above."
        )
        if feat_df is not None and account_id in feat_df.index:
            feat_dict = feat_df.loc[account_id].to_dict()
            feat_table = pd.DataFrame([
                {"Feature": k, "Value": round(v, 6) if isinstance(v, float) else v}
                for k, v in feat_dict.items() if pd.notna(v)
            ])
            st.dataframe(feat_table, use_container_width=True, height=500, hide_index=True)
        else:
            empty_state("No features computed for this account")

else:
    st.markdown(
        "<div style='text-align:center;padding:80px 20px;color:#7b6cbf;'>"
        "<div style='font-size:4rem;margin-bottom:16px;opacity:0.3;'>🔎</div>"
        "<div style='font-size:1.1rem;color:#7b6cbf;font-weight:600;'>Enter an account ID to begin inspection</div>"
        "<div style='font-size:0.85rem;margin-top:8px;color:#7b6cbf;'>Use the sidebar to search for an account</div>"
        "</div>",
        unsafe_allow_html=True,
    )
