"""Page 1: Dataset Overview — statistics, distributions, trends."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.components.charts import (
    _apply_theme, MULE_COLOR, LEGIT_COLOR, ACCENT_COLOR,
    NEON_CYAN, NEON_PINK, NEON_YELLOW, NEON_BLUE, NEON_MAGENTA, PALETTE,
    _hex_to_rgba,
)
from frontend.components.layout import page_header, section, empty_state, pipeline_flow

st.set_page_config(page_title="Overview | RBI Mule Detection", page_icon="📊", layout="wide")
page_header("Dataset Overview", "Statistics, class distribution, transaction volume, and amount patterns")

pipeline_flow([
    ("📁", "Raw Data", "10 CSV files", "cyan"),
    ("🔄", "Merge & Clean", "Validate schema", "purple"),
    ("⚙️", "Feature Eng.", "57 features", "magenta"),
    ("📊", "Analysis", "You are here", "yellow"),
], highlight=3)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


@st.cache_data
def load_labels():
    path = RAW_DIR / "train_labels.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_features():
    for name in ["features_matrix.parquet", "feature_matrix.parquet"]:
        path = PROCESSED_DIR / name
        if path.exists():
            return pd.read_parquet(path)
    return None


@st.cache_data
def load_transactions_sample():
    parts = sorted(RAW_DIR.glob("transactions_part_*.csv"))
    if not parts:
        return None
    df = pd.read_csv(parts[0], low_memory=False)
    rename = {"transaction_timestamp": "transaction_date", "amount": "transaction_amount", "txn_type": "transaction_type"}
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    return df


@st.cache_data
def load_test_accounts():
    path = RAW_DIR / "test_accounts.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


labels = load_labels()
features = load_features()
txn = load_transactions_sample()
test_accts = load_test_accounts()

# --- Key Metrics ---
total_accounts = len(features) if features is not None else 0
mule_count = int(labels["is_mule"].sum()) if labels is not None else 0
train_count = len(labels) if labels is not None else 0
test_count = len(test_accts) if test_accts is not None else 0
mule_rate = mule_count / train_count if train_count > 0 else 0
n_features = features.shape[1] if features is not None else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Total Accounts", f"{total_accounts:,}")
with c2:
    st.metric("Train Set", f"{train_count:,}")
with c3:
    st.metric("Test Set", f"{test_count:,}")
with c4:
    st.metric("Mule Accounts", f"{mule_count:,}")
with c5:
    st.metric("Mule Rate", f"{mule_rate:.2%}")
with c6:
    st.metric("Features", f"{n_features}")

st.markdown("")

# ═══════════════════════════════════════════════════════════════
# TOP ROW: Class distribution + Amount distribution (always visible)
# ═══════════════════════════════════════════════════════════════
top_col1, top_col2 = st.columns(2)

with top_col1:
    section("Class Distribution")
    if labels is not None:
        counts = labels["is_mule"].value_counts().sort_index()
        legit_n = counts.get(0, 0)
        mule_n = counts.get(1, 0)

        fig = go.Figure(go.Pie(
            labels=["Legitimate", "Mule"],
            values=[legit_n, mule_n],
            marker=dict(colors=[NEON_CYAN, NEON_PINK], line=dict(color="rgba(123,97,255,0.12)", width=2)),
            hole=0.65,
            textinfo="label+percent",
            textfont=dict(size=13, color="#f0ecff"),
            hovertemplate="%{label}: %{value:,} accounts (%{percent})<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{train_count:,}</b><br><span style='font-size:11px;color:#7b6cbf'>accounts</span>",
            x=0.5, y=0.5, font=dict(size=22, color="#f0ecff"), showarrow=False,
        )
        fig.update_layout(
            title="<b>Train Set: Mule vs Legitimate</b>",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=400,
        )
        st.plotly_chart(_apply_theme(fig), use_container_width=True)

with top_col2:
    section("Transaction Amount Distribution")
    if txn is not None and "transaction_amount" in txn.columns:
        amounts = txn["transaction_amount"].dropna()
        if len(amounts) > 25000:
            amounts = amounts.sample(25000, random_state=42)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=amounts, nbinsx=80,
            marker=dict(
                color=NEON_BLUE, opacity=0.75,
                line=dict(color=_hex_to_rgba(NEON_BLUE, 0.4), width=1),
            ),
            hovertemplate="Amount: %{x}<br>Count: %{y:,}<extra></extra>",
        ))
        fig.update_layout(
            title="<b>Transaction Amounts (Log Scale)</b>",
            xaxis_title="Amount", yaxis_title="Count",
            height=400,
        )
        fig.update_xaxes(type="log")
        st.plotly_chart(_apply_theme(fig), use_container_width=True)
    else:
        empty_state("No transaction data")

st.markdown("")

# ═══════════════════════════════════════════════════════════════
# SECOND ROW: Monthly volume + Train/Test split
# ═══════════════════════════════════════════════════════════════
mid_col1, mid_col2 = st.columns([2, 1])

with mid_col1:
    section("Monthly Transaction Volume")
    if txn is not None and "transaction_date" in txn.columns:
        txn_monthly = txn.set_index("transaction_date").resample("ME").size().reset_index()
        txn_monthly.columns = ["month", "count"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=txn_monthly["month"], y=txn_monthly["count"],
            mode="lines+markers",
            line=dict(color=ACCENT_COLOR, width=3, shape="spline"),
            marker=dict(size=7, color=ACCENT_COLOR, line=dict(width=1, color="#020209")),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(ACCENT_COLOR, 0.08),
            hovertemplate="%{x|%b %Y}<br>%{y:,} transactions<extra></extra>",
        ))
        fig.update_layout(
            title="<b>Transaction Volume by Month</b>",
            xaxis_title="Month", yaxis_title="Count",
            height=380,
        )
        st.plotly_chart(_apply_theme(fig), use_container_width=True)

with mid_col2:
    section("Dataset Split")
    fig = go.Figure(go.Bar(
        x=["Train", "Test"],
        y=[train_count, test_count],
        marker=dict(color=[ACCENT_COLOR, NEON_MAGENTA], line=dict(color="rgba(123,97,255,0.12)", width=1.5)),
        text=[f"{train_count:,}", f"{test_count:,}"],
        textposition="auto",
        textfont=dict(color="#f0ecff", size=14),
    ))
    fig.update_layout(title="<b>Train / Test</b>", height=380)
    st.plotly_chart(_apply_theme(fig), use_container_width=True)

st.markdown("")

# ═══════════════════════════════════════════════════════════════
# THIRD ROW: Amount stats + Transaction types + Data quality
# ═══════════════════════════════════════════════════════════════
tab_stats, tab_quality = st.tabs(["Transaction Stats", "Data Quality"])

with tab_stats:
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        if txn is not None and "transaction_amount" in txn.columns:
            raw_amounts = txn["transaction_amount"].dropna()
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Mean Amount", f"{raw_amounts.mean():,.0f}")
            with m2:
                st.metric("Median", f"{raw_amounts.median():,.0f}")
            with m3:
                st.metric("Max", f"{raw_amounts.max():,.0f}")
            with m4:
                st.metric("Total Txns", f"{len(txn):,}")

    with s_col2:
        if txn is not None and "transaction_type" in txn.columns:
            section("Transaction Types")
            type_counts = txn["transaction_type"].value_counts()
            fig = go.Figure(go.Bar(
                x=type_counts.index.astype(str), y=type_counts.values,
                marker=dict(color=PALETTE[:len(type_counts)], line=dict(color="rgba(123,97,255,0.12)", width=1)),
                text=[f"{v:,}" for v in type_counts.values],
                textposition="auto", textfont=dict(color="#f0ecff"),
            ))
            fig.update_layout(title="<b>Transaction Types</b>", height=320)
            st.plotly_chart(_apply_theme(fig), use_container_width=True)

with tab_quality:
    if features is not None:
        q1, q2 = st.columns(2)
        with q1:
            missing = features.isna().sum().sort_values(ascending=False)
            missing_top = missing[missing > 0].head(15)
            if len(missing_top) > 0:
                fig = go.Figure(go.Bar(
                    x=missing_top.values, y=missing_top.index, orientation="h",
                    marker=dict(color=NEON_YELLOW, opacity=0.85, line=dict(color="rgba(123,97,255,0.12)", width=0.5)),
                    text=[f"{v:,}" for v in missing_top.values], textposition="auto",
                    textfont=dict(color="#f0ecff"),
                ))
                fig.update_layout(title="<b>Missing Values by Feature</b>", height=max(350, len(missing_top) * 30))
                st.plotly_chart(_apply_theme(fig), use_container_width=True)
            else:
                st.success("No missing values in feature matrix.")

        with q2:
            variances = features.select_dtypes(include="number").var().sort_values()
            low_var = variances.head(10)
            fig = go.Figure(go.Bar(
                x=low_var.values, y=low_var.index, orientation="h",
                marker=dict(color=NEON_PINK, opacity=0.85, line=dict(color="rgba(123,97,255,0.12)", width=0.5)),
            ))
            fig.update_layout(title="<b>Lowest Variance Features</b>", height=350)
            st.plotly_chart(_apply_theme(fig), use_container_width=True)

        st.markdown("")
        section("Raw Data Files")
        raw_files = sorted(RAW_DIR.glob("*.csv"))
        if raw_files:
            file_data = [{"File": f.name, "Size (MB)": round(f.stat().st_size / (1024 * 1024), 2)} for f in raw_files]
            st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)
    else:
        empty_state("Feature matrix not found", "Run: <code>make features</code>")
