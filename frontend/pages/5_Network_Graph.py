"""Page 5: Network Graph — transaction network visualization with pyvis."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.components.charts import (
    _apply_theme, MULE_COLOR, LEGIT_COLOR, ACCENT_COLOR,
    NEON_PURPLE, NEON_PINK, NEON_CYAN, NEON_BLUE, NEON_MAGENTA,
    _hex_to_rgba,
)
from frontend.components.layout import page_header, section, empty_state, info_callout, neon_legend, pipeline_flow
import plotly.graph_objects as go

st.set_page_config(page_title="Network Graph | RBI Mule Detection", page_icon="🕸", layout="wide")
page_header(
    "Transaction Network Graph",
    "Visualize how accounts are connected through transactions — mule accounts often form distinct network patterns"
)

pipeline_flow([
    ("💳", "Transactions", "Raw CSV files", "cyan"),
    ("🔗", "Edge List", "Account pairs", "purple"),
    ("🕸", "Graph Build", "NetworkX", "magenta"),
    ("📍", "PageRank", "Centrality scores", "pink"),
    ("👁", "Visualize", "You are here", "yellow"),
], highlight=4)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")


@st.cache_data
def load_graph_data():
    """Load transaction data and build edge list."""
    edge_path = PROCESSED_DIR / "transaction_edges.parquet"
    if edge_path.exists():
        return pd.read_parquet(edge_path)

    parts = sorted(RAW_DIR.glob("transactions_part_*.csv"))
    if not parts:
        return None

    edges_list = []
    for p in parts[:2]:
        df = pd.read_csv(p, low_memory=False)
        rename = {"amount": "transaction_amount"}
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        if "account_id" in df.columns and "counterparty_id" in df.columns and "transaction_amount" in df.columns:
            grouped = df.groupby(["account_id", "counterparty_id"]).agg(
                volume=("transaction_amount", "sum"),
                count=("transaction_amount", "count"),
            ).reset_index()
            edges_list.append(grouped)

    if edges_list:
        return pd.concat(edges_list).groupby(["account_id", "counterparty_id"]).sum().reset_index()
    return None


@st.cache_data
def load_labels():
    path = RAW_DIR / "train_labels.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_pagerank():
    for name in ["features_matrix.parquet", "feature_matrix.parquet"]:
        path = PROCESSED_DIR / name
        if path.exists():
            try:
                df = pd.read_parquet(path, columns=["pagerank"])
                df = df.reset_index().rename(columns={"index": "account_id"}) if "account_id" not in df.columns else df
                return df
            except Exception:
                continue
    return None


edges = load_graph_data()
labels_df = load_labels()
pagerank_df = load_pagerank()

if edges is not None:
    info_callout(
        "What you're seeing",
        "Each node is an account. Edges represent transactions between accounts. "
        "Red nodes are known mules, cyan are legitimate. Mule accounts often cluster together "
        "or act as intermediaries with high centrality (many connections)."
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Graph Controls")
        top_n = st.slider("Top N accounts (by PageRank/degree)", 50, 1000, 500, step=50)
        min_weight = st.slider("Min edge weight (count)", 1, 20, 2)
        show_mule_only = st.checkbox("Highlight mule connections only", False)
        st.markdown("")
        st.markdown(
            "<div style='color:#7b6cbf; font-size:0.78rem; line-height:1.6;'>"
            "Increase min edge weight to simplify the graph. "
            "Toggle 'mule connections only' to focus on suspicious patterns."
            "</div>",
            unsafe_allow_html=True,
        )

    # Get top accounts
    if pagerank_df is not None and len(pagerank_df) > 0:
        top_accounts = set(pagerank_df.nlargest(top_n, "pagerank")["account_id"])
    else:
        degree = edges["account_id"].value_counts()
        top_accounts = set(degree.head(top_n).index)

    # Filter edges
    filtered = edges[
        (edges["account_id"].isin(top_accounts)) &
        (edges["count"] >= min_weight)
    ].copy()

    mule_ids = set()
    if labels_df is not None:
        mule_ids = set(labels_df[labels_df["is_mule"] == 1]["account_id"])

    if show_mule_only and mule_ids:
        filtered = filtered[
            (filtered["account_id"].isin(mule_ids)) |
            (filtered["counterparty_id"].isin(mule_ids))
        ]

    test_ids = set()
    for test_dir in [RAW_DIR, PROCESSED_DIR.parent / "raw"]:
        test_path = test_dir / "test_accounts.csv"
        if test_path.exists():
            test_ids = set(pd.read_csv(test_path)["account_id"])
            break

    # Stats
    all_nodes = set(filtered["account_id"]) | set(filtered["counterparty_id"])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Nodes", f"{len(all_nodes):,}")
    with c2:
        st.metric("Edges", f"{len(filtered):,}")
    with c3:
        st.metric("Mule Nodes", f"{len(all_nodes & mule_ids):,}")
    with c4:
        st.metric("Min Edge Weight", f"{min_weight}")

    st.markdown("")

    if len(filtered) > 0 and len(filtered) < 5000:
        try:
            from pyvis.network import Network

            net = Network(height="700px", width="100%", bgcolor="#020209",
                          font_color="#b8aae8", directed=True)
            net.barnes_hut(gravity=-3000, central_gravity=0.3,
                           spring_length=150, spring_strength=0.01)

            for node_id in all_nodes:
                if node_id in mule_ids:
                    color = NEON_PINK
                    title = f"MULE: {node_id}"
                    size = 18
                elif node_id in test_ids:
                    color = "#4f4280"
                    title = f"TEST: {node_id}"
                    size = 12
                else:
                    color = NEON_CYAN
                    title = f"LEGIT: {node_id}"
                    size = 10
                net.add_node(str(node_id), label="", title=title, color=color, size=size)

            max_vol = filtered["volume"].max() if filtered["volume"].max() > 0 else 1
            for _, row in filtered.iterrows():
                width = max(0.5, 4 * row["volume"] / max_vol)
                net.add_edge(
                    str(row["account_id"]), str(row["counterparty_id"]),
                    value=width,
                    title=f"Volume: {row['volume']:,.0f} | Count: {row['count']}",
                    color=_hex_to_rgba(NEON_PURPLE, 0.4),
                )

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                net.save_graph(f.name)
                html_content = Path(f.name).read_text(encoding='utf-8')

            components.html(html_content, height=720, scrolling=False)

            neon_legend([
                (NEON_PINK, "Mule"),
                (NEON_CYAN, "Legitimate"),
                ("#4f4280", "Test/Unknown"),
            ])
            st.markdown(
                "<div style='text-align:center; color:#7b6cbf; font-size:0.78rem;'>"
                "Edge thickness = transaction volume between accounts"
                "</div>",
                unsafe_allow_html=True,
            )

        except ImportError:
            st.error("pyvis not installed. Run: `pip install pyvis`")
        except Exception as e:
            st.error(f"Error rendering network graph: {e}")

    elif len(filtered) >= 5000:
        st.warning(
            f"Too many edges ({len(filtered):,}) to render interactively. "
            "Try increasing the min edge weight or reducing the number of accounts."
        )

        section("Top Connected Accounts")
        degree_df = pd.concat([
            filtered["account_id"].value_counts().rename("out_degree"),
            filtered["counterparty_id"].value_counts().rename("in_degree"),
        ], axis=1).fillna(0).astype(int)
        degree_df["total_degree"] = degree_df["out_degree"] + degree_df["in_degree"]
        degree_df["is_mule"] = degree_df.index.isin(mule_ids).astype(int)
        degree_df = degree_df.sort_values("total_degree", ascending=False).head(50)
        st.dataframe(degree_df, use_container_width=True)

        section("Degree Distribution")
        fig = go.Figure(go.Histogram(
            x=degree_df["total_degree"],
            marker=dict(color=ACCENT_COLOR, opacity=0.85,
                        line=dict(color="rgba(123,97,255,0.12)", width=0.5)),
            nbinsx=30,
        ))
        fig.update_layout(
            title="<b>Node Degree Distribution</b>",
            xaxis_title="Total Degree", yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(_apply_theme(fig), use_container_width=True)
    else:
        st.info("No edges match the current filters. Try adjusting the controls.")

else:
    empty_state(
        "No transaction data found for graph building",
        "Expected: <code>data/raw/transactions_part_*.csv</code> or <code>data/processed/transaction_edges.parquet</code>"
    )
