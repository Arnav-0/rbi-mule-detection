"""RBI Mule Account Detection Dashboard."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(
    page_title="RBI Mule Detection",
    page_icon="https://em-content.zobj.net/source/twitter/408/detective_1f575-fe0f.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def get_live_stats():
    processed = PROJECT_ROOT / "data" / "processed"
    raw = PROJECT_ROOT / "data" / "raw"
    stats = {"accounts": "40,038", "features": "56", "models": "6",
             "transactions": "7.4M", "mule_rate": "1.09%", "auc": "N/A",
             "best_model": "N/A", "validation": "5-fold CV"}
    for name in ["features_matrix.parquet", "feature_matrix.parquet"]:
        path = processed / name
        if path.exists():
            df = pd.read_parquet(path)
            stats["accounts"] = f"{len(df):,}"
            stats["features"] = str(df.shape[1])
            break
    labels_path = raw / "train_labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        rate = labels["is_mule"].mean() * 100
        stats["mule_rate"] = f"{rate:.2f}%"
    import json
    bench_path = PROJECT_ROOT / "outputs" / "reports" / "benchmark_results.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding='utf-8'))
        if isinstance(bench, dict) and bench:
            best_name = max(bench.keys(),
                           key=lambda k: bench[k].get("metrics", {}).get("auc_roc", 0))
            best_metrics = bench[best_name].get("metrics", {})
            auc = best_metrics.get("auc_roc", 0)
            std = bench[best_name].get("cv_std", {}).get("auc_roc", 0)
            if std > 0:
                stats["auc"] = f"{auc:.4f} +/- {std:.4f}"
            else:
                stats["auc"] = f"{auc:.4f}"
            stats["best_model"] = best_name
    return stats


live = get_live_stats()

# ── Hero ──
st.markdown(
    '<div style="text-align:center; padding:40px 0 8px;">'
    '<div style="font-size:0.7rem; font-weight:700; letter-spacing:0.2em; text-transform:uppercase;'
    ' color:#7b61ff; margin-bottom:16px; text-shadow:0 0 18px rgba(123,97,255,0.5);">'
    'RESERVE BANK OF INDIA INNOVATION HUB</div>'
    '<div style="font-size:2.8rem; font-weight:900; letter-spacing:-0.04em; line-height:1.1;'
    ' margin-bottom:14px; background:linear-gradient(135deg,#d4c8ff,#7b61ff,#e040fb,#00f5d4);'
    ' background-size:300% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent;'
    ' animation:shimmer 4s linear infinite; filter:drop-shadow(0 0 25px rgba(123,97,255,0.3));">'
    'Mule Account Detection</div>'
    '<div class="hero-subtitle">'
    'AI-powered detection of money mule accounts with explainable predictions,'
    ' fairness auditing, and real-time API serving</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Tech Stack ──
techs = ["LightGBM", "XGBoost", "CatBoost", "SHAP", "NetworkX", "FastAPI", "Streamlit", "Plotly"]
st.markdown(
    '<div style="display:flex; flex-wrap:wrap; gap:6px; justify-content:center; padding:16px 0 8px;">'
    + "".join(f'<span class="tech-stack-item">{t}</span>' for t in techs)
    + '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Key Stats ──
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Accounts", live["accounts"])
with c2:
    st.metric("Features", live["features"])
with c3:
    st.metric("Models", live["models"])
with c4:
    st.metric("Transactions", live["transactions"])
with c5:
    st.metric("Mule Rate", live["mule_rate"])
with c6:
    st.metric("Best AUC", live["auc"])

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Pipeline ──
st.markdown("## End-to-End Pipeline")
st.markdown("")

_steps = [
    ("1", "Raw Data", "10 CSV files", False),
    ("2", "Features", f"{live['features']} engineered", False),
    ("3", "Model", f"Best: {live['best_model']}", True),
    ("4", "Predictions", "16K test accounts", False),
    ("5", "Explain", "SHAP + Fairness", False),
]
_icons = ["&#128193;", "&#9881;", "&#129504;", "&#127919;", "&#128269;"]

pipeline_html = []
for idx, (num, title, sub, active) in enumerate(_steps):
    cls = "pipeline-step active" if active else "pipeline-step"
    pipeline_html.append(
        f'<div class="{cls}" style="flex:1; min-width:120px; max-width:200px;">'
        f'<div class="pipeline-step-num">{num}</div>'
        f'<div class="pipeline-step-icon">{_icons[idx]}</div>'
        f'<div class="pipeline-step-title">{title}</div>'
        f'<div class="pipeline-step-sub">{sub}</div>'
        f'</div>'
    )

arrow = '<div class="pipeline-arrow">&#10132;</div>'
joined = arrow.join(pipeline_html)

st.markdown(
    '<div style="display:flex; align-items:center; justify-content:center; gap:0;'
    ' flex-wrap:wrap; padding:8px 0 16px;">' + joined + '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Navigation Cards ──
st.markdown("## Dashboard Pages")
st.markdown("")

row1 = st.columns(4)
pages = [
    ("&#128202;", "Overview", "Dataset statistics, distributions, transaction volume trends"),
    ("&#128300;", "Feature Explorer", f"{live['features']} features across 8 groups with SHAP importance"),
    ("&#127942;", "Model Comparison", "ROC/PR curves, confusion matrices, CV benchmarks"),
    ("&#129504;", "Explainability", "Global SHAP, per-account waterfall, beeswarm plots"),
]
for i, (icon, title, desc) in enumerate(pages):
    with row1[i]:
        st.markdown(
            f'<div class="nav-card"><span class="nav-card-icon">{icon}</span>'
            f'<div class="nav-card-title">{title}</div>'
            f'<div class="nav-card-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("")

row2 = st.columns(4)
pages2 = [
    ("&#128376;", "Network Graph", "Interactive transaction network with PageRank centrality"),
    ("&#9878;", "Fairness Audit", "Demographic parity, equalized odds, 80% rule"),
    ("&#128270;", "Account Inspector", "Deep-dive: profile, timeline, SHAP, heatmaps"),
    ("&#128640;", "API Demo", "Live prediction interface with batch support"),
]
for i, (icon, title, desc) in enumerate(pages2):
    with row2[i]:
        st.markdown(
            f'<div class="nav-card"><span class="nav-card-icon">{icon}</span>'
            f'<div class="nav-card-title">{title}</div>'
            f'<div class="nav-card-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── System Architecture ──
st.markdown("## System Architecture")
st.markdown("")

_arch = [
    ("&#128229;", "Data Pipeline", "10 raw CSVs &rarr; schema validation &rarr; merge &rarr; clean"),
    ("&#9881;", "Feature Engineering", f"8 generators &rarr; {live['features']} features &rarr; registry &rarr; parquet"),
    ("&#129504;", "Model Training", f"6 models &rarr; Optuna + 5-fold CV &rarr; {live['best_model']} best"),
    ("&#128300;", "Explainability", "SHAP TreeExplainer &rarr; per-account &rarr; NL text"),
    ("&#9201;", "Temporal Detection", "Z-score anomaly &rarr; suspicious windows &rarr; IoU"),
    ("&#128640;", "Serving", "FastAPI 13 endpoints &rarr; SQLite &rarr; Dashboard"),
]

arch_html = []
for icon, title, desc in _arch:
    arch_html.append(
        f'<div class="arch-card">'
        f'<div class="arch-card-icon">{icon}</div>'
        f'<div class="arch-card-title">{title}</div>'
        f'<div class="arch-card-desc">{desc}</div>'
        f'</div>'
    )

st.markdown(
    '<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; padding:0 0 12px;">'
    + "".join(arch_html) + '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Quick Test ──
st.markdown("## Quick Test")

tab_existing, tab_upload = st.tabs(["Existing Account", "New Data (CSV Upload)"])

with tab_existing:
    st.caption("Enter any account ID from the dataset to get an instant mule risk prediction.")
    qt_col1, qt_col2 = st.columns([1, 2])

    with qt_col1:
        qt_account = st.text_input("Account ID", value="", placeholder="e.g. ACCT_000077", key="qt_acct")
        qt_go = st.button("Predict", type="primary", use_container_width=True, key="qt_btn")

    with qt_col2:
        if qt_go and qt_account:
            try:
                import joblib
                model = joblib.load(str(PROJECT_ROOT / "outputs" / "models" / "best_model.joblib"))
                feat_df = pd.read_parquet(str(PROJECT_ROOT / "data" / "processed" / "features_matrix.parquet"))
                if qt_account in feat_df.index:
                    x = feat_df.loc[[qt_account]]
                    prob = float(model.predict_proba(x)[:, 1][0])
                    risk = "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.4 else "LOW")
                    label = "MULE" if prob >= 0.5 else "LEGITIMATE"
                    color = "#ff4d6d" if prob >= 0.5 else "#00f5d4"

                    st.markdown(
                        f'<div style="text-align:center; padding:8px 0;">'
                        f'<span style="font-size:1.8rem; font-weight:900; color:{color}; '
                        f'letter-spacing:0.03em; text-shadow:0 0 24px {color};">{label}</span></div>',
                        unsafe_allow_html=True,
                    )

                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric("Probability", f"{prob:.4f}")
                    with r2:
                        st.metric("Prediction", label)
                    with r3:
                        st.metric("Risk Level", risk)

                    importances = model.feature_importances_
                    feat_vals = x.iloc[0]
                    top_idx = importances.argsort()[::-1][:5]
                    st.markdown("**Top features driving this prediction:**")
                    for i in top_idx:
                        fname = feat_df.columns[i]
                        fval = feat_vals.iloc[i]
                        st.markdown(f"- **{fname}** = {fval:.4f}")
                else:
                    st.warning(f"Account `{qt_account}` not found in feature matrix.")
            except Exception as e:
                st.error(f"Error: {e}")
        elif qt_go:
            st.warning("Enter an account ID first.")

with tab_upload:
    st.caption("Upload a CSV of transactions for an account not in the dataset. "
               "Features will be computed in real time.")
    csv_file = st.file_uploader("Transaction CSV", type=["csv"], key="qt_csv")
    csv_col1, csv_col2 = st.columns([1, 2])
    with csv_col1:
        csv_acct_id = st.text_input("Account ID", value="NEW_ACCOUNT", key="csv_acct_id")
        csv_open_date = st.text_input("Account Opening Date (optional)", value="", key="csv_open_date",
                                       placeholder="YYYY-MM-DD")
        csv_balance = st.number_input("Avg Balance (optional)", value=0.0, key="csv_balance")
        csv_go = st.button("Compute & Predict", type="primary", use_container_width=True, key="csv_btn")
    with csv_col2:
        if csv_go and csv_file is not None:
            try:
                import joblib
                from src.features.realtime import compute_features_realtime
                txn_df = pd.read_csv(csv_file)
                features = compute_features_realtime(
                    txn_df,
                    account_id=csv_acct_id,
                    account_opening_date=csv_open_date if csv_open_date else None,
                    avg_balance=csv_balance,
                )
                model = joblib.load(str(PROJECT_ROOT / "outputs" / "models" / "best_model.joblib"))
                prob = float(model.predict_proba(features)[:, 1][0])
                risk = "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.4 else "LOW")
                label = "MULE" if prob >= 0.5 else "LEGITIMATE"
                color = "#ff4d6d" if prob >= 0.5 else "#00f5d4"

                st.markdown(
                    f'<div style="text-align:center; padding:8px 0;">'
                    f'<span style="font-size:1.8rem; font-weight:900; color:{color}; '
                    f'letter-spacing:0.03em; text-shadow:0 0 24px {color};">{label}</span></div>',
                    unsafe_allow_html=True,
                )

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Probability", f"{prob:.4f}")
                with r2:
                    st.metric("Prediction", label)
                with r3:
                    st.metric("Risk Level", risk)

                st.markdown(f"**Computed {features.shape[1]} features from {len(txn_df)} transactions.**")
                importances = model.feature_importances_
                feat_vals = features.iloc[0]
                top_idx = importances.argsort()[::-1][:5]
                st.markdown("**Top features driving this prediction:**")
                for i in top_idx:
                    fname = features.columns[i]
                    fval = feat_vals.iloc[i]
                    st.markdown(f"- **{fname}** = {fval:.4f}")
            except Exception as e:
                st.error(f"Error: {e}")
        elif csv_go:
            st.warning("Upload a transaction CSV first.")

# ── Sidebar ──
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding:20px 0 10px;">'
        '<div style="font-size:1.1rem; font-weight:900; letter-spacing:0.08em;'
        ' font-family:Orbitron,Inter,system-ui,sans-serif;'
        ' background:linear-gradient(135deg,#7b61ff,#e040fb,#00f5d4);'
        ' -webkit-background-clip:text; -webkit-text-fill-color:transparent;">'
        'RBI MULE DETECTION</div>'
        '<div style="color:#4f4280; font-size:0.72rem; margin-top:6px;'
        ' letter-spacing:0.1em; text-transform:uppercase; font-weight:600;">'
        'v1.0 &mdash; Production ML System</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
**Model:** {live['best_model']}
**AUC:** {live['auc']}
**Features:** {live['features']} engineered
**Validation:** {live['validation']}
**Explain:** SHAP TreeExplainer
**API:** FastAPI + SQLite
    """)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption("Reserve Bank of India Innovation Hub")
