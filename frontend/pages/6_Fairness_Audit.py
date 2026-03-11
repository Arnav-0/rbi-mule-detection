"""Page 6: Fairness Audit — bias analysis across sensitive features."""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.components.charts import (
    _apply_theme, MULE_COLOR, LEGIT_COLOR, ACCENT_COLOR,
    NEON_PURPLE, NEON_PINK, NEON_CYAN, NEON_MAGENTA, NEON_YELLOW, NEON_BLUE, PALETTE,
    _hex_to_rgba,
)
from frontend.components.tables import format_fairness_table, highlight_fairness
from frontend.components.layout import page_header, section, empty_state, info_callout, pipeline_flow

st.set_page_config(page_title="Fairness Audit | RBI Mule Detection", page_icon="⚖", layout="wide")
page_header(
    "Fairness Audit",
    "Ensure the model treats all demographic groups equitably — a requirement for responsible AI deployment"
)

pipeline_flow([
    ("🎯", "Predictions", "Model output", "purple"),
    ("👥", "Group Rates", "By demographic", "cyan"),
    ("📏", "DP & EO Tests", "Statistical tests", "magenta"),
    ("⚖️", "80% Rule", "Compliance check", "pink"),
    ("📋", "Report", "You are here", "yellow"),
], highlight=4)

REPORTS_DIR = Path("outputs/reports")


@st.cache_data
def load_fairness_reports():
    reports = []
    for f in REPORTS_DIR.glob("fairness_*.json"):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            if isinstance(data, list):
                reports.extend(data)
            else:
                reports.append(data)
        except json.JSONDecodeError:
            continue

    main_path = REPORTS_DIR / "fairness_report.json"
    if main_path.exists() and not reports:
        try:
            data = json.loads(main_path.read_text(encoding='utf-8'))
            if isinstance(data, list):
                reports = data
            elif isinstance(data, dict):
                if "reports" in data:
                    reports = data["reports"]
                else:
                    reports = [data]
        except json.JSONDecodeError:
            pass
    return reports


reports = load_fairness_reports()

if reports:
    sensitive_features = sorted(set(r.get("sensitive_feature", "unknown") for r in reports))

    with st.sidebar:
        st.markdown("### Audit Controls")
        selected_feature = st.selectbox("Sensitive Feature", sensitive_features)
        st.markdown("")
        st.markdown(
            "<div style='color:#7b6cbf; font-size:0.78rem; line-height:1.6;'>"
            "Select a sensitive attribute to see whether the model's predictions "
            "are fair across its groups."
            "</div>",
            unsafe_allow_html=True,
        )

    filtered = [r for r in reports if r.get("sensitive_feature") == selected_feature]

    if filtered:
        info_callout(
            "What is fairness auditing?",
            "We check if the model flags mule accounts at similar rates across demographic groups. "
            "The 80% rule requires that the lowest-flagged group is within 80% of the highest-flagged group. "
            "Demographic parity means equal positive prediction rates; equalized odds means equal error rates."
        )

        # Summary metrics
        passing = sum(1 for r in filtered if r.get("pass_80_rule"))
        total = len(filtered)
        overall = "PASS" if passing == total else "FAIL"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Audits", total)
        with c2:
            st.metric("Passing 80% Rule", f"{passing}/{total}")
        with c3:
            st.metric("Overall Status", overall,
                      delta="Compliant" if overall == "PASS" else "Action Required",
                      delta_color="normal" if overall == "PASS" else "inverse")
        with c4:
            avg_dp = np.mean([r.get("demographic_parity_diff", 0) for r in filtered])
            st.metric("Avg DP Diff", f"{avg_dp:.4f}")

        st.markdown("")

        # Tabs
        tab_table, tab_viz, tab_recs = st.tabs(["Detailed Results", "Visualizations", "Recommendations"])

        with tab_table:
            fair_df = format_fairness_table(filtered)
            st.dataframe(highlight_fairness(fair_df), use_container_width=True)

        with tab_viz:
            col_gap, col_group = st.columns(2)

            with col_gap:
                section("Fairness Gap Analysis")
                dp_diffs = [r.get("demographic_parity_diff", 0) for r in filtered]
                eo_diffs = [r.get("equalized_odds_diff", 0) for r in filtered]
                gap_labels = [r.get("sensitive_feature", "?") for r in filtered]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=gap_labels, y=dp_diffs, name="Demographic Parity Diff",
                    marker=dict(color=NEON_PURPLE, line=dict(color="rgba(123,97,255,0.12)", width=1)),
                    hovertemplate="%{x}<br>DP Diff: %{y:.4f}<extra></extra>",
                ))
                fig.add_trace(go.Bar(
                    x=gap_labels, y=eo_diffs, name="Equalized Odds Diff",
                    marker=dict(color=NEON_MAGENTA, line=dict(color="rgba(123,97,255,0.12)", width=1)),
                    hovertemplate="%{x}<br>EO Diff: %{y:.4f}<extra></extra>",
                ))
                fig.add_hline(y=0.1, line_dash="dash", line_color=NEON_YELLOW,
                              annotation_text="Threshold (0.10)",
                              annotation_font_color=NEON_YELLOW)
                fig.update_layout(
                    title="<b>Fairness Gap Analysis</b>",
                    barmode="group",
                    yaxis_title="Difference",
                    height=420,
                )
                st.plotly_chart(_apply_theme(fig), use_container_width=True)

            with col_group:
                section("Metrics by Demographic Group")
                details_data = []
                for r in filtered:
                    details = r.get("details", r.get("details_json", {}))
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except json.JSONDecodeError:
                            details = {}
                    if isinstance(details, dict):
                        for group, metrics in details.items():
                            if isinstance(metrics, dict):
                                details_data.append({"Group": str(group), **metrics})

                if details_data:
                    details_df = pd.DataFrame(details_data)
                    metric_cols = [c for c in details_df.columns
                                   if c != "Group" and details_df[c].dtype in ["float64", "int64"]]

                    if metric_cols:
                        fig = go.Figure()
                        for i, col in enumerate(metric_cols[:6]):
                            fig.add_trace(go.Bar(
                                x=details_df["Group"], y=details_df[col],
                                name=col, marker_color=PALETTE[i % len(PALETTE)],
                            ))
                        fig.update_layout(
                            title=f"<b>Metrics by {selected_feature} Group</b>",
                            barmode="group",
                            xaxis_title="Group", yaxis_title="Value",
                            height=420,
                        )
                        st.plotly_chart(_apply_theme(fig), use_container_width=True)
                else:
                    st.info("No per-group detail data available.")

            # 80% Rule gauge
            st.markdown("")
            section("80% Rule Compliance")
            info_callout(
                "The 80% rule",
                "The minimum group's positive prediction rate must be at least 80% of the "
                "maximum group's rate. The gauge shows how close we are — 80%+ is passing."
            )
            rule_cols = st.columns(min(4, total))
            for i, r in enumerate(filtered[:4]):
                with rule_cols[i]:
                    passes = r.get("pass_80_rule", False)
                    dp = r.get("demographic_parity_diff", 0)
                    color = NEON_CYAN if passes else NEON_PINK
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=max(0, 1 - dp) * 100,
                        number=dict(suffix="%", font=dict(size=28, color="#f0ecff")),
                        title=dict(text=r.get("sensitive_feature", ""), font=dict(size=13, color="#7b6cbf")),
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor="#4a3d8f"),
                            bar=dict(color=color),
                            bgcolor="rgba(12,10,36,0.8)",
                            bordercolor="rgba(123,97,255,0.12)",
                            threshold=dict(line=dict(color=NEON_YELLOW, width=2), thickness=0.8, value=80),
                        ),
                    ))
                    fig.update_layout(height=220)
                    st.plotly_chart(_apply_theme(fig), use_container_width=True)

        with tab_recs:
            if passing == total:
                st.success(
                    "All fairness audits pass the 80% rule. The model shows acceptable "
                    "demographic parity across the tested sensitive features."
                )
                st.markdown(
                    "**Ongoing best practices:**\n"
                    "- Continue monitoring fairness metrics during model retraining\n"
                    "- Expand to additional sensitive features as data permits\n"
                    "- Consider intersectional fairness analysis"
                )
            else:
                failing = [r for r in filtered if not r.get("pass_80_rule")]
                st.warning("The following audits did not pass the 80% rule:")
                for r in failing:
                    st.markdown(
                        f"- **{r.get('sensitive_feature')}**: "
                        f"DP diff = {r.get('demographic_parity_diff', 0):.4f}, "
                        f"EO diff = {r.get('equalized_odds_diff', 0):.4f}"
                    )
                st.markdown(
                    "**Suggested remediation actions:**\n"
                    "1. Apply fairness-aware threshold tuning per group\n"
                    "2. Consider resampling or reweighting training data\n"
                    "3. Add fairness constraints during model training (e.g., adversarial debiasing)\n"
                    "4. Review feature set for proxy variables that encode sensitive attributes\n"
                    "5. Consider calibrating predictions separately per demographic group"
                )
    else:
        st.info(f"No reports found for feature: {selected_feature}")

else:
    empty_state(
        "No fairness reports found",
        "Run the fairness audit: <code>python -m src.explainability.fairness</code>"
    )

    st.markdown("")
    section("Fairness Metrics Explained")
    info_callout(
        "Why fairness matters",
        "A mule detection model deployed at scale affects real people. If the model disproportionately "
        "flags accounts from certain demographics, it could cause harm and violate regulatory requirements."
    )
    st.markdown(
        "| Metric | What it measures | Passing threshold |\n"
        "|--------|-----------------|-------------------|\n"
        "| **Demographic Parity** | Equal positive prediction rates across groups | Diff < 0.10 |\n"
        "| **Equalized Odds** | Equal true/false positive rates across groups | Diff < 0.10 |\n"
        "| **80% Rule** | Min group rate >= 80% of max group rate | Ratio >= 0.80 |\n"
        "| **Calibration** | Predicted probabilities match true rates per group | Diff < 0.05 |"
    )
