"""Reusable Plotly chart components — Neon Futuristic Theme."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Color Palette — vivid neon futuristic ──
MULE_COLOR = "#ff4d6d"
LEGIT_COLOR = "#00f5d4"
ACCENT_COLOR = "#7b61ff"
WARN_COLOR = "#ffe66d"

NEON_PURPLE = "#7b61ff"
NEON_PINK = "#ff4d6d"
NEON_CYAN = "#00f5d4"
NEON_BLUE = "#00bbf9"
NEON_YELLOW = "#ffe66d"
NEON_ORANGE = "#ff9e3d"
NEON_MINT = "#00f5d4"
NEON_MAGENTA = "#e040fb"
NEON_LIME = "#b8ff3d"
NEON_CORAL = "#ff6b8a"

PALETTE = [ACCENT_COLOR, NEON_PINK, NEON_CYAN, NEON_BLUE, NEON_YELLOW,
           NEON_ORANGE, NEON_MAGENTA, NEON_LIME, NEON_CORAL, NEON_MINT]

BG_DARK = "rgba(2,2,9,0)"
BG_PLOT = "rgba(12,10,36,0.6)"
GRID_COLOR = "rgba(123,97,255,0.06)"
GRID_ZERO = "rgba(123,97,255,0.12)"

DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PLOT,
        font=dict(color="#b8aae8", family="Inter, system-ui, sans-serif", size=12),
        title=dict(
            font=dict(size=16, color="#f0ecff", family="Inter, system-ui, sans-serif"),
            x=0.02, xanchor="left", y=0.97,
        ),
        legend=dict(
            bgcolor="rgba(12,10,36,0.8)", font=dict(color="#b8aae8", size=11),
            borderwidth=0,
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_ZERO,
            tickfont=dict(size=11, color="#7b6cbf"),
            linecolor="rgba(123,97,255,0.12)",
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_ZERO,
            tickfont=dict(size=11, color="#7b6cbf"),
            linecolor="rgba(123,97,255,0.12)",
        ),
        colorway=PALETTE,
        margin=dict(l=55, r=30, t=60, b=45),
        hoverlabel=dict(
            bgcolor="rgba(12,10,36,0.95)",
            bordercolor="rgba(123,97,255,0.3)",
            font=dict(color="#f0ecff", size=12, family="Inter, system-ui, sans-serif"),
        ),
    )
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**DARK_TEMPLATE["layout"])
    return fig


def _neon_glow_line(color: str, width: float = 2.5):
    return dict(color=color, width=width)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_roc_curves(results: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="rgba(123,97,255,0.15)", dash="dot", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    for i, (name, data) in enumerate(results.items()):
        color = PALETTE[i % len(PALETTE)]
        auc_val = data.get("auc", 0)
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            name=f"{name} (AUC={auc_val:.4f})",
            mode="lines",
            line=_neon_glow_line(color, 2.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.06),
            hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title="<b>ROC Curves</b>",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1], constrain="domain"),
        yaxis=dict(range=[0, 1.02], scaleanchor="x", scaleratio=1),
        height=480,
    )
    return _apply_theme(fig)


def plot_pr_curves(results: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    for i, (name, data) in enumerate(results.items()):
        color = PALETTE[i % len(PALETTE)]
        auc_val = data.get("auc_pr", 0)
        fig.add_trace(go.Scatter(
            x=data["recall"], y=data["precision"],
            name=f"{name} (AP={auc_val:.4f})",
            mode="lines",
            line=_neon_glow_line(color, 2.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.07),
            hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title="<b>Precision-Recall Curves</b>",
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
        height=480,
    )
    return _apply_theme(fig)


def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix") -> go.Figure:
    if labels is None:
        labels = ["Legitimate", "Mule"]
    total = cm.sum()
    text = [[f"<b style='font-size:20px'>{v:,}</b><br><span style='font-size:11px;color:#b8aae8'>{v/total*100:.1f}%</span>"
             for v in row] for row in cm]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels, text=text,
        texttemplate="%{text}",
        colorscale=[
            [0, "#020209"],
            [0.2, "#0a081e"],
            [0.4, "#3a1f8e"],
            [0.7, "#7b61ff"],
            [1, "#e040fb"],
        ],
        showscale=False,
        hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: <b>%{z:,}</b><extra></extra>",
    ))
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Predicted", yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
        height=380,
    )
    return _apply_theme(fig)


def plot_feature_importance(names: list, values: list, top_n: int = 20,
                            title="Feature Importance") -> go.Figure:
    idx = np.argsort(values)[-top_n:]
    sorted_names = [names[i] for i in idx]
    sorted_vals = [values[i] for i in idx]

    n = len(sorted_vals)
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r = int(79 + t * 44)
        g = int(61 + t * 36)
        b = int(180 + t * 75)
        colors.append(f"rgb({r},{g},{b})")

    fig = go.Figure(go.Bar(
        x=sorted_vals, y=sorted_names, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(123,97,255,0.08)", width=0.5)),
        text=[f"{v:.4f}" for v in sorted_vals],
        textposition="auto",
        textfont=dict(color="#f0ecff", size=10),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"<b>{title}</b>", xaxis_title="Importance",
        height=max(420, top_n * 30),
    )
    return _apply_theme(fig)


def plot_distribution_comparison(legit_values, mule_values,
                                 feature_name: str, log_scale: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=legit_values, name="Legitimate", opacity=0.6,
        marker=dict(color=LEGIT_COLOR, line=dict(color=_hex_to_rgba(LEGIT_COLOR, 0.5), width=1)),
        nbinsx=50,
        hovertemplate="<b>Legitimate</b><br>Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=mule_values, name="Mule", opacity=0.65,
        marker=dict(color=MULE_COLOR, line=dict(color=_hex_to_rgba(MULE_COLOR, 0.5), width=1)),
        nbinsx=50,
        hovertemplate="<b>Mule</b><br>Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=f"<b>Distribution: {feature_name}</b>",
        xaxis_title=feature_name, yaxis_title="Count",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if log_scale:
        fig.update_xaxes(type="log")
    return _apply_theme(fig)


def plot_timeline(dates, amounts, colors=None, title="Transaction Timeline") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=dates, y=amounts, mode="markers",
        marker=dict(
            size=6,
            color=colors if colors is not None else NEON_BLUE,
            opacity=0.75,
            line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
        ),
        hovertemplate="%{x}<br>Amount: %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(title=f"<b>{title}</b>", xaxis_title="Date", yaxis_title="Amount")
    return _apply_theme(fig)


def plot_calibration(y_true, y_prob, n_bins: int = 10, title="Calibration Plot") -> go.Figure:
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="rgba(123,97,255,0.2)", dash="dot", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true, mode="lines+markers",
        name="Calibration",
        line=dict(color=ACCENT_COLOR, width=2.5),
        marker=dict(size=9, color=ACCENT_COLOR, line=dict(width=2, color="#020209")),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(ACCENT_COLOR, 0.06),
        hovertemplate="Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
        height=420,
    )
    return _apply_theme(fig)


def plot_gauge(value: float, title: str = "Risk Score") -> go.Figure:
    if value > 0.7:
        bar_color = MULE_COLOR
    elif value > 0.4:
        bar_color = WARN_COLOR
    else:
        bar_color = LEGIT_COLOR

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number=dict(suffix="%", font=dict(size=42, color="#f0ecff", family="Inter, system-ui")),
        title=dict(text=title, font=dict(size=13, color="#7b6cbf")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#4f4280", tickwidth=1,
                      tickfont=dict(size=10, color="#7b6cbf")),
            bar=dict(color=bar_color, thickness=0.8),
            bgcolor="rgba(12,10,36,0.8)",
            bordercolor="rgba(123,97,255,0.15)",
            borderwidth=1,
            steps=[
                dict(range=[0, 40], color="rgba(0,245,212,0.08)"),
                dict(range=[40, 70], color="rgba(255,230,109,0.08)"),
                dict(range=[70, 100], color="rgba(255,77,109,0.1)"),
            ],
            threshold=dict(
                line=dict(color="#b8aae8", width=2),
                thickness=0.85, value=50,
            ),
        ),
    ))
    fig.update_layout(height=290)
    return _apply_theme(fig)


def plot_correlation_heatmap(corr_matrix, feature_names: list,
                             title="Feature Correlation") -> go.Figure:
    text = [[f"{corr_matrix[i][j]:.2f}" for j in range(len(feature_names))]
            for i in range(len(feature_names))]
    fig = go.Figure(go.Heatmap(
        z=corr_matrix, x=feature_names, y=feature_names,
        text=text, texttemplate="%{text}",
        textfont=dict(size=8, color="#b8aae8"),
        colorscale=[
            [0, "#ff4d6d"],
            [0.25, "#3a1030"],
            [0.5, "#020209"],
            [0.75, "#0a2850"],
            [1, "#00bbf9"],
        ],
        zmin=-1, zmax=1,
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"<b>{title}</b>",
        height=max(520, len(feature_names) * 28),
    )
    return _apply_theme(fig)


def plot_amount_hour_heatmap(hours, amounts_binned, counts,
                             title="Transaction Amount x Hour Heatmap") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=counts, x=hours, y=amounts_binned,
        colorscale=[
            [0, "#020209"],
            [0.3, "#0a081e"],
            [0.5, "#3a1f8e"],
            [0.8, "#7b61ff"],
            [1, "#e040fb"],
        ],
        hovertemplate="Hour: <b>%{x}</b><br>Amount: <b>%{y}</b><br>Count: <b>%{z}</b><extra></extra>",
    ))
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Hour of Day", yaxis_title="Amount Bin",
    )
    return _apply_theme(fig)


def plot_radar(categories: list, values_dict: dict[str, list],
               title: str = "Performance Radar") -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(values_dict.items()):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=_hex_to_rgba(color, 0.1),
            line=dict(color=color, width=2.5),
            name=name,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COLOR,
                            linecolor="rgba(123,97,255,0.08)"),
            bgcolor="rgba(12,10,36,0.6)",
            angularaxis=dict(gridcolor=GRID_COLOR, linecolor="rgba(123,97,255,0.08)"),
        ),
        title=f"<b>{title}</b>",
        height=500,
    )
    return _apply_theme(fig)


def plot_waterfall(names: list, values: list, title: str = "SHAP Waterfall") -> go.Figure:
    colors = [MULE_COLOR if v > 0 else NEON_BLUE for v in values]
    borders = [_hex_to_rgba(MULE_COLOR, 0.4) if v > 0 else _hex_to_rgba(NEON_BLUE, 0.4) for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colors, line=dict(color=borders, width=1)),
        text=[f"{v:+.4f}" for v in values],
        textposition="auto",
        textfont=dict(color="#f0ecff", size=11),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="SHAP Value (positive = higher mule risk)",
        height=max(420, len(names) * 34),
        yaxis=dict(autorange="reversed"),
    )
    return _apply_theme(fig)
