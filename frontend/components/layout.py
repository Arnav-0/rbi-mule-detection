"""Shared layout helpers — Neon Futuristic Theme."""

import streamlit as st
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "style.css"


def inject_css():
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    inject_css()
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(
            f'<div style="color:#7b6cbf; font-size:0.92rem; margin-bottom:4px; line-height:1.55;">{subtitle}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def section(title: str, icon: str = ""):
    prefix = f"{icon} " if icon else ""
    st.markdown(f"### {prefix}{title}")


def subsection(title: str):
    st.markdown(
        f'<div style="color:#b8aae8; font-weight:600; font-size:0.95rem; '
        f'margin:14px 0 6px;">{title}</div>',
        unsafe_allow_html=True,
    )


def glass_metric(label: str, value, delta: str = None):
    st.metric(label, value, delta=delta)


def status_badge(text: str, variant: str = "default"):
    css_class = {
        "mule": "badge-mule",
        "legit": "badge-legit",
        "warning": "badge-warning",
    }.get(variant, "badge-legit")
    st.markdown(f'<span class="{css_class}">{text}</span>', unsafe_allow_html=True)


def empty_state(message: str, hint: str = ""):
    st.markdown(
        f'<div style="text-align:center; padding:50px 20px; color:#4f4280;">'
        f'<div style="font-size:2.5rem; margin-bottom:14px; opacity:0.3;">&#128237;</div>'
        f'<div style="font-size:1rem; font-weight:600; color:#b8aae8; margin-bottom:6px;">{message}</div>'
        f'<div style="font-size:0.85rem; line-height:1.6;">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def nav_card(icon: str, title: str, desc: str):
    st.markdown(
        f'<div class="nav-card"><span class="nav-card-icon">{icon}</span>'
        f'<div class="nav-card-title">{title}</div>'
        f'<div class="nav-card-desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )


def tech_stack_badges(items: list[str]):
    badges = " ".join(f'<span class="tech-stack-item">{item}</span>' for item in items)
    st.markdown(
        f'<div style="display:flex; flex-wrap:wrap; gap:6px; justify-content:center;">{badges}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(metrics: list[tuple]):
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        with cols[i]:
            if len(m) == 3:
                st.metric(m[0], m[1], delta=m[2])
            else:
                st.metric(m[0], m[1])


def neon_legend(items: list[tuple]):
    """Render a centered color legend. items: list of (color, label)."""
    parts = []
    for color, label in items:
        parts.append(
            f"<span style='display:inline-flex; align-items:center; gap:6px;'>"
            f"<span style='width:10px; height:10px; border-radius:50%; background:{color}; "
            f"display:inline-block; box-shadow:0 0 6px {color};'></span>"
            f"<span>{label}</span></span>"
        )
    st.markdown(
        f"<div style='display:flex; gap:24px; justify-content:center; padding:10px 0; "
        f"color:#b8aae8; font-size:0.82rem; font-weight:500;'>"
        + " ".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


def info_callout(title: str, text: str):
    st.markdown(
        f'<div class="info-box"><div class="info-box-title">{title}</div>'
        f'<div class="info-box-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def pipeline_flow(steps: list[tuple], highlight: int = -1):
    """Render a visual pipeline flow with cards and arrows.
    steps: list of (icon, title, subtitle, color_key)
    """
    cards = []
    for i, (icon, title, subtitle, _color_key) in enumerate(steps):
        cls = "pipeline-step active" if i == highlight else "pipeline-step"
        cards.append(
            f'<div class="{cls}" style="flex:1; min-width:110px; max-width:185px;">'
            f'<div class="pipeline-step-num">{i + 1}</div>'
            f'<div class="pipeline-step-icon">{icon}</div>'
            f'<div class="pipeline-step-title">{title}</div>'
            f'<div class="pipeline-step-sub">{subtitle}</div></div>'
        )

    arrow = '<div class="pipeline-arrow">&#10132;</div>'
    html_parts = []
    for i, card in enumerate(cards):
        html_parts.append(card)
        if i < len(cards) - 1:
            html_parts.append(arrow)

    st.markdown(
        '<div style="display:flex; align-items:center; justify-content:center; gap:0;'
        ' flex-wrap:wrap; padding:8px 0 16px;">' + "".join(html_parts) + '</div>',
        unsafe_allow_html=True,
    )
