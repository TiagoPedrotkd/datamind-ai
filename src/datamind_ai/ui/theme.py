"""Tema visual — cores da marca; texto herda o tema Streamlit (claro/escuro)."""

from __future__ import annotations

import streamlit as st

BRAND_PRIMARY = "#1f4e79"
BRAND_ACCENT = "#2e86ab"


def _hex_luminance(hex_color: str) -> float | None:
    hex_color = hex_color.lower().strip()
    if not hex_color.startswith("#") or len(hex_color) < 7:
        return None
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _is_dark_theme() -> bool:
    try:
        base = st.get_option("theme.base")
        if base == "dark":
            return True
        if base == "light":
            return False
    except Exception:
        pass
    for option in ("theme.backgroundColor", "theme.secondaryBackgroundColor"):
        try:
            bg = st.get_option(option) or ""
            lum = _hex_luminance(str(bg))
            if lum is not None:
                return lum < 140
        except Exception:
            continue
    try:
        tc = st.get_option("theme.textColor") or ""
        lum = _hex_luminance(str(tc))
        if lum is not None:
            return lum > 180
    except Exception:
        pass
    return True  # Streamlit usa escuro por defeito


def streamlit_text_color() -> str:
    """Cor de texto do tema Streamlit activo."""
    try:
        color = st.get_option("theme.textColor")
        if color:
            return str(color)
    except Exception:
        pass
    return "#FAFAFA" if _is_dark_theme() else "#262730"


def streamlit_primary_color(fallback: str = BRAND_PRIMARY) -> str:
    try:
        color = st.get_option("theme.primaryColor")
        if color:
            return str(color)
    except Exception:
        pass
    return "#7eb8e0" if _is_dark_theme() else fallback


def inject_professional_theme(brand_color: str = BRAND_PRIMARY) -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body {{
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        .main .block-container {{
            padding-top: 1.5rem;
            max-width: 1280px;
        }}

        [data-testid="stSidebar"] {{
            border-right: 2px solid {brand_color};
        }}

        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label {{
            font-weight: 600;
            font-size: 0.85rem;
        }}

        /* Componentes custom — herdam cores do tema Streamlit */
        .dm-page-header {{
            border-bottom: 2px solid {brand_color};
            padding-bottom: 0.75rem;
            margin-bottom: 1.25rem;
        }}
        .dm-page-header h1 {{
            color: var(--text-color);
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
        }}
        .dm-page-header p {{
            color: var(--text-color);
            opacity: 0.88;
            margin: 0.35rem 0 0 0;
            font-size: 0.95rem;
        }}
        .dm-breadcrumb {{
            color: var(--text-color);
            opacity: 0.72;
            font-size: 0.8rem;
            margin-bottom: 0.25rem;
            letter-spacing: 0.02em;
        }}
        .dm-breadcrumb strong {{
            color: var(--text-color);
            opacity: 1;
            font-weight: 600;
        }}

        .dm-chart-card {{
            border-left: 3px solid {brand_color};
            padding: 0.25rem 0 0.25rem 0.75rem;
            margin-bottom: 1.25rem;
        }}
        .dm-chart-title {{
            color: var(--text-color);
            font-size: 1rem;
            font-weight: 700;
            margin: 0 0 0.15rem 0;
        }}

        .dm-section-label {{
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-color);
            opacity: 0.65;
            margin: 1rem 0 0.35rem 0;
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 8px;
        }}

        .dm-business-context {{
            border: 2px solid {brand_color};
            border-radius: 10px;
            padding: 1rem 1.15rem 0.5rem 1.15rem;
            margin-bottom: 0.5rem;
            background: rgba(128, 128, 128, 0.12);
        }}
        .dm-business-context h3 {{
            color: var(--text-color);
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
        }}
        .dm-business-context p {{
            color: var(--text-color);
            opacity: 0.88;
            font-size: 0.88rem;
            margin: 0 0 0.75rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    description: str,
    breadcrumb: str,
    dataset_name: str | None = None,
) -> None:
    ds_line = f" · Dataset: <strong>{dataset_name}</strong>" if dataset_name else ""
    st.markdown(
        f"""
        <div class="dm-page-header">
            <div class="dm-breadcrumb">{breadcrumb}{ds_line}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_divider(label: str) -> None:
    st.markdown(f'<p class="dm-section-label">{label}</p>', unsafe_allow_html=True)
