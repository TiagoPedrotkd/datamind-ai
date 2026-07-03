"""Tema Plotly — fundo transparente; eixos legíveis em tema claro e escuro."""

from __future__ import annotations

import plotly.graph_objects as go

from datamind_ai.dashboard.chart_catalog import ChartMeta
from datamind_ai.ui.theme import _is_dark_theme, streamlit_primary_color

DEFAULT_COLORWAY = ["#1f4e79", "#2e86ab", "#457b9d", "#a8dadc", "#e63946", "#f4a261"]

# Cores explícitas — Plotly não herda CSS do Streamlit
_DARK = {
    "text": "#FAFAFA",
    "grid": "rgba(255,255,255,0.25)",
    "line": "rgba(255,255,255,0.45)",
    "template": "plotly_dark",
}
_LIGHT = {
    "text": "#1e293b",
    "grid": "rgba(31,78,121,0.18)",
    "line": "rgba(0,0,0,0.2)",
    "template": "plotly_white",
}


def _chart_palette(brand_color: str = "#1f4e79") -> dict[str, str]:
    palette = _DARK if _is_dark_theme() else _LIGHT
    return {
        **palette,
        "title": streamlit_primary_color(brand_color),
    }


def _apply_axis_style(fig: go.Figure, text_color: str, grid_color: str, line_color: str) -> None:
    """Aplica estilo a todos os eixos, incluindo subplots e marginais."""
    axis_style = dict(
        gridcolor=grid_color,
        linecolor=line_color,
        zerolinecolor=grid_color,
        tickfont=dict(color=text_color, size=12),
        title_font=dict(color=text_color, size=13),
        color=text_color,
        showline=True,
        mirror=False,
    )

    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.for_each_xaxis(lambda axis: axis.update(axis_style))
    fig.for_each_yaxis(lambda axis: axis.update(axis_style))

    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=text_color, size=12),
            title_font=dict(color=text_color, size=13),
            outlinewidth=0,
        ),
    )

    # Títulos de subplots (ex.: comparação lado a lado)
    fig.update_annotations(font=dict(color=text_color, size=13))

    # Valores dentro de heatmaps / correlações
    fig.update_traces(
        selector=dict(type="heatmap"),
        textfont=dict(color=text_color, size=11),
    )


def apply_chart_theme(
    fig: go.Figure,
    meta: ChartMeta,
    *,
    brand_color: str = "#1f4e79",
    column_label: str | None = None,
    show_title: bool = True,
) -> go.Figure:
    title = meta.title
    if column_label:
        title = f"{meta.title} — {column_label}"

    colors = _chart_palette(brand_color)
    text_color = colors["text"]
    dark = colors["template"] == "plotly_dark"

    fig.update_layout(
        template=colors["template"],
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=text_color),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=56, b=48, l=56, r=24),
        colorway=DEFAULT_COLORWAY,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=text_color, size=12),
        ),
        hoverlabel=dict(
            font_size=12,
            font_family="Inter, Segoe UI",
            font_color="#FAFAFA" if dark else "#1e293b",
            bgcolor="rgba(14,17,23,0.95)" if dark else "rgba(255,255,255,0.96)",
        ),
    )

    if show_title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=16, color=colors["title"], family="Inter, Segoe UI, sans-serif"),
                x=0,
                xanchor="left",
            ),
        )
    else:
        fig.update_layout(margin=dict(t=24, b=48, l=56, r=24))

    _apply_axis_style(fig, text_color, colors["grid"], colors["line"])

    return fig
