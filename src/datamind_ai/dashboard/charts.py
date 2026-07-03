from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datamind_ai.dashboard.labels import display_label
from datamind_ai.explorer.profile import _infer_column_type


def overview_kpis(overview: dict[str, Any]) -> dict[str, str]:
    cols = overview.get("columns", [])
    total_cells = overview.get("row_count", 0) * max(len(cols), 1)
    total_nulls = sum(c.get("null_count", 0) for c in cols)
    null_pct = round(total_nulls / total_cells * 100, 2) if total_cells else 0
    return {
        "Linhas": f"{overview.get('row_count', 0):,}",
        "Colunas": str(overview.get("column_count", 0)),
        "% Valores em falta": f"{null_pct}%",
        "Duplicados": f"{overview.get('duplicate_rows', 0):,}",
    }


def fig_dtype_distribution(overview: dict[str, Any]) -> go.Figure:
    cols = overview.get("columns", [])
    counts: dict[str, int] = {}
    for c in cols:
        dtype = c.get("dtype", "String")
        counts[dtype] = counts.get(dtype, 0) + 1
    fig = px.pie(
        names=list(counts.keys()),
        values=list(counts.values()),
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def fig_missing_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    row_labels: list[str] | None = None,
) -> go.Figure:
    use_cols = columns or [str(c) for c in df.columns]
    if not use_cols:
        return go.Figure()
    labels = row_labels or use_cols
    sample = df[use_cols].head(500) if len(df) > 500 else df[use_cols]
    missing = sample.isna().astype(int)
    y_idx = list(range(len(use_cols)))
    fig = px.imshow(
        missing.T,
        labels=dict(x="Registo", y="Coluna", color="Em falta"),
        x=list(range(len(sample))),
        y=y_idx,
        color_continuous_scale=["#2ecc71", "#e74c3c"],
        aspect="auto",
    )
    fig.update_yaxes(ticktext=labels, tickvals=y_idx)
    fig.update_layout(height=max(320, len(use_cols) * 28))
    return fig


def fig_histogram(
    df: pd.DataFrame,
    column: str,
    label_map: dict[str, str],
    technical: bool = True,
) -> go.Figure:
    title_col = display_label(column, label_map, technical)
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        marginal="box",
    )
    fig.update_layout(bargap=0.05)
    return fig


def fig_top_categories(
    df: pd.DataFrame,
    column: str,
    label_map: dict[str, str],
    technical: bool = True,
    top_n: int = 10,
) -> go.Figure:
    title_col = display_label(column, label_map, technical)
    counts = df[column].astype(str).value_counts().head(top_n).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(counts, x=column, y="count")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def fig_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str],
    label_map: dict[str, str],
    technical: bool = True,
) -> go.Figure:
    if len(columns) < 2:
        return go.Figure()
    corr = df[columns].corr(numeric_only=True)
    labels = [display_label(c, label_map, technical) for c in corr.columns]
    fig = px.imshow(
        corr,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    return fig


def fig_boxplot(
    df: pd.DataFrame,
    column: str,
    label_map: dict[str, str],
    technical: bool = True,
) -> go.Figure:
    title_col = display_label(column, label_map, technical)
    fig = px.box(df, y=column)
    return fig


def fig_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_map: dict[str, str],
    technical: bool = True,
) -> go.Figure:
    x_label = display_label(x_col, label_map, technical)
    y_label = display_label(y_col, label_map, technical)
    fig = px.scatter(df, x=x_col, y=y_col, opacity=0.6)
    return fig


def fig_column_comparison(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_map: dict[str, str],
    technical: bool = True,
) -> go.Figure:
    from plotly.subplots import make_subplots

    label_a = display_label(col_a, label_map, technical)
    label_b = display_label(col_b, label_map, technical)
    fig = make_subplots(rows=1, cols=2, subplot_titles=[label_a, label_b])
    if pd.api.types.is_numeric_dtype(df[col_a]):
        fig.add_trace(go.Histogram(x=df[col_a], name=label_a), row=1, col=1)
    else:
        vc = df[col_a].astype(str).value_counts().head(10)
        fig.add_trace(go.Bar(x=vc.index, y=vc.values, name=label_a), row=1, col=1)
    if pd.api.types.is_numeric_dtype(df[col_b]):
        fig.add_trace(go.Histogram(x=df[col_b], name=label_b), row=1, col=2)
    else:
        vc = df[col_b].astype(str).value_counts().head(10)
        fig.add_trace(go.Bar(x=vc.index, y=vc.values, name=label_b), row=1, col=2)
    fig.update_layout(showlegend=False)
    return fig


def fig_dataset_column_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    column: str,
    name_a: str,
    name_b: str,
    label_map: dict[str, str],
    technical: bool = True,
) -> go.Figure:
    from plotly.subplots import make_subplots

    label = display_label(column, label_map, technical)
    fig = make_subplots(rows=1, cols=2, subplot_titles=[name_a, name_b])

    for idx, (df, title) in enumerate([(df_a, name_a), (df_b, name_b)], start=1):
        if column not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            fig.add_trace(go.Histogram(x=df[column], name=title), row=1, col=idx)
        else:
            vc = df[column].astype(str).value_counts().head(10)
            fig.add_trace(go.Bar(x=vc.index.astype(str), y=vc.values, name=title), row=1, col=idx)

    fig.update_layout(showlegend=False)
    return fig


def export_figure_png(fig: go.Figure) -> bytes:
    return fig.to_image(format="png", scale=2)
