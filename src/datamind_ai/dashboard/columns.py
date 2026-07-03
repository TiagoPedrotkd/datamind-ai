from __future__ import annotations

import pandas as pd

from datamind_ai.explorer.profile import _infer_column_type


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        cols.append(str(c))
    return cols


def score_numeric_column(series: pd.Series, null_pct: float) -> float:
    clean = series.dropna()
    if len(clean) < 2:
        return null_pct
    cv = float(clean.std() / clean.mean()) if clean.mean() != 0 else float(clean.std())
    return null_pct * 0.3 + min(cv, 10) * 0.7


def score_categorical_column(series: pd.Series, null_pct: float) -> float:
    clean = series.dropna()
    if len(clean) == 0:
        return null_pct
    nunique = clean.nunique()
    balance = min(nunique, 50) / 50
    return null_pct * 0.4 + balance * 0.6


def select_highlight_columns(
    df: pd.DataFrame,
    overview_columns: list[dict],
    max_numeric: int = 6,
    max_categorical: int = 6,
) -> tuple[list[str], list[str]]:
    null_by_col = {c["column"]: c.get("null_pct", 0) for c in overview_columns}

    num_scores: list[tuple[str, float]] = []
    for col in numeric_columns(df):
        num_scores.append((col, score_numeric_column(df[col], null_by_col.get(col, 0))))
    num_scores.sort(key=lambda x: x[1], reverse=True)
    selected_num = [c for c, _ in num_scores[:max_numeric]]

    cat_scores: list[tuple[str, float]] = []
    for col in categorical_columns(df):
        cat_scores.append((col, score_categorical_column(df[col], null_by_col.get(col, 0))))
    cat_scores.sort(key=lambda x: x[1], reverse=True)
    selected_cat = [c for c, _ in cat_scores[:max_categorical]]

    return selected_num, selected_cat
