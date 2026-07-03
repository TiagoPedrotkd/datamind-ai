from __future__ import annotations

from typing import Any

import pandas as pd


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "Integer"
    if pd.api.types.is_float_dtype(series):
        return "Float"
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DateTime"
    return "String"


def build_overview(df: pd.DataFrame) -> dict[str, Any]:
    total_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())

    columns = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        columns.append(
            {
                "column": str(col),
                "dtype": _infer_column_type(df[col]),
                "pandas_dtype": str(df[col].dtype),
                "null_count": null_count,
                "null_pct": null_pct,
            }
        )

    return {
        "row_count": total_rows,
        "column_count": len(df.columns),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round((duplicate_rows / total_rows) * 100, 2)
        if total_rows > 0
        else 0.0,
        "columns": columns,
    }


def build_statistics(df: pd.DataFrame) -> dict[str, Any]:
    numeric_stats: list[dict[str, Any]] = []
    categorical_stats: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            clean = series.dropna()
            if len(clean) == 0:
                continue
            numeric_stats.append(
                {
                    "column": str(col),
                    "mean": round(float(clean.mean()), 4),
                    "median": round(float(clean.median()), 4),
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                    "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                }
            )
        else:
            clean = series.dropna().astype(str)
            if len(clean) == 0:
                continue
            mode_val = clean.mode()
            top_value = mode_val.iloc[0] if len(mode_val) > 0 else None
            top_freq = int((clean == top_value).sum()) if top_value is not None else 0
            categorical_stats.append(
                {
                    "column": str(col),
                    "distinct_count": int(clean.nunique()),
                    "mode": top_value,
                    "mode_frequency": top_freq,
                }
            )

    return {
        "numeric": numeric_stats,
        "categorical": categorical_stats,
    }


def statistics_to_dataframes(stats: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_df = pd.DataFrame(stats.get("numeric", []))
    categorical_df = pd.DataFrame(stats.get("categorical", []))
    return numeric_df, categorical_df
