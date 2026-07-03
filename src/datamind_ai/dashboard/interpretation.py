"""Interpretação de gráficos via IA (Ollama local) com base nos dados."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from datamind_ai.llm.base import LLMProvider

INTERPRET_PROMPT = """\
Tu és um analista de dados sénior. O utilizador está a ver um gráfico do DataMind AI.
Com base APENAS nos dados estatísticos fornecidos abaixo, escreve uma interpretação
profissional em Português de Portugal.

Estrutura da resposta:
1. **O que o gráfico representa** (1-2 frases)
2. **Principais padrões** (factos extraídos dos números)
3. **Pontos de atenção** (anomalias, valores em falta, desbalanceamento, etc.)
4. **Sugestões** (opcional — marcar claramente como hipóteses a validar)

Regras estritas:
- Nunca inventes valores que não estejam nos dados fornecidos.
- Em gráficos de correlação ou scatter: correlação NÃO implica causalidade.
- Se os dados forem insuficientes, indica-o.
- Se o utilizador fornecer contexto de negócio, usa-o para enquadrar a análise — sem inventar factos extra.
- Tom profissional, conciso, adequado a relatório de consultoria.
"""


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna()
    if len(clean) == 0:
        return {"count": 0}
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    outliers = 0
    if iqr > 0:
        outliers = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
    return {
        "count": int(len(clean)),
        "null_pct": round(series.isna().mean() * 100, 2),
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4) if len(clean) > 1 else 0,
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "outliers_iqr": outliers,
    }


def build_chart_context(
    chart_type: str,
    df: pd.DataFrame,
    *,
    title: str,
    column: str | None = None,
    columns: list[str] | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    col_a: str | None = None,
    col_b: str | None = None,
    overview: dict[str, Any] | None = None,
    dataset_a: str | None = None,
    dataset_b: str | None = None,
    stats_b: dict[str, Any] | None = None,
    rows_b: int | None = None,
    row_count: int | None = None,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "chart_type": chart_type,
        "title": title,
        "dataset_rows": len(df),
    }

    if chart_type == "dtypes" and overview:
        counts: dict[str, int] = {}
        for c in overview.get("columns", []):
            dt = c.get("dtype", "String")
            counts[dt] = counts.get(dt, 0) + 1
        ctx["type_counts"] = counts
        ctx["total_columns"] = overview.get("column_count", 0)

    elif chart_type == "missing_heatmap" and columns:
        ctx["missing_by_column"] = {
            col: round(df[col].isna().mean() * 100, 2) for col in columns if col in df.columns
        }
        ctx["columns_shown"] = len(columns)

    elif chart_type == "histogram" and column and column in df.columns:
        ctx["column"] = column
        ctx["stats"] = _numeric_summary(df[column])

    elif chart_type == "categories" and column and column in df.columns:
        vc = df[column].astype(str).value_counts().head(10)
        ctx["column"] = column
        ctx["distinct"] = int(df[column].nunique())
        ctx["null_pct"] = round(df[column].isna().mean() * 100, 2)
        ctx["top_values"] = {str(k): int(v) for k, v in vc.items()}

    elif chart_type == "correlation" and columns:
        num_cols = [c for c in columns if c in df.columns]
        if len(num_cols) >= 2:
            corr = df[num_cols].corr(numeric_only=True)
            pairs = []
            for i, a in enumerate(num_cols):
                for b in num_cols[i + 1 :]:
                    pairs.append({"a": a, "b": b, "r": round(float(corr.loc[a, b]), 3)})
            pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
            ctx["correlations"] = pairs[:10]
            ctx["columns"] = num_cols

    elif chart_type == "boxplot" and column and column in df.columns:
        ctx["column"] = column
        ctx["stats"] = _numeric_summary(df[column])

    elif chart_type == "scatter" and x_col and y_col:
        if x_col in df.columns and y_col in df.columns:
            sub = df[[x_col, y_col]].dropna()
            r = sub[x_col].corr(sub[y_col]) if len(sub) > 1 else None
            ctx["x"] = x_col
            ctx["y"] = y_col
            ctx["pearson_r"] = round(float(r), 4) if r is not None and pd.notna(r) else None
            ctx["points"] = len(sub)

    elif chart_type == "comparison" and col_a and col_b:
        ctx["column_a"] = {"name": col_a, "summary": _summarize_column(df, col_a)}
        ctx["column_b"] = {"name": col_b, "summary": _summarize_column(df, col_b)}

    elif chart_type == "dataset_comparison" and column:
        ctx["column"] = column
        ctx["dataset_a"] = dataset_a
        ctx["dataset_b"] = dataset_b
        ctx["stats_a"] = _summarize_column(df, column)
        ctx["stats_b"] = stats_b or {}
        ctx["rows_a"] = len(df)
        ctx["rows_b"] = rows_b

    return ctx


def _summarize_column(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {}
    if pd.api.types.is_numeric_dtype(df[col]):
        return _numeric_summary(df[col])
    vc = df[col].astype(str).value_counts().head(5)
    return {
        "type": "categorical",
        "distinct": int(df[col].nunique()),
        "top": {str(k): int(v) for k, v in vc.items()},
    }


def interpret_chart(
    provider: LLMProvider,
    context: dict[str, Any],
    *,
    business_context: str = "",
) -> str:
    biz_block = ""
    if business_context.strip():
        biz_block = (
            f"\nContexto de negócio (fornecido pelo utilizador):\n"
            f"{business_context.strip()}\n"
        )

    user_prompt = (
        f"Gráfico: {context.get('title', '')}\n"
        f"Tipo: {context.get('chart_type', '')}\n"
        f"{biz_block}\n"
        f"Dados estatísticos:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        "Escreve a interpretação."
    )
    response = provider.complete(INTERPRET_PROMPT, user_prompt)
    return response.content.strip()
