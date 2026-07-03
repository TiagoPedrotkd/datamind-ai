from __future__ import annotations

import re
from typing import Any

import pandas as pd

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_PATTERN = re.compile(r"^[\d\s\-\+\(\)]{7,20}$")


def _severity(pct: float) -> str:
    if pct >= 20:
        return "alta"
    if pct >= 5:
        return "média"
    return "baixa"


def _recommendation(category: str, column: str, pct: float) -> str:
    recommendations = {
        "missing_values": f"Rever origem dos dados para a coluna '{column}' — {pct:.1f}% em falta.",
        "duplicates": "Avaliar se os registos duplicados são esperados ou erro de ingestão.",
        "type_inconsistency": f"Normalizar tipos na coluna '{column}' ou corrigir na origem.",
        "outlier": f"Validar se os valores extremos em '{column}' são legítimos.",
        "format_inconsistency": f"Padronizar formato dos valores em '{column}'.",
    }
    return recommendations.get(category, "Rever dados e regras de negócio associadas.")


def _detect_type_inconsistencies(series: pd.Series) -> list[dict[str, Any]]:
    issues = []
    non_null = series.dropna()
    if len(non_null) == 0:
        return issues

    if pd.api.types.is_numeric_dtype(series):
        as_str = non_null.astype(str)
        non_numeric = as_str[~as_str.str.match(r"^-?\d+\.?\d*$", na=False)]
        if len(non_numeric) > 0:
            pct = round(len(non_numeric) / len(series) * 100, 2)
            issues.append(
                {
                    "category": "type_inconsistency",
                    "column": str(series.name),
                    "count": len(non_numeric),
                    "pct": pct,
                    "severity": _severity(pct),
                    "description": f"Valores não numéricos numa coluna numérica ({len(non_numeric)} ocorrências).",
                    "recommendation": _recommendation("type_inconsistency", str(series.name), pct),
                    "ai_suggestion": True,
                }
            )
    return issues


def _detect_outliers(series: pd.Series) -> list[dict[str, Any]]:
    issues = []
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return issues

    clean = series.dropna()
    if len(clean) < 4:
        return issues

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return issues

    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = clean[(clean < lower) | (clean > upper)]
    if len(outliers) > 0:
        pct = round(len(outliers) / len(series) * 100, 2)
        issues.append(
            {
                "category": "outlier",
                "column": str(series.name),
                "count": len(outliers),
                "pct": pct,
                "severity": _severity(pct),
                "description": f"Outliers estatísticos detetados (método IQR): {len(outliers)} valores.",
                "recommendation": _recommendation("outlier", str(series.name), pct),
                "ai_suggestion": True,
            }
        )
    return issues


def _detect_format_issues(series: pd.Series) -> list[dict[str, Any]]:
    issues = []
    col_name = str(series.name).lower()
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return issues

    if "email" in col_name:
        invalid = non_null[~non_null.str.match(EMAIL_PATTERN, na=False)]
        if len(invalid) > 0:
            pct = round(len(invalid) / len(series) * 100, 2)
            issues.append(
                {
                    "category": "format_inconsistency",
                    "column": str(series.name),
                    "count": len(invalid),
                    "pct": pct,
                    "severity": _severity(pct),
                    "description": f"Formatos de email inválidos: {len(invalid)} ocorrências.",
                    "recommendation": _recommendation("format_inconsistency", str(series.name), pct),
                    "ai_suggestion": True,
                }
            )

    if "phone" in col_name or "telefone" in col_name:
        invalid = non_null[~non_null.str.match(PHONE_PATTERN, na=False)]
        if len(invalid) > 0:
            pct = round(len(invalid) / len(series) * 100, 2)
            issues.append(
                {
                    "category": "format_inconsistency",
                    "column": str(series.name),
                    "count": len(invalid),
                    "pct": pct,
                    "severity": _severity(pct),
                    "description": f"Formatos de telefone inválidos: {len(invalid)} ocorrências.",
                    "recommendation": _recommendation("format_inconsistency", str(series.name), pct),
                    "ai_suggestion": True,
                }
            )

    if "date" in col_name or "data" in col_name:
        parsed = pd.to_datetime(non_null, errors="coerce")
        invalid_count = int(parsed.isna().sum())
        if invalid_count > 0:
            pct = round(invalid_count / len(series) * 100, 2)
            issues.append(
                {
                    "category": "format_inconsistency",
                    "column": str(series.name),
                    "count": invalid_count,
                    "pct": pct,
                    "severity": _severity(pct),
                    "description": f"Datas com formato inconsistente: {invalid_count} ocorrências.",
                    "recommendation": _recommendation("format_inconsistency", str(series.name), pct),
                    "ai_suggestion": True,
                }
            )

    return issues


def analyze_quality(df: pd.DataFrame) -> dict[str, Any]:
    total_rows = len(df)
    issues: list[dict[str, Any]] = []

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            pct = round(null_count / total_rows * 100, 2) if total_rows > 0 else 0.0
            issues.append(
                {
                    "category": "missing_values",
                    "column": str(col),
                    "count": null_count,
                    "pct": pct,
                    "severity": _severity(pct),
                    "description": f"Valores em falta: {pct}% dos registos.",
                    "recommendation": _recommendation("missing_values", str(col), pct),
                    "ai_suggestion": True,
                }
            )

        issues.extend(_detect_type_inconsistencies(df[col]))
        issues.extend(_detect_outliers(df[col]))
        issues.extend(_detect_format_issues(df[col]))

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        pct = round(dup_count / total_rows * 100, 2) if total_rows > 0 else 0.0
        issues.append(
            {
                "category": "duplicates",
                "column": "(dataset completo)",
                "count": dup_count,
                "pct": pct,
                "severity": _severity(pct),
                "description": f"Registos duplicados: {dup_count} ({pct}%).",
                "recommendation": _recommendation("duplicates", "", pct),
                "ai_suggestion": True,
            }
        )

    issues.sort(key=lambda x: x["pct"], reverse=True)

    return {
        "total_issues": len(issues),
        "issues": issues,
    }
