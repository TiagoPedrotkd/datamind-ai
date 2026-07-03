from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from datamind_ai.llm.base import LLMProvider

BUSINESS_RULES_PROMPT = """\
Tu és um analista de dados. Com base nos padrões estatísticos fornecidos, formula regras de negócio
como HIPÓTESES a validar (nunca como factos confirmados).

Responde em JSON (array de objetos):
- rule: descrição da regra em linguagem natural
- columns: colunas envolvidas
- evidence_pct: percentagem de suporte (usa os valores fornecidos)
- hypothesis: true

Português de Portugal. Responde APENAS com JSON válido.
"""


def _detect_statistical_patterns(df: pd.DataFrame) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    total = len(df)
    if total == 0:
        return patterns

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            min_val, max_val = series.min(), series.max()
            if min_val == max_val:
                pct = round(len(series) / total * 100, 2)
                patterns.append(
                    {
                        "type": "constant_numeric",
                        "column": str(col),
                        "rule": f"{col} = {min_val}",
                        "evidence_pct": pct,
                        "detail": f"Todos os valores numéricos são {min_val}",
                    }
                )
            elif min_val >= 0 and (series >= 0).all():
                pct = round((series >= 0).sum() / total * 100, 2)
                if pct >= 90:
                    patterns.append(
                        {
                            "type": "threshold",
                            "column": str(col),
                            "rule": f"{col} >= 0",
                            "evidence_pct": pct,
                            "detail": f"{pct}% dos registos têm {col} >= 0",
                        }
                    )
            if "age" in str(col).lower() or "idade" in str(col).lower():
                adult_pct = round((series >= 18).sum() / total * 100, 2)
                if adult_pct >= 90:
                    patterns.append(
                        {
                            "type": "threshold",
                            "column": str(col),
                            "rule": f"{col} >= 18",
                            "evidence_pct": adult_pct,
                            "detail": f"{adult_pct}% dos registos têm idade >= 18",
                        }
                    )
        else:
            value_counts = series.astype(str).value_counts(normalize=True)
            if len(value_counts) == 1:
                val = value_counts.index[0]
                pct = round(value_counts.iloc[0] * 100, 2)
                patterns.append(
                    {
                        "type": "constant_categorical",
                        "column": str(col),
                        "rule": f"{col} = '{val}'",
                        "evidence_pct": pct,
                        "detail": f"Valor constante: '{val}'",
                    }
                )
            elif len(value_counts) > 0 and value_counts.iloc[0] >= 0.9:
                val = value_counts.index[0]
                pct = round(value_counts.iloc[0] * 100, 2)
                patterns.append(
                    {
                        "type": "dominant_value",
                        "column": str(col),
                        "rule": f"{col} IN ('{val}')",
                        "evidence_pct": pct,
                        "detail": f"'{val}' representa {pct}% dos valores",
                    }
                )
            if len(value_counts) <= 5 and len(value_counts) > 0:
                allowed = "', '".join(value_counts.index[:5].tolist())
                coverage = round(value_counts.sum() * 100, 2)
                patterns.append(
                    {
                        "type": "limited_values",
                        "column": str(col),
                        "rule": f"{col} IN ('{allowed}')",
                        "evidence_pct": coverage,
                        "detail": f"Apenas {len(value_counts)} valores distintos",
                    }
                )

    return patterns


def detect_business_rules(
    df: pd.DataFrame,
    provider: LLMProvider | None = None,
) -> list[dict[str, Any]]:
    patterns = _detect_statistical_patterns(df)

    rules = [
        {
            "rule": p["rule"],
            "columns": [p["column"]],
            "evidence_pct": float(p["evidence_pct"]),
            "detail": p["detail"],
            "is_hypothesis": True,
            "source": "statistical",
        }
        for p in patterns
    ]

    if provider and provider.is_available() and patterns:
        try:
            user_prompt = f"Padrões estatísticos detetados:\n{json.dumps(patterns, ensure_ascii=False, indent=2)}"
            response = provider.complete(BUSINESS_RULES_PROMPT, user_prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            ai_rules = json.loads(content)
            if isinstance(ai_rules, list):
                for r in ai_rules:
                    if isinstance(r, dict):
                        rules.append(
                            {
                                "rule": r.get("rule", ""),
                                "columns": r.get("columns", []),
                                "evidence_pct": r.get("evidence_pct", 0),
                                "detail": "Sugestão gerada por IA",
                                "is_hypothesis": True,
                                "source": "ai",
                            }
                        )
        except (json.JSONDecodeError, Exception):
            pass

    rules.sort(key=lambda x: x.get("evidence_pct", 0), reverse=True)
    return rules
