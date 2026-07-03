from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from datamind_ai.llm.base import LLMProvider
from datamind_ai.utils.sampling import sample_dataframe

KPI_PROMPT = """\
Tu és um consultor de BI. Com base na estrutura do dataset, sugere KPIs de negócio relevantes.

Responde em JSON (array de objetos):
- name: nome do KPI
- description: descrição breve
- formula: fórmula de cálculo (SQL ou expressão)
- required_columns: lista de colunas necessárias
- domain: domínio de negócio inferido

Regras:
1. Usa APENAS colunas que existem no dataset.
2. Português de Portugal.
3. Responde APENAS com JSON válido.
"""


def _heuristic_kpis(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [str(c) for c in df.columns]
    col_lower = {c: c.lower() for c in columns}
    kpis: list[dict[str, Any]] = []

    revenue_cols = [c for c, cl in col_lower.items() if any(k in cl for k in ("revenue", "amount", "total", "valor", "preco", "price"))]
    if revenue_cols:
        col = revenue_cols[0]
        kpis.append(
            {
                "name": "Receita Total",
                "description": "Soma total dos valores de receita",
                "formula": f"SUM({col})",
                "required_columns": [col],
                "domain": "Financeiro",
                "columns_available": True,
            }
        )

    id_cols = [c for c, cl in col_lower.items() if any(k in cl for k in ("id", "customer", "cliente", "user"))]
    if id_cols:
        col = id_cols[0]
        kpis.append(
            {
                "name": "Total de Registos Únicos",
                "description": f"Contagem de {col} distintos",
                "formula": f"COUNT(DISTINCT {col})",
                "required_columns": [col],
                "domain": "Operacional",
                "columns_available": True,
            }
        )

    date_cols = [c for c in columns if pd.api.types.is_datetime64_any_dtype(df[c]) or any(k in col_lower[c] for k in ("date", "data", "created", "updated"))]
    if date_cols and revenue_cols:
        kpis.append(
            {
                "name": "Receita Média",
                "description": "Valor médio por registo",
                "formula": f"AVG({revenue_cols[0]})",
                "required_columns": [revenue_cols[0]],
                "domain": "Financeiro",
                "columns_available": True,
            }
        )

    return kpis


def _check_columns_available(required: list[str], available: set[str]) -> bool:
    available_lower = {c.lower() for c in available}
    return all(r.lower() in available_lower for r in required)


def suggest_kpis(
    df: pd.DataFrame,
    provider: LLMProvider | None = None,
) -> list[dict[str, Any]]:
    available_cols = {str(c) for c in df.columns}
    sampled = sample_dataframe(df)
    schema_lines = []
    for col in sampled.columns:
        schema_lines.append(f"- {col} ({sampled[col].dtype})")

    kpis = _heuristic_kpis(df)

    if provider and provider.is_available():
        try:
            sample_note = ""
            if len(sampled) < len(df):
                sample_note = (
                    f"\n(Amostra de {len(sampled):,} linhas de {len(df):,} totais.)"
                )
            user_prompt = (
                f"Dataset com {len(df)} linhas.{sample_note}\n"
                f"Colunas:\n" + "\n".join(schema_lines)
            )
            response = provider.complete(KPI_PROMPT, user_prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            ai_kpis = json.loads(content)
            if isinstance(ai_kpis, list):
                for k in ai_kpis:
                    if not isinstance(k, dict):
                        continue
                    required = k.get("required_columns", [])
                    kpis.append(
                        {
                            "name": k.get("name", ""),
                            "description": k.get("description", ""),
                            "formula": k.get("formula", ""),
                            "required_columns": required,
                            "domain": k.get("domain", ""),
                            "columns_available": _check_columns_available(
                                required, available_cols
                            ),
                        }
                    )
        except (json.JSONDecodeError, Exception):
            pass

    seen = set()
    unique: list[dict[str, Any]] = []
    for k in kpis:
        key = k.get("name", "")
        if key and key not in seen:
            seen.add(key)
            if "columns_available" not in k:
                k["columns_available"] = _check_columns_available(
                    k.get("required_columns", []), available_cols
                )
            unique.append(k)

    return unique
