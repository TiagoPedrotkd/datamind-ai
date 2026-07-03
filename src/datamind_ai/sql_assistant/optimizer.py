from __future__ import annotations

import json
import re
from typing import Any

import sqlparse

from datamind_ai.llm.base import LLMProvider

SQL_OPTIMIZE_PROMPT = """\
Tu és um especialista em otimização SQL (DuckDB dialect).
Analisa a query fornecida e sugere otimizações concretas.

Para cada sugestão, responde em JSON (array de objetos):
- id: identificador numérico
- issue: problema identificado
- reason: motivo da alteração
- original_snippet: trecho original relevante
- optimized_snippet: trecho otimizado
- optimized_sql: query SQL completa otimizada (apenas na última sugestão, ou uma query final consolidada)

Padrões a identificar:
- SELECT * quando colunas específicas bastam
- Filtros ineficientes (funções sobre colunas indexáveis)
- Cálculos repetidos que poderiam usar CTEs
- Joins desnecessários

Responde APENAS com JSON válido. Português de Portugal.
"""


def _rule_based_suggestions(sql: str) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    upper = sql.upper()

    if re.search(r"SELECT\s+\*", upper):
        suggestions.append(
            {
                "id": len(suggestions) + 1,
                "issue": "SELECT * detetado",
                "reason": "Selecionar apenas colunas necessárias reduz I/O e melhora desempenho.",
                "original_snippet": "SELECT *",
                "optimized_snippet": "SELECT col1, col2, ...",
                "optimized_sql": None,
            }
        )

    func_on_column = re.findall(
        r"WHERE\s+(\w+)\s*\(\s*(\w+)\s*\)", sql, re.IGNORECASE
    )
    for func, col in func_on_column:
        suggestions.append(
            {
                "id": len(suggestions) + 1,
                "issue": f"Função {func}() aplicada sobre coluna '{col}'",
                "reason": "Filtros com funções sobre colunas impedem uso de índices. Prefira comparações diretas ou ranges.",
                "original_snippet": f"{func}({col})",
                "optimized_snippet": f"{col} >= ... AND {col} < ...",
                "optimized_sql": None,
            }
        )

    subquery_count = len(re.findall(r"\(\s*SELECT", upper))
    if subquery_count >= 2:
        suggestions.append(
            {
                "id": len(suggestions) + 1,
                "issue": f"{subquery_count} subqueries detetadas",
                "reason": "Subqueries repetidas podem ser consolidadas numa CTE (WITH) para legibilidade e reutilização.",
                "original_snippet": "(SELECT ...) repetido",
                "optimized_snippet": "WITH cte AS (...) SELECT ...",
                "optimized_sql": None,
            }
        )

    return suggestions


def optimize_sql(provider: LLMProvider, sql: str) -> dict[str, Any]:
    from datamind_ai.sql_assistant.engine import validate_sql_syntax

    syntax_error = validate_sql_syntax(sql)
    if syntax_error:
        return {"suggestions": [], "optimized_sql": None, "error": syntax_error}

    rule_suggestions = _rule_based_suggestions(sql)

    user_prompt = f"Query SQL a otimizar:\n\n{sql}"
    try:
        response = provider.complete(SQL_OPTIMIZE_PROMPT, user_prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        ai_suggestions = json.loads(content)
        if not isinstance(ai_suggestions, list):
            ai_suggestions = []
    except (json.JSONDecodeError, Exception):
        ai_suggestions = []

    all_suggestions = rule_suggestions + [
        s for s in ai_suggestions if isinstance(s, dict)
    ]

    optimized_sql = sql
    for s in reversed(all_suggestions):
        if s.get("optimized_sql"):
            optimized_sql = s["optimized_sql"]
            break

    if not any(s.get("optimized_sql") for s in all_suggestions) and rule_suggestions:
        optimized_sql = sql

    return {
        "suggestions": all_suggestions,
        "optimized_sql": optimized_sql,
        "error": None,
    }
