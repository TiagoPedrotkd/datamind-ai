from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd
import sqlparse

from datamind_ai.llm.base import LLMProvider
from datamind_ai.utils.sampling import sample_dataframe


def _sanitize_table_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def build_schema_context(datasets: dict[str, pd.DataFrame]) -> str:
    lines = []
    for name, df in datasets.items():
        table = _sanitize_table_name(name)
        sampled = sample_dataframe(df)
        lines.append(f"Tabela: {table} (dataset: {name})")
        lines.append(f"  Linhas totais: {len(df)}")
        if len(sampled) < len(df):
            lines.append(f"  (amostra IA: {len(sampled)} linhas)")
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = sampled[col].dropna().head(3).tolist()
            lines.append(f"  - {col} ({dtype}) exemplos: {sample}")
        lines.append("")
    return "\n".join(lines)


def register_datasets(conn: duckdb.DuckDBPyConnection, datasets: dict[str, pd.DataFrame]) -> dict[str, str]:
    mapping = {}
    for name, df in datasets.items():
        table = _sanitize_table_name(name)
        conn.register(table, df)
        mapping[name] = table
    return mapping


def validate_sql_syntax(sql: str) -> str | None:
    try:
        parsed = sqlparse.parse(sql)
        if not parsed or not parsed[0].tokens:
            return "Query SQL vazia ou inválida."
    except Exception as exc:
        return f"Erro de sintaxe SQL: {exc}"
    return None


SQL_GENERATION_PROMPT = """\
Tu és um assistente SQL especializado. Gera apenas queries SQL válidas (DuckDB dialect).
Regras:
1. Usa APENAS tabelas e colunas do schema fornecido.
2. Se o pedido for ambíguo ou impossível, pede clarificação em vez de inventar colunas.
3. Responde apenas com a query SQL, sem explicação adicional (a menos que precises de pedir clarificação).
4. Responde em Português de Portugal se pedires clarificação.
"""

SQL_EXPLAIN_PROMPT = """\
Tu és um assistente SQL. Explica queries SQL em linguagem natural clara (Português de Portugal).
Cobre: tabelas envolvidas, filtros, joins e agregações.
Se a query for inválida, indica o erro — não especules sobre o que faria.
"""


def generate_sql(
    provider: LLMProvider,
    schema_context: str,
    user_request: str,
) -> dict[str, Any]:
    user_prompt = (
        f"Schema disponível:\n{schema_context}\n\n"
        f"Pedido do utilizador: {user_request}\n\n"
        "Gera a query SQL correspondente."
    )
    response = provider.complete(SQL_GENERATION_PROMPT, user_prompt)
    sql = response.content.strip()

    if sql.startswith("```"):
        sql = re.sub(r"^```\w*\n?", "", sql)
        sql = re.sub(r"\n?```$", "", sql)

    syntax_error = validate_sql_syntax(sql)
    return {
        "sql": sql,
        "syntax_error": syntax_error,
        "needs_clarification": "?" in sql and "SELECT" not in sql.upper(),
    }


def explain_sql(provider: LLMProvider, sql: str) -> dict[str, Any]:
    syntax_error = validate_sql_syntax(sql)
    if syntax_error:
        return {"explanation": None, "error": syntax_error}

    user_prompt = f"Explica esta query SQL:\n\n{sql}"
    response = provider.complete(SQL_EXPLAIN_PROMPT, user_prompt)
    return {"explanation": response.content, "error": None}


def execute_sql(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    limit: int = 100,
) -> dict[str, Any]:
    syntax_error = validate_sql_syntax(sql)
    if syntax_error:
        return {"success": False, "error": syntax_error, "df": None}

    try:
        df = conn.execute(sql).fetchdf()
        if len(df) > limit:
            df = df.head(limit)
        return {"success": True, "error": None, "df": df}
    except Exception as exc:
        return {"success": False, "error": str(exc), "df": None}
