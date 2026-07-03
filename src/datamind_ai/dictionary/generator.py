from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from datamind_ai.explorer.profile import _infer_column_type
from datamind_ai.llm.base import LLMProvider
from datamind_ai.utils.sampling import sample_dataframe

DICTIONARY_PROMPT = """\
Tu és um especialista em dicionários de dados. Para cada coluna do dataset, gera:
- description: explicação clara da coluna
- business_meaning: possível significado de negócio (como SUGESTÃO, nunca como facto)

Responde APENAS em JSON válido, array de objetos com chaves:
column, description, business_meaning

Regras:
1. Português de Portugal.
2. Nunca inventes dados — baseia-te apenas no nome da coluna, tipo e exemplos fornecidos.
3. business_meaning deve ser claramente uma hipótese/sugestão.
"""


def _sample_value(series: pd.Series) -> str:
    val = series.dropna().head(1)
    if len(val) == 0:
        return "(sem valores)"
    return str(val.iloc[0])


def build_column_context(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = []
    for col in df.columns:
        columns.append(
            {
                "column": str(col),
                "dtype": _infer_column_type(df[col]),
                "sample": _sample_value(df[col]),
                "null_pct": round(df[col].isna().mean() * 100, 2),
            }
        )
    return columns


def generate_dictionary(
    provider: LLMProvider,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    sampled_df = sample_dataframe(df)
    columns = build_column_context(sampled_df)
    sample_note = ""
    if len(sampled_df) < len(df):
        sample_note = (
            f"\nNota: análise baseada em amostra de {len(sampled_df):,} "
            f"linhas de um total de {len(df):,}."
        )
    user_prompt = (
        f"Colunas do dataset:\n{json.dumps(columns, ensure_ascii=False, indent=2)}"
        f"{sample_note}"
    )

    response = provider.complete(DICTIONARY_PROMPT, user_prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        ai_entries = json.loads(content)
    except json.JSONDecodeError:
        ai_entries = []

    ai_by_col = {e.get("column", ""): e for e in ai_entries if isinstance(e, dict)}

    result = []
    for col_info in columns:
        col_name = col_info["column"]
        ai = ai_by_col.get(col_name, {})
        result.append(
            {
                "column": col_name,
                "description": ai.get("description", ""),
                "dtype": col_info["dtype"],
                "sample_value": col_info["sample"],
                "business_meaning": ai.get("business_meaning", ""),
                "is_ai_suggestion": True,
            }
        )
    return result
