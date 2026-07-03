from __future__ import annotations

import pandas as pd

from datamind_ai.config import Settings


def sample_dataframe(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    settings = Settings.from_env()
    limit = max_rows or settings.llm_sample_rows
    if len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=42)


def is_large_dataset(df: pd.DataFrame) -> bool:
    settings = Settings.from_env()
    return len(df) >= settings.large_dataset_rows


def large_dataset_warning(df: pd.DataFrame) -> str | None:
    if not is_large_dataset(df):
        return None
    settings = Settings.from_env()
    return (
        f"Dataset grande ({len(df):,} linhas). "
        f"Análises usam dados completos; módulos de IA usam amostra de "
        f"{settings.llm_sample_rows:,} linhas para manter desempenho."
    )
