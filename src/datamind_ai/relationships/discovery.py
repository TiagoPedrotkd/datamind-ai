from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd


def _normalize_col_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def _name_similarity(a: str, b: str) -> float:
    na, nb = _normalize_col_name(a), _normalize_col_name(b)
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.7
    return 0.0


def _value_overlap(left: pd.Series, right: pd.Series, sample_size: int = 5000) -> float:
    left_vals = left.dropna().astype(str)
    right_vals = right.dropna().astype(str)
    if len(left_vals) == 0 or len(right_vals) == 0:
        return 0.0

    if len(left_vals) > sample_size:
        left_vals = left_vals.sample(sample_size, random_state=42)
    right_set = set(right_vals.unique())

    matches = left_vals.isin(right_set).sum()
    return round(matches / len(left_vals) * 100, 2)


def discover_relationships(
    datasets: dict[str, pd.DataFrame],
    min_confidence: float = 30.0,
) -> list[dict[str, Any]]:
    if len(datasets) < 2:
        return []

    relationships: list[dict[str, Any]] = []

    for (name_a, df_a), (name_b, df_b) in combinations(datasets.items(), 2):
        for col_a in df_a.columns:
            for col_b in df_b.columns:
                name_sim = _name_similarity(str(col_a), str(col_b))
                overlap = _value_overlap(df_a[col_a], df_b[col_b])
                confidence = round(name_sim * 40 + overlap * 0.6, 2)

                if confidence < min_confidence:
                    continue

                level = "alta" if confidence >= 70 else "média" if confidence >= 50 else "baixa"
                relationships.append(
                    {
                        "dataset_a": name_a,
                        "column_a": str(col_a),
                        "dataset_b": name_b,
                        "column_b": str(col_b),
                        "name_similarity": round(name_sim * 100, 1),
                        "value_overlap_pct": overlap,
                        "confidence": confidence,
                        "confidence_level": level,
                        "is_suggestion": True,
                        "join_hint": (
                            f"LEFT JOIN {name_b} ON {name_a}.{col_a} = {name_b}.{col_b}"
                        ),
                    }
                )

    relationships.sort(key=lambda x: x["confidence"], reverse=True)
    return relationships
