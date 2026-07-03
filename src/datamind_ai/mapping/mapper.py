from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from datamind_ai.explorer.profile import _infer_column_type


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _content_similarity(left: pd.Series, right: pd.Series, sample_size: int = 500) -> float:
    left_vals = left.dropna().astype(str)
    right_vals = right.dropna().astype(str)
    if len(left_vals) == 0 or len(right_vals) == 0:
        return 0.0

    if len(left_vals) > sample_size:
        left_vals = left_vals.sample(sample_size, random_state=42)
    right_set = set(right_vals.unique())
    matches = left_vals.isin(right_set).sum()
    return round(matches / len(left_vals), 4)


def _suggest_transformation(src_type: str, dst_type: str) -> str | None:
    if src_type == dst_type:
        return None
    type_map = {
        ("String", "DateTime"): "Converter string para data (TO_DATE / STRPTIME)",
        ("Integer", "Float"): "Cast para FLOAT",
        ("Float", "Integer"): "Arredondar e cast para INTEGER",
        ("String", "Integer"): "Validar e converter para INTEGER",
    }
    return type_map.get((src_type, dst_type), f"Converter de {src_type} para {dst_type}")


def propose_mappings(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str = "origem",
    target_name: str = "destino",
) -> dict[str, Any]:
    source_cols = set(source_df.columns)
    target_cols = set(target_df.columns)

    mappings: list[dict[str, Any]] = []
    used_target: set[str] = set()

    for src_col in source_df.columns:
        best_match = None
        best_score = 0.0

        for tgt_col in target_df.columns:
            if tgt_col in used_target:
                continue
            name_sim = _name_similarity(str(src_col), str(tgt_col))
            content_sim = _content_similarity(source_df[src_col], target_df[tgt_col])
            score = round(name_sim * 0.4 + content_sim * 0.6, 4)

            if score > best_score:
                best_score = score
                best_match = tgt_col

        src_type = _infer_column_type(source_df[src_col])
        if best_match is not None and best_score >= 0.3:
            tgt_type = _infer_column_type(target_df[best_match])
            transformation = _suggest_transformation(src_type, tgt_type)
            used_target.add(best_match)
            mappings.append(
                {
                    "source_column": str(src_col),
                    "target_column": str(best_match),
                    "confidence": round(best_score * 100, 1),
                    "source_type": src_type,
                    "target_type": tgt_type,
                    "transformation": transformation,
                    "status": "pendente",
                    "is_suggestion": True,
                }
            )
        else:
            mappings.append(
                {
                    "source_column": str(src_col),
                    "target_column": "",
                    "confidence": 0.0,
                    "source_type": src_type,
                    "target_type": "",
                    "transformation": None,
                    "status": "sem correspondência",
                    "is_suggestion": True,
                }
            )

    missing_in_target = [str(c) for c in source_cols - {m["target_column"] for m in mappings if m["target_column"]}]
    extra_in_target = [str(c) for c in target_cols - used_target]

    non_obvious: list[dict[str, Any]] = []
    for src_col in source_cols:
        already_mapped = any(m["source_column"] == str(src_col) and m["target_column"] for m in mappings)
        if already_mapped:
            continue
        for tgt_col in target_cols:
            if tgt_col in used_target:
                continue
            content_sim = _content_similarity(source_df[src_col], target_df[tgt_col])
            name_sim = _name_similarity(str(src_col), str(tgt_col))
            if content_sim > 0.5 and name_sim < 0.5:
                non_obvious.append(
                    {
                        "source_column": str(src_col),
                        "target_column": str(tgt_col),
                        "reason": "Alta similaridade de conteúdo com nomes diferentes",
                        "content_similarity_pct": round(content_sim * 100, 1),
                    }
                )

    discrepancies: list[dict[str, Any]] = []
    for col in missing_in_target:
        discrepancies.append(
            {
                "type": "coluna_em_falta_no_destino",
                "column": col,
                "action": f"Adicionar coluna '{col}' ao destino ou mapear para coluna existente.",
            }
        )
    for col in extra_in_target:
        discrepancies.append(
            {
                "type": "coluna_extra_no_destino",
                "column": col,
                "action": f"Coluna '{col}' existe no destino mas não na origem — verificar se é esperada.",
            }
        )
    for m in mappings:
        if m.get("transformation"):
            discrepancies.append(
                {
                    "type": "transformacao_necessaria",
                    "column": m["source_column"],
                    "action": m["transformation"],
                }
            )

    return {
        "source": source_name,
        "target": target_name,
        "mappings": mappings,
        "discrepancies": discrepancies,
        "non_obvious_mappings": non_obvious,
    }
