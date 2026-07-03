from __future__ import annotations

import re
from typing import Any


def humanize_column(name: str) -> str:
    label = re.sub(r"[_\-]+", " ", str(name))
    return label.strip().title()


def build_label_map(data_dictionary: list[dict[str, Any]] | None) -> dict[str, str]:
    if not data_dictionary:
        return {}
    labels: dict[str, str] = {}
    for entry in data_dictionary:
        col = entry.get("column", "")
        if not col:
            continue
        desc = (entry.get("description") or "").strip()
        meaning = (entry.get("business_meaning") or "").strip()
        if desc and len(desc) < 60:
            labels[col] = desc
        elif meaning and len(meaning) < 60:
            labels[col] = meaning
        else:
            labels[col] = humanize_column(col)
    return labels


def display_label(column: str, label_map: dict[str, str], technical: bool = True) -> str:
    if technical:
        return str(column)
    return label_map.get(str(column), humanize_column(column))
