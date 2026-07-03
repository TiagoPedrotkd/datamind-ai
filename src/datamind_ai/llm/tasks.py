from __future__ import annotations

from enum import Enum


class LLMTask(str, Enum):
    CHAT = "chat"
    DICTIONARY = "dictionary"
    SQL = "sql"
    BUSINESS_RULES = "business_rules"
    KPI = "kpi"
    RELATIONSHIPS = "relationships"
    QUALITY = "quality"
