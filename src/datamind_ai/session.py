from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetInfo:
    name: str
    df: pd.DataFrame
    source_filename: str
    upload_fingerprint: str | None = None
    sheet_name: str | None = None
    overview: dict[str, Any] | None = None
    statistics: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    data_dictionary: list[dict[str, Any]] | None = None
    business_rules: list[dict[str, Any]] | None = None
    kpi_suggestions: list[dict[str, Any]] | None = None
    business_context: str = ""


@dataclass
class AppSession:
    datasets: dict[str, DatasetInfo] = field(default_factory=dict)
    active_dataset: str | None = None
    chat_history: list[dict[str, str]] = field(default_factory=list)
    mappings: dict[str, Any] = field(default_factory=dict)
    presentation_config: dict[str, Any] | None = None
    llm_backend_name: str = "—"
    llm_model_name: str = "—"

    def add_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        source_filename: str,
        upload_fingerprint: str | None = None,
        sheet_name: str | None = None,
    ) -> None:
        self.datasets[name] = DatasetInfo(
            name=name,
            df=df,
            source_filename=source_filename,
            upload_fingerprint=upload_fingerprint,
            sheet_name=sheet_name,
        )
        if self.active_dataset is None:
            self.active_dataset = name

    def get_active(self) -> DatasetInfo | None:
        if self.active_dataset and self.active_dataset in self.datasets:
            return self.datasets[self.active_dataset]
        return None

    def remove_dataset(self, name: str) -> str | None:
        info = self.datasets.pop(name, None)
        if self.active_dataset == name:
            self.active_dataset = next(iter(self.datasets), None)
        return info.upload_fingerprint if info else None

    def mapping_key(self, source: str, target: str) -> str:
        return f"{source}::{target}"
