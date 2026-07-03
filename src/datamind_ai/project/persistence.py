from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from datamind_ai.session import AppSession, DatasetInfo


def _serialize_analyses(info: DatasetInfo) -> dict[str, Any]:
    return {
        "name": info.name,
        "source_filename": info.source_filename,
        "sheet_name": info.sheet_name,
        "overview": info.overview,
        "statistics": info.statistics,
        "quality_report": info.quality_report,
        "data_dictionary": info.data_dictionary,
        "business_rules": info.business_rules,
        "kpi_suggestions": info.kpi_suggestions,
        "business_context": info.business_context,
    }


def save_project(session: AppSession, project_dir: Path, mappings: dict[str, Any]) -> Path:
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)

    datasets_meta = []
    for name, info in session.datasets.items():
        parquet_path = data_dir / f"{name}.parquet"
        info.df.to_parquet(parquet_path, index=False)
        datasets_meta.append(_serialize_analyses(info))

    manifest = {
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "active_dataset": session.active_dataset,
        "chat_history": session.chat_history,
        "datasets": datasets_meta,
        "mappings": mappings,
        "presentation_config": session.presentation_config,
    }

    manifest_path = project_dir / "project.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return project_dir


def load_project(project_dir: Path) -> tuple[AppSession, dict[str, Any]]:
    project_dir = Path(project_dir)
    manifest_path = project_dir / "project.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Projeto não encontrado: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session = AppSession()
    data_dir = project_dir / "data"

    for ds_meta in manifest.get("datasets", []):
        name = ds_meta["name"]
        parquet_path = data_dir / f"{name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Dataset em falta: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        info = DatasetInfo(
            name=name,
            df=df,
            source_filename=ds_meta.get("source_filename", name),
            sheet_name=ds_meta.get("sheet_name"),
            overview=ds_meta.get("overview"),
            statistics=ds_meta.get("statistics"),
            quality_report=ds_meta.get("quality_report"),
            data_dictionary=ds_meta.get("data_dictionary"),
            business_rules=ds_meta.get("business_rules"),
            kpi_suggestions=ds_meta.get("kpi_suggestions"),
            business_context=ds_meta.get("business_context", ""),
        )
        session.datasets[name] = info

    session.active_dataset = manifest.get("active_dataset")
    session.chat_history = manifest.get("chat_history", [])
    session.presentation_config = manifest.get("presentation_config")
    mappings = manifest.get("mappings", {})
    return session, mappings
