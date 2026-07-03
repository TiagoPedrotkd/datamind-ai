from __future__ import annotations

from datetime import datetime
from typing import Any

from datamind_ai.session import DatasetInfo


def _format_issues(quality: dict[str, Any] | None) -> str:
    if not quality or not quality.get("issues"):
        return "Nenhum problema detetado.\n"
    lines = []
    for issue in quality["issues"]:
        lines.append(
            f"- **{issue['category']}** — {issue['column']}: "
            f"{issue['count']} ocorrências ({issue['pct']}%) — {issue['recommendation']}"
        )
    return "\n".join(lines) + "\n"


def _format_business_rules(rules: list[dict[str, Any]] | None) -> str:
    if not rules:
        return "Regras não analisadas.\n"
    lines = []
    for rule in rules:
        lines.append(
            f"- **{rule.get('rule', '')}** — {rule.get('evidence_pct', 0)}% suporte "
            f"_(hipótese a validar)_"
        )
    return "\n".join(lines) + "\n"


def _format_kpis(kpis: list[dict[str, Any]] | None) -> str:
    if not kpis:
        return "KPIs não sugeridos.\n"
    lines = []
    for kpi in kpis:
        avail = "✅" if kpi.get("columns_available") else "❌"
        lines.append(
            f"- **{kpi.get('name', '')}** ({kpi.get('domain', '')}): "
            f"`{kpi.get('formula', '')}` — Colunas disponíveis: {avail}"
        )
    return "\n".join(lines) + "\n"


def _format_dictionary(dictionary: list[dict[str, Any]] | None) -> str:
    if not dictionary:
        return "Dicionário não gerado.\n"
    lines = ["| Coluna | Tipo | Descrição | Exemplo | Significado (sugestão) |",
             "|--------|------|-----------|---------|------------------------|"]
    for entry in dictionary:
        lines.append(
            f"| {entry.get('column', '')} | {entry.get('dtype', '')} | "
            f"{entry.get('description', '')} | {entry.get('sample_value', '')} | "
            f"{entry.get('business_meaning', '')} |"
        )
    return "\n".join(lines) + "\n"


def generate_markdown_report(info: DatasetInfo) -> str:
    overview = info.overview or {}
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    cols = overview.get("columns", [])

    col_summary = ""
    if cols:
        header = "| Coluna | Tipo | Valores em falta | % em falta |\n|--------|------|------------------|------------|\n"
        rows = [
            f"| {c['column']} | {c['dtype']} | {c['null_count']} | {c['null_pct']}% |"
            for c in cols
        ]
        col_summary = header + "\n".join(rows)

    return f"""# Relatório DataMind AI — {info.name}

**Ficheiro:** {info.source_filename}
**Gerado em:** {date_str}

## Resumo Executivo

O dataset **{info.name}** contém **{overview.get('row_count', '?')}** linhas e
**{overview.get('column_count', '?')}** colunas. Foram detetados
**{overview.get('duplicate_rows', 0)}** registos duplicados
({overview.get('duplicate_pct', 0)}%).

## Visão Geral das Colunas

{col_summary}

## Relatório de Qualidade de Dados

{_format_issues(info.quality_report)}

## Dicionário de Dados

{_format_dictionary(info.data_dictionary)}

## Regras de Negócio (Hipóteses)

{_format_business_rules(info.business_rules)}

## KPIs Sugeridos

{_format_kpis(info.kpi_suggestions)}

---
*Relatório gerado automaticamente pelo DataMind AI. Interpretações de IA devem ser validadas.*
"""


def export_filename(dataset_name: str, extension: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    safe_name = dataset_name.replace(" ", "_")
    return f"datamind_{safe_name}_{date_str}.{extension}"
