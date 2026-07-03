"""Navegação principal da aplicação."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class NavPage:
    id: str
    title: str
    section: str
    description: str
    icon: str = ""


NAV_PAGES: list[NavPage] = [
    NavPage("dashboard", "Dashboard Visual", "Visual Analytics", "KPIs e gráficos interativos gerados automaticamente", "📊"),
    NavPage("drilldown", "Análise Detalhada", "Visual Analytics", "Filtros, correlações e exploração por coluna", "🔍"),
    NavPage("overview", "Visão Geral", "Exploração", "Estrutura, tipos e pré-visualização dos dados", "📋"),
    NavPage("statistics", "Estatísticas", "Exploração", "Métricas descritivas numéricas e categóricas", "📈"),
    NavPage("quality", "Qualidade de Dados", "Exploração", "Problemas detetados e recomendações", "✅"),
    NavPage("relationships", "Relações", "Exploração", "Possíveis joins entre datasets", "🔗"),
    NavPage("mapping", "Mapeamento", "Exploração", "Comparação origem vs. destino", "↔️"),
    NavPage("dictionary", "Dicionário de Dados", "Assistentes IA", "Documentação de colunas gerada por IA", "📖"),
    NavPage("sql", "SQL Assistant", "Assistentes IA", "Gerar, explicar e otimizar queries", "💾"),
    NavPage("rules", "Regras de Negócio", "Assistentes IA", "Padrões e hipóteses nos dados", "📐"),
    NavPage("kpis", "KPIs Sugeridos", "Assistentes IA", "Indicadores de negócio recomendados", "🎯"),
    NavPage("chat", "AI Chat", "Assistentes IA", "Perguntas em linguagem natural", "💬"),
    NavPage("reports", "Relatórios", "Entregáveis", "Exportação consolidada", "📤"),
]

SECTIONS = ["Visual Analytics", "Exploração", "Assistentes IA", "Entregáveis"]

SECTION_ICONS = {
    "Visual Analytics": "📊",
    "Exploração": "📁",
    "Assistentes IA": "🤖",
    "Entregáveis": "📤",
}


def pages_in_section(section: str) -> list[NavPage]:
    return [p for p in NAV_PAGES if p.section == section]


def get_page(page_id: str) -> NavPage | None:
    for p in NAV_PAGES:
        if p.id == page_id:
            return p
    return None


def render_sidebar_navigation() -> str:
    """Two-level navigation; returns selected page id."""
    if "nav_section" not in st.session_state:
        st.session_state.nav_section = SECTIONS[0]
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = NAV_PAGES[0].id

    st.markdown("**Navegação**")

    section = st.selectbox(
        "Secção",
        SECTIONS,
        index=SECTIONS.index(st.session_state.nav_section)
        if st.session_state.nav_section in SECTIONS
        else 0,
        format_func=lambda s: f"{SECTION_ICONS.get(s, '')} {s}",
        label_visibility="collapsed",
        key="nav_section_select",
    )
    st.session_state.nav_section = section

    section_pages = pages_in_section(section)
    page_labels = {p.id: f"{p.icon} {p.title}" for p in section_pages}
    page_ids = [p.id for p in section_pages]

    if st.session_state.nav_page not in page_ids:
        st.session_state.nav_page = page_ids[0]

    selected = st.radio(
        "Página",
        page_ids,
        index=page_ids.index(st.session_state.nav_page),
        format_func=lambda pid: page_labels[pid],
        label_visibility="collapsed",
        key=f"nav_page_radio_{section}",
    )
    st.session_state.nav_page = selected

    page = get_page(selected)
    if page:
        st.caption(page.description)

    return selected
