"""DataMind AI — Streamlit application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import duckdb
import pandas as pd
import streamlit as st

from datamind_ai.config import Settings
from datamind_ai.business_rules.detector import detect_business_rules
from datamind_ai.dashboard.ui import (
    PresentationConfig,
    render_drilldown_dashboard,
    render_overview_dashboard,
    render_presentation_config_sidebar,
    render_presentation_mode,
    sync_business_context_from_state,
)
from datamind_ai.dictionary.generator import generate_dictionary
from datamind_ai.export.formats import export_to_excel, export_to_pdf, export_to_word
from datamind_ai.export.reporter import export_filename, generate_markdown_report
from datamind_ai.explorer.profile import (
    build_overview,
    build_statistics,
    statistics_to_dataframes,
)
from datamind_ai.kpi.generator import suggest_kpis
from datamind_ai.llm.factory import (
    OLLAMA_SETUP_MSG,
    create_provider,
    create_provider_for_task,
    ensure_ollama_available,
    get_installed_ollama_models,
)
from datamind_ai.llm.tasks import LLMTask
from datamind_ai.mapping.mapper import propose_mappings
from datamind_ai.project.persistence import load_project, save_project
from datamind_ai.quality.analyzer import analyze_quality
from datamind_ai.relationships.discovery import discover_relationships
from datamind_ai.session import AppSession
from datamind_ai.sql_assistant.engine import (
    build_schema_context,
    execute_sql,
    explain_sql,
    generate_sql,
    register_datasets,
)
from datamind_ai.sql_assistant.optimizer import optimize_sql
from datamind_ai.upload.loader import (
    MultiUploadResult,
    file_fingerprint,
    load_file,
    make_dataset_name,
)
from datamind_ai.ui.navigation import get_page, render_sidebar_navigation
from datamind_ai.ui.theme import inject_professional_theme, render_page_header
from datamind_ai.utils.sampling import large_dataset_warning

BRAND_COLOR = "#1f4e79"

st.set_page_config(
    page_title="DataMind AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = """\
Tu és o motor de análise de dados do DataMind AI.
Regras estritas:
1. Responde sempre em Português de Portugal, de forma clara e concisa.
2. Nunca inventes valores, relações ou regras de negócio que não estejam suportados pelos dados carregados.
3. Distingue sempre "facto extraído dos dados" de "sugestão/interpretação gerada".
4. Se não tiveres informação suficiente, indica-o explicitamente.
"""

PROJECTS_DIR = Path(__file__).parent / "projects"
SETTINGS = Settings.from_env()


def init_session() -> AppSession:
    if "app_session" not in st.session_state:
        st.session_state.app_session = AppSession()
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()
    if "presentation_mode" not in st.session_state:
        st.session_state.presentation_mode = False
    return st.session_state.app_session


def get_presentation_config(session: AppSession) -> PresentationConfig:
    if session.presentation_config:
        return PresentationConfig(**session.presentation_config)
    return PresentationConfig()


def save_presentation_config(session: AppSession, config: PresentationConfig) -> None:
    session.presentation_config = {
        "visible_widgets": config.visible_widgets,
        "brand_color": config.brand_color,
        "presentation_title": config.presentation_title,
        "company_name": config.company_name,
    }


def llm_error_message() -> str:
    return OLLAMA_SETUP_MSG if SETTINGS.confidential_mode else (
        "Backend de IA não disponível. Inicie o Ollama ou configure OPENAI_API_KEY."
    )


def init_llm() -> None:
    if "llm_provider" not in st.session_state:
        provider = create_provider()
        st.session_state.llm_provider = provider
        session = init_session()
        session.llm_backend_name = provider.name
        session.llm_model_name = provider.model


def process_upload(session: AppSession, filename: str, content: bytes) -> list[str]:
    """Process a new upload; returns status messages."""
    messages: list[str] = []
    fp = file_fingerprint(filename, content)
    if fp in st.session_state.processed_uploads:
        return messages

    result = load_file(filename, content)
    items: list = []
    if isinstance(result, MultiUploadResult):
        if not result.success:
            return [f"❌ {filename}: sem dados"]
        items = result.datasets
    else:
        if not result.success or result.df is None:
            return [f"❌ {filename}: {result.error}"]
        items = [result]

    for item in items:
        if item.df is None:
            continue
        name = make_dataset_name(
            filename,
            set(session.datasets.keys()),
            sheet_name=item.sheet_name,
        )
        label = filename
        if item.sheet_name:
            label = f"{filename} [{item.sheet_name}]"
        session.add_dataset(
            name,
            item.df,
            filename,
            upload_fingerprint=fp,
            sheet_name=item.sheet_name,
        )
        refresh_analyses(name)
        messages.append(f"✅ {label} → `{name}`")
        warn = large_dataset_warning(item.df)
        if warn:
            messages.append(f"⚠️ `{name}`: {warn}")

    st.session_state.processed_uploads.add(fp)
    return messages


def refresh_analyses(dataset_name: str) -> None:
    session = init_session()
    info = session.datasets.get(dataset_name)
    if info is None:
        return
    info.overview = build_overview(info.df)
    info.statistics = build_statistics(info.df)
    info.quality_report = analyze_quality(info.df)


def render_sidebar(session: AppSession) -> None:
    st.sidebar.markdown("### DataMind AI")
    st.sidebar.caption("Análise de dados · modo local")

    if SETTINGS.confidential_mode:
        st.sidebar.success("🔒 Dados processados apenas no seu PC")

    st.sidebar.divider()

    # --- Dados ---
    st.sidebar.markdown("**Dados**")
    uploaded_files = st.sidebar.file_uploader(
        "Carregar ficheiros",
        type=["csv", "xlsx", "json"],
        accept_multiple_files=True,
        help="CSV, Excel (todas as folhas), JSON",
        label_visibility="collapsed",
    )
    if uploaded_files:
        for uploaded in uploaded_files:
            for msg in process_upload(session, uploaded.name, uploaded.getvalue()):
                if msg.startswith("✅"):
                    st.sidebar.success(msg)
                elif msg.startswith("⚠️"):
                    st.sidebar.warning(msg)
                else:
                    st.sidebar.error(msg)

    if session.datasets:
        dataset_names = list(session.datasets.keys())
        selected = st.sidebar.selectbox(
            "Dataset ativo",
            dataset_names,
            index=dataset_names.index(session.active_dataset)
            if session.active_dataset in dataset_names
            else 0,
        )
        session.active_dataset = selected

        st.sidebar.divider()
        render_sidebar_navigation()

        st.sidebar.divider()
        with st.sidebar.expander("Projeto"):
            PROJECTS_DIR.mkdir(exist_ok=True)
            existing_projects = sorted(
                p.name for p in PROJECTS_DIR.iterdir()
                if p.is_dir() and (p / "project.json").exists()
            )
            project_name = st.text_input(
                "Nome",
                value=st.session_state.get("project_name", ""),
                placeholder="cliente_xyz_2026",
            )
            st.session_state.project_name = project_name
            if st.button("Guardar", use_container_width=True, disabled=not project_name):
                sync_business_context_from_state(session)
                path = save_project(session, PROJECTS_DIR / project_name, session.mappings)
                st.success(f"Guardado: {path.name}")
            if existing_projects:
                load_choice = st.selectbox("Abrir", existing_projects)
                if st.button("Carregar", use_container_width=True):
                    loaded_session, loaded_mappings = load_project(PROJECTS_DIR / load_choice)
                    st.session_state.app_session = loaded_session
                    st.session_state.processed_uploads = {
                        info.upload_fingerprint
                        for info in loaded_session.datasets.values()
                        if info.upload_fingerprint
                    }
                    for name, ds in loaded_session.datasets.items():
                        st.session_state[f"business_context_{name}"] = ds.business_context or ""
                    loaded_session.mappings = loaded_mappings
                    st.session_state.project_name = load_choice
                    st.rerun()

        with st.sidebar.expander("Modo Apresentação"):
            pres_config = get_presentation_config(session)
            pres_config = render_presentation_config_sidebar(pres_config)
            save_presentation_config(session, pres_config)
            if st.button("Entrar em apresentação", use_container_width=True):
                save_presentation_config(session, pres_config)
                st.session_state.presentation_mode = True
                st.rerun()

        if st.sidebar.button("Remover dataset", type="secondary", use_container_width=True):
            fp = session.remove_dataset(selected)
            if fp:
                st.session_state.processed_uploads.discard(fp)
            st.rerun()
    else:
        st.sidebar.info("Carregue um ficheiro para começar.")

    with st.sidebar.expander("Configuração técnica"):
        ollama_error = ensure_ollama_available()
        if ollama_error:
            st.error("Ollama não detetado")
            st.markdown(ollama_error)
        else:
            init_llm()
            provider = st.session_state.llm_provider
            st.write(f"**Modelo:** `{provider.model}`")
            for m in get_installed_ollama_models():
                st.write(f"- `{m}`")
            if getattr(provider, "resolve_warning", None):
                st.warning(provider.resolve_warning)


def render_visual_dashboard_tab(info) -> None:
    page = get_page("dashboard")
    render_page_header(
        page.title if page else "Dashboard Visual",
        page.description if page else "",
        breadcrumb="Visual Analytics",
        dataset_name=info.name,
    )
    show_all = st.toggle("Ver todas as colunas", value=False, help="Mostrar heatmap e gráficos para todas as colunas")
    render_overview_dashboard(info, technical=True, show_all_columns=show_all, brand_color=BRAND_COLOR)


def render_drilldown_tab(info, session: AppSession) -> None:
    page = get_page("drilldown")
    render_page_header(
        page.title if page else "Análise Detalhada",
        page.description if page else "",
        breadcrumb="Visual Analytics",
        dataset_name=info.name,
    )
    render_drilldown_dashboard(info, session, technical=True, brand_color=BRAND_COLOR)


def render_overview_tab(info) -> None:
    page = get_page("overview")
    render_page_header(
        page.title if page else "Visão Geral",
        page.description if page else "",
        breadcrumb="Exploração",
        dataset_name=info.name,
    )
    overview = info.overview
    if overview is None:
        st.warning("Visão geral não disponível.")
        return

    warn = large_dataset_warning(info.df)
    if warn:
        st.warning(warn)
    if info.sheet_name:
        st.caption(f"Folha Excel: **{info.sheet_name}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Linhas", f"{overview['row_count']:,}")
    col2.metric("Colunas", overview["column_count"])
    col3.metric("Duplicados", f"{overview['duplicate_rows']:,}")
    col4.metric("% Duplicados", f"{overview['duplicate_pct']}%")

    st.subheader("Colunas")
    cols_df = pd.DataFrame(overview["columns"])
    cols_df = cols_df.rename(
        columns={
            "column": "Coluna",
            "dtype": "Tipo",
            "null_count": "Valores em falta",
            "null_pct": "% em falta",
        }
    )
    st.dataframe(cols_df, use_container_width=True, hide_index=True)

    st.subheader("Pré-visualização")
    st.dataframe(info.df.head(100), use_container_width=True)


def render_statistics_tab(info) -> None:
    page = get_page("statistics")
    render_page_header(
        page.title if page else "Estatísticas",
        page.description if page else "",
        breadcrumb="Exploração",
        dataset_name=info.name,
    )
    stats = info.statistics
    if stats is None:
        st.warning("Estatísticas não disponíveis.")
        return

    numeric_df, categorical_df = statistics_to_dataframes(stats)

    st.subheader("Estatísticas numéricas")
    if numeric_df.empty:
        st.info("Nenhuma coluna numérica encontrada.")
    else:
        st.dataframe(numeric_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar estatísticas numéricas (CSV)",
            numeric_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{info.name}_stats_numeric.csv",
            mime="text/csv",
        )

    st.subheader("Estatísticas categóricas")
    if categorical_df.empty:
        st.info("Nenhuma coluna categórica encontrada.")
    else:
        st.dataframe(categorical_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar estatísticas categóricas (CSV)",
            categorical_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{info.name}_stats_categorical.csv",
            mime="text/csv",
        )


def render_quality_tab(info) -> None:
    report = info.quality_report
    if report is None:
        st.warning("Relatório de qualidade não disponível.")
        return

    if report["total_issues"] == 0:
        st.success("Nenhum problema de qualidade detetado.")
        return

    st.metric("Total de problemas", report["total_issues"])

    for issue in report["issues"]:
        severity_icon = {"alta": "🔴", "média": "🟠", "baixa": "🟡"}.get(
            issue["severity"], "⚪"
        )
        with st.expander(
            f"{severity_icon} {issue['category']} — {issue['column']} "
            f"({issue['pct']}%)"
        ):
            st.write(f"**Descrição:** {issue['description']}")
            st.write(f"**Ocorrências:** {issue['count']:,} ({issue['pct']}%)")
            st.write(f"**Severidade:** {issue['severity']}")
            st.info(f"**Recomendação:** {issue['recommendation']}")
            if issue.get("ai_suggestion"):
                st.caption("⚠️ Recomendação gerada automaticamente — validar com contexto de negócio.")


def render_chat_tab(info, session: AppSession) -> None:
    st.caption(
        "Faça perguntas sobre o dataset carregado. "
        "As respostas são fundamentadas nos dados disponíveis."
    )

    for msg in session.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pergunte sobre os dados..."):
        session.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        overview = info.overview or {}
        context = (
            f"Dataset: {info.name}\n"
            f"Ficheiro: {info.source_filename}\n"
            f"Linhas: {overview.get('row_count', '?')}\n"
            f"Colunas: {overview.get('column_count', '?')}\n"
            f"Nomes das colunas: {[c['column'] for c in overview.get('columns', [])]}\n"
            f"Duplicados: {overview.get('duplicate_rows', '?')}\n"
        )

        quality = info.quality_report
        if quality and quality.get("issues"):
            top_issues = quality["issues"][:5]
            context += "\nPrincipais problemas de qualidade:\n"
            for issue in top_issues:
                context += f"- {issue['category']} em {issue['column']}: {issue['pct']}%\n"

        user_prompt = f"Contexto do dataset:\n{context}\n\nPergunta do utilizador: {prompt}"

        provider = create_provider_for_task(LLMTask.CHAT)
        with st.chat_message("assistant"):
            if not provider.is_available():
                response_text = (
                    "Não foi possível responder — o Ollama não está disponível. "
                    "Verifique a configuração na barra lateral."
                )
                st.warning(response_text)
            else:
                with st.spinner("A analisar..."):
                    try:
                        response = provider.complete(SYSTEM_PROMPT, user_prompt)
                        response_text = response.content
                        st.markdown(response_text)
                    except Exception as exc:
                        response_text = f"Erro ao contactar o backend de IA: {exc}"
                        st.error(response_text)

        session.chat_history.append({"role": "assistant", "content": response_text})


def render_dictionary_tab(info) -> None:
    st.caption(
        "Dicionário de dados gerado por IA (Ollama local). "
        "Interpretações de negócio são **sugestões** — pode editar e guardar."
    )

    if st.button("Gerar / Regenerar dicionário", type="primary"):
        provider = create_provider_for_task(LLMTask.DICTIONARY)
        if not provider.is_available():
            st.error(llm_error_message())
        else:
            if getattr(provider, "resolve_warning", None):
                st.warning(provider.resolve_warning)
            with st.spinner("A gerar dicionário de dados..."):
                try:
                    info.data_dictionary = generate_dictionary(provider, info.df)
                    st.success("Dicionário gerado com sucesso.")
                except Exception as exc:
                    st.error(f"Erro ao gerar dicionário: {exc}")

    if not info.data_dictionary:
        st.info("Clique em 'Gerar dicionário' para criar o dicionário de dados.")
        return

    dict_df = pd.DataFrame(info.data_dictionary)
    display_df = dict_df.rename(
        columns={
            "column": "Coluna",
            "description": "Descrição",
            "dtype": "Tipo",
            "sample_value": "Exemplo",
            "business_meaning": "Significado de Negócio (sugestão)",
        }
    )
    edited = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Coluna", "Tipo", "Exemplo"],
        key=f"dict_editor_{info.name}",
    )
    if st.button("Guardar alterações ao dicionário", key=f"save_dict_{info.name}"):
        col_map = {
            "Coluna": "column",
            "Descrição": "description",
            "Tipo": "dtype",
            "Exemplo": "sample_value",
            "Significado de Negócio (sugestão)": "business_meaning",
        }
        info.data_dictionary = [
            {col_map[k]: row[k] for k in col_map if k in edited.columns}
            for _, row in edited.iterrows()
        ]
        st.success("Dicionário atualizado.")

    st.caption("⚠️ Campos de significado de negócio são sugestões geradas por IA.")

    st.download_button(
        "Exportar dicionário (CSV)",
        edited.to_csv(index=False).encode("utf-8"),
        file_name=f"{info.name}_dictionary.csv",
        mime="text/csv",
    )


def render_sql_tab(session: AppSession) -> None:
    datasets = {name: info.df for name, info in session.datasets.items()}
    schema_context = build_schema_context(datasets)

    with st.expander("Schema disponível"):
        st.code(schema_context, language=None)

    col_gen, col_explain = st.columns(2)

    with col_gen:
        st.subheader("Gerar SQL")
        nl_request = st.text_area(
            "Pedido em linguagem natural",
            placeholder="Ex: Mostrar os 10 clientes com mais encomendas",
            height=100,
            key="sql_nl_request",
        )
        if st.button("Gerar SQL", key="btn_gen_sql"):
            provider = create_provider_for_task(LLMTask.SQL)
            if not provider.is_available():
                st.error(llm_error_message())
            elif not nl_request.strip():
                st.warning("Introduza um pedido.")
            else:
                with st.spinner("A gerar query..."):
                    try:
                        result = generate_sql(provider, schema_context, nl_request)
                        st.session_state.generated_sql = result["sql"]
                        if result.get("syntax_error"):
                            st.warning(f"Aviso de sintaxe: {result['syntax_error']}")
                    except Exception as exc:
                        st.error(f"Erro: {exc}")

        generated = st.session_state.get("generated_sql", "")
        sql_to_run = st.text_area("Query SQL", value=generated, height=150, key="sql_editor")

        if st.button("Executar query", key="btn_run_sql"):
            conn = duckdb.connect(":memory:")
            register_datasets(conn, datasets)
            exec_result = execute_sql(conn, sql_to_run)
            if exec_result["success"] and exec_result["df"] is not None:
                st.dataframe(exec_result["df"], use_container_width=True)
            else:
                st.error(exec_result["error"])

    with col_explain:
        st.subheader("Explicar SQL")
        sql_explain_input = st.text_area(
            "Cole uma query SQL",
            height=150,
            key="sql_explain_input",
        )
        if st.button("Explicar", key="btn_explain_sql"):
            provider = create_provider_for_task(LLMTask.SQL)
            if not provider.is_available():
                st.error(llm_error_message())
            elif not sql_explain_input.strip():
                st.warning("Introduza uma query SQL.")
            else:
                with st.spinner("A analisar query..."):
                    result = explain_sql(provider, sql_explain_input)
                    if result["error"]:
                        st.error(result["error"])
                    else:
                        st.markdown(result["explanation"])

    st.divider()
    st.subheader("Otimizar SQL")
    sql_optimize_input = st.text_area(
        "Query SQL para otimizar",
        height=120,
        key="sql_optimize_input",
    )
    if st.button("Analisar otimizações", key="btn_optimize_sql"):
        provider = create_provider_for_task(LLMTask.SQL)
        if not provider.is_available():
            st.error(llm_error_message())
        elif not sql_optimize_input.strip():
            st.warning("Introduza uma query SQL.")
        else:
            with st.spinner("A analisar otimizações..."):
                result = optimize_sql(provider, sql_optimize_input)
                if result["error"]:
                    st.error(result["error"])
                else:
                    st.session_state.sql_optimizations = result

    opt_result = st.session_state.get("sql_optimizations")
    if opt_result and opt_result.get("suggestions"):
        for suggestion in opt_result["suggestions"]:
            with st.expander(
                f"#{suggestion.get('id', '?')} — {suggestion.get('issue', 'Sugestão')}"
            ):
                st.write(f"**Motivo:** {suggestion.get('reason', '')}")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("Original")
                    st.code(suggestion.get("original_snippet", ""), language="sql")
                with col_b:
                    st.caption("Otimizado")
                    st.code(suggestion.get("optimized_snippet", ""), language="sql")

        if opt_result.get("optimized_sql"):
            st.subheader("Query otimizada")
            st.code(opt_result["optimized_sql"], language="sql")


def render_mapping_tab(session: AppSession) -> None:
    if len(session.datasets) < 2:
        st.info("Carregue pelo menos 2 datasets para comparar e mapear colunas.")
        return

    names = list(session.datasets.keys())
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Dataset origem", names, key="map_source")
    with col2:
        target_options = [n for n in names if n != source]
        target = st.selectbox("Dataset destino", target_options, key="map_target")

    map_key = session.mapping_key(source, target)

    if st.button("Gerar mapeamento", type="primary", key="btn_map"):
        src_info = session.datasets[source]
        tgt_info = session.datasets[target]
        session.mappings[map_key] = propose_mappings(
            src_info.df, tgt_info.df, source, target
        )

    mapping = session.mappings.get(map_key)
    if not mapping:
        st.info("Clique em 'Gerar mapeamento' para comparar os datasets.")
        return

    st.subheader("Mapeamento sugerido")
    map_df = pd.DataFrame(mapping["mappings"])
    edited = st.data_editor(
        map_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "status": st.column_config.SelectboxColumn(
                "status",
                options=["pendente", "aceite", "rejeitado", "sem correspondência"],
            ),
            "target_column": st.column_config.TextColumn("target_column"),
        },
        disabled=["source_column", "confidence", "source_type", "target_type", "is_suggestion"],
        key=f"map_editor_{source}_{target}",
    )
    if st.button("Guardar mapeamento", key=f"save_map_{source}_{target}"):
        mapping["mappings"] = edited.to_dict(orient="records")
        session.mappings[map_key] = mapping
        st.success("Mapeamento guardado.")
    st.caption("⚠️ Mapeamentos são sugestões — aceite, rejeite ou corrija manualmente.")

    if mapping.get("discrepancies"):
        st.subheader("Discrepâncias estruturais")
        for disc in mapping["discrepancies"]:
            icon = {"coluna_em_falta_no_destino": "❌", "coluna_extra_no_destino": "➕", "transformacao_necessaria": "🔄"}.get(
                disc["type"], "ℹ️"
            )
            st.write(f"{icon} **{disc['column']}** — {disc['action']}")

    if mapping.get("non_obvious_mappings"):
        st.subheader("Mapeamentos não óbvios")
        st.dataframe(
            pd.DataFrame(mapping["non_obvious_mappings"]),
            use_container_width=True,
            hide_index=True,
        )


def render_business_rules_tab(info) -> None:
    st.caption(
        "Regras detetadas são **hipóteses a validar**, não regras confirmadas. "
        "Cada regra indica a % de registos que a suportam."
    )

    if st.button("Detetar regras de negócio", type="primary", key="btn_rules"):
        provider = create_provider_for_task(LLMTask.BUSINESS_RULES)
        with st.spinner("A analisar padrões..."):
            info.business_rules = detect_business_rules(info.df, provider)

    if not info.business_rules:
        st.info("Clique no botão para detetar regras de negócio implícitas.")
        return

    rules_df = pd.DataFrame(info.business_rules)
    display = rules_df[["rule", "columns", "evidence_pct", "detail", "source"]].rename(
        columns={
            "rule": "Regra (hipótese)",
            "columns": "Colunas",
            "evidence_pct": "% Suporte",
            "detail": "Detalhe",
            "source": "Origem",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("⚠️ Todas as regras são hipóteses — validar com stakeholders de negócio.")


def render_kpi_tab(info) -> None:
    st.caption(
        "KPIs sugeridos com base na estrutura do dataset. "
        "Verifique se as colunas necessárias estão disponíveis."
    )

    if st.button("Sugerir KPIs", type="primary", key="btn_kpis"):
        provider = create_provider_for_task(LLMTask.KPI)
        with st.spinner("A sugerir KPIs..."):
            info.kpi_suggestions = suggest_kpis(info.df, provider)

    if not info.kpi_suggestions:
        st.info("Clique no botão para gerar sugestões de KPIs.")
        return

    for kpi in info.kpi_suggestions:
        avail_icon = "✅" if kpi.get("columns_available") else "❌"
        with st.expander(f"{kpi.get('name', 'KPI')} — {kpi.get('domain', '')}"):
            st.write(f"**Descrição:** {kpi.get('description', '')}")
            st.code(kpi.get("formula", ""), language="sql")
            st.write(f"**Colunas necessárias:** {kpi.get('required_columns', [])}")
            st.write(f"**Colunas disponíveis no dataset:** {avail_icon}")


def render_relationships_tab(session: AppSession) -> None:
    if len(session.datasets) < 2:
        st.info("Carregue pelo menos 2 datasets para detetar relações.")
        return

    st.caption(
        "Relações sugeridas com base em nomes de colunas e sobreposição de valores. "
        "Todas as sugestões devem ser validadas."
    )

    datasets = {name: info.df for name, info in session.datasets.items()}
    relationships = discover_relationships(datasets)

    if not relationships:
        st.warning("Nenhuma relação provável detetada com confiança suficiente.")
        return

    rel_df = pd.DataFrame(relationships)
    display_df = rel_df[
        [
            "dataset_a",
            "column_a",
            "dataset_b",
            "column_b",
            "confidence",
            "confidence_level",
            "value_overlap_pct",
            "join_hint",
        ]
    ].rename(
        columns={
            "dataset_a": "Dataset A",
            "column_a": "Coluna A",
            "dataset_b": "Dataset B",
            "column_b": "Coluna B",
            "confidence": "Confiança (%)",
            "confidence_level": "Nível",
            "value_overlap_pct": "Overlap valores (%)",
            "join_hint": "Sugestão JOIN",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("⚠️ Sugestões geradas automaticamente — não são relações confirmadas.")


def render_reports_tab(info) -> None:
    st.subheader("Relatório Consolidado")
    report_md = generate_markdown_report(info)
    st.markdown(report_md)

    st.divider()
    st.subheader("Exportar")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            "Markdown (.md)",
            report_md.encode("utf-8"),
            file_name=export_filename(info.name, "md"),
            mime="text/markdown",
        )
    with col2:
        try:
            excel_bytes = export_to_excel(info)
            st.download_button(
                "Excel (.xlsx)",
                excel_bytes,
                file_name=export_filename(info.name, "xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as exc:
            st.error(f"Erro Excel: {exc}")
    with col3:
        try:
            pdf_bytes = export_to_pdf(info)
            st.download_button(
                "PDF (.pdf)",
                pdf_bytes,
                file_name=export_filename(info.name, "pdf"),
                mime="application/pdf",
            )
        except Exception as exc:
            st.error(f"Erro PDF: {exc}")
    with col4:
        try:
            word_bytes = export_to_word(info)
            st.download_button(
                "Word (.docx)",
                word_bytes,
                file_name=export_filename(info.name, "docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception as exc:
            st.error(f"Erro Word: {exc}")


def main() -> None:
    session = init_session()
    inject_professional_theme(BRAND_COLOR)

    active = session.get_active()
    if active and st.session_state.get("presentation_mode"):
        render_presentation_mode(
            active,
            session,
            get_presentation_config(session),
        )
        return

    render_sidebar(session)

    active = session.get_active()
    if active is None:
        render_page_header(
            "DataMind AI",
            "Carregue um dataset na barra lateral para iniciar a análise.",
            breadcrumb="Início",
        )
        st.markdown(
            """
            ### Como começar
            1. **Carregar dados** — CSV, Excel ou JSON na sidebar
            2. **Dashboard Visual** — visão gráfica automática
            3. **Explorar** — qualidade, estatísticas, relações
            4. **Assistentes IA** — dicionário, SQL, chat (Ollama local)
            5. **Guardar projeto** — retomar análise mais tarde
            """
        )
        return

    page_id = st.session_state.get("nav_page", "dashboard")
    page = get_page(page_id) or get_page("dashboard")
    assert page is not None

    if page.id == "dashboard":
        render_visual_dashboard_tab(active)
    elif page.id == "drilldown":
        render_drilldown_tab(active, session)
    elif page.id == "overview":
        render_overview_tab(active)
    elif page.id == "statistics":
        render_statistics_tab(active)
    elif page.id == "quality":
        render_page_header(page.title, page.description, page.section, active.name)
        render_quality_tab(active)
    elif page.id == "relationships":
        render_page_header(page.title, page.description, page.section, active.name)
        render_relationships_tab(session)
    elif page.id == "mapping":
        render_page_header(page.title, page.description, page.section, active.name)
        render_mapping_tab(session)
    elif page.id == "dictionary":
        render_page_header(page.title, page.description, page.section, active.name)
        render_dictionary_tab(active)
    elif page.id == "sql":
        render_page_header(page.title, page.description, page.section, active.name)
        render_sql_tab(session)
    elif page.id == "rules":
        render_page_header(page.title, page.description, page.section, active.name)
        render_business_rules_tab(active)
    elif page.id == "kpis":
        render_page_header(page.title, page.description, page.section, active.name)
        render_kpi_tab(active)
    elif page.id == "chat":
        render_page_header(page.title, page.description, page.section, active.name)
        render_chat_tab(active, session)
    elif page.id == "reports":
        render_page_header(page.title, page.description, page.section, active.name)
        render_reports_tab(active)


if __name__ == "__main__":
    main()
