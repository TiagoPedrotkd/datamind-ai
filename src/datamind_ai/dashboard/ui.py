from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datamind_ai.dashboard.chart_catalog import CHART_CATALOG, ChartMeta
from datamind_ai.dashboard.chart_theme import apply_chart_theme
from datamind_ai.dashboard.charts import (
    export_figure_png,
    fig_boxplot,
    fig_column_comparison,
    fig_dataset_column_comparison,
    fig_correlation_matrix,
    fig_dtype_distribution,
    fig_histogram,
    fig_missing_heatmap,
    fig_scatter,
    fig_top_categories,
    overview_kpis,
)
from datamind_ai.dashboard.columns import (
    categorical_columns,
    numeric_columns,
    select_highlight_columns,
)
from datamind_ai.dashboard.interpretation import (
    _summarize_column,
    build_chart_context,
    interpret_chart,
)
from datamind_ai.dashboard.labels import build_label_map, display_label
from datamind_ai.llm.factory import create_provider_for_task
from datamind_ai.llm.tasks import LLMTask
from datamind_ai.session import AppSession, DatasetInfo

CORRELATION_DISCLAIMER = (
    "Correlação mede associação linear, **não causalidade**. "
    "Valide sempre com o contexto de negócio antes de tirar conclusões."
)

DEFAULT_VISIBLE_WIDGETS = [
    "overview_kpis",
    "overview_dtypes",
    "overview_missing_heatmap",
    "overview_numeric_histograms",
    "overview_categorical_bars",
    "drilldown_correlation",
]

WIDGET_LABELS = {
    "overview_kpis": "KPIs de resumo",
    "overview_dtypes": "Composição por tipo de dados",
    "overview_missing_heatmap": "Mapa de completude",
    "overview_numeric_histograms": "Distribuições numéricas",
    "overview_categorical_bars": "Categorias frequentes",
    "drilldown_correlation": "Matriz de correlação",
    "drilldown_distribution": "Distribuição por coluna",
    "drilldown_boxplot": "Box plot (outliers)",
    "drilldown_scatter": "Relação entre variáveis",
    "drilldown_comparison": "Comparação lado a lado",
}


@dataclass
class PresentationConfig:
    visible_widgets: list[str] = field(default_factory=lambda: list(DEFAULT_VISIBLE_WIDGETS))
    brand_color: str = "#1f4e79"
    presentation_title: str = "Resumo de Dados"
    company_name: str = "DataMind AI"


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    filtered = df.copy()
    for col, spec in filters.items():
        if col not in filtered.columns:
            continue
        if spec.get("type") == "categorical" and spec.get("values"):
            filtered = filtered[filtered[col].astype(str).isin(spec["values"])]
        elif spec.get("type") == "numeric_range":
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None:
                filtered = filtered[filtered[col] >= lo]
            if hi is not None:
                filtered = filtered[filtered[col] <= hi]
    return filtered


def render_business_context_editor(info: DatasetInfo) -> None:
    """Caixa de contexto de negócio — melhora interpretações IA dos gráficos."""
    key = f"business_context_{info.name}"
    if key not in st.session_state:
        st.session_state[key] = info.business_context or ""

    st.markdown(
        """
        <div class="dm-business-context">
            <h3>📝 Contexto de negócio</h3>
            <p>Descreva o assunto do dataset, o cliente e o objetivo da análise.
            A IA usa este texto ao interpretar os gráficos abaixo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_area(
        "Contexto de negócio",
        key=key,
        height=100,
        placeholder=(
            "Ex.: Dataset de deteção de fraude bancária (2024). "
            "Coluna «is_fraud» = transação fraudulenta (0/1). "
            "Objetivo: perceber padrões nas transações suspeitas."
        ),
        label_visibility="collapsed",
    )

    info.business_context = st.session_state[key]


def sync_business_context_from_state(session: AppSession) -> None:
    for name, info in session.datasets.items():
        key = f"business_context_{name}"
        if key in st.session_state:
            info.business_context = st.session_state[key]


def render_chart_panel(
    fig: go.Figure,
    meta: ChartMeta,
    chart_key: str,
    *,
    df: pd.DataFrame | None = None,
    chart_type: str | None = None,
    interpretation_kwargs: dict[str, Any] | None = None,
    column_label: str | None = None,
    brand_color: str = "#1f4e79",
    show_export: bool = True,
    enable_ai_interpretation: bool = True,
    business_context: str = "",
) -> None:
    themed = apply_chart_theme(
        fig, meta, brand_color=brand_color, column_label=column_label, show_title=False,
    )

    st.markdown('<div class="dm-chart-card">', unsafe_allow_html=True)
    if meta.description:
        chart_title = meta.title + (f" — {column_label}" if column_label else "")
        st.markdown(f'<p class="dm-chart-title">{chart_title}</p>', unsafe_allow_html=True)
        st.caption(meta.description)
    st.plotly_chart(themed, use_container_width=True, key=f"chart_{chart_key}")

    if enable_ai_interpretation and df is not None and chart_type:
        ctx = build_chart_context(
            chart_type,
            df,
            title=meta.title if not column_label else f"{meta.title} — {column_label}",
            **(interpretation_kwargs or {}),
        )
        cache_key = f"chart_ai_interp_{chart_key}_{hashlib.sha256(business_context.encode()).hexdigest()[:8]}"

        btn_col, _ = st.columns([1, 3])
        with btn_col:
            interpret_clicked = st.button(
                "Interpretar com IA",
                key=f"btn_interp_{chart_key}",
                help="A IA analisa os dados deste gráfico com o contexto de negócio (Ollama local)",
            )

        if interpret_clicked:
            provider = create_provider_for_task(LLMTask.CHAT)
            if not provider.is_available():
                st.error("Ollama não disponível. Verifique a configuração na sidebar.")
            else:
                with st.spinner("A gerar interpretação..."):
                    try:
                        st.session_state[cache_key] = interpret_chart(
                            provider, ctx, business_context=business_context,
                        )
                    except Exception as exc:
                        st.session_state[cache_key] = f"Erro: {exc}"

        if cache_key in st.session_state:
            with st.expander("Interpretação IA", expanded=True):
                st.markdown(st.session_state[cache_key])
                st.caption(
                    "Interpretação gerada por IA com base nos dados estatísticos do gráfico. "
                    "Validar sempre no contexto de negócio."
                )

    if show_export:
        col1, _ = st.columns([1, 4])
        with col1:
            try:
                png = export_figure_png(themed)
                st.download_button(
                    "Exportar PNG",
                    png,
                    file_name=f"datamind_{chart_key}.png",
                    mime="image/png",
                    key=f"dl_{chart_key}",
                    use_container_width=True,
                )
            except Exception:
                st.caption("Instale `kaleido` para exportar PNG.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_overview_dashboard(
    info: DatasetInfo,
    *,
    technical: bool = True,
    show_all_columns: bool = False,
    visible_widgets: set[str] | None = None,
    brand_color: str = "#1f4e79",
) -> None:
    overview = info.overview
    if overview is None:
        st.warning("Dashboard indisponível — carregue um dataset.")
        return

    label_map = build_label_map(info.data_dictionary)
    visible = visible_widgets or set(WIDGET_LABELS.keys())

    if technical:
        render_business_context_editor(info)
    biz_ctx = (info.business_context or "").strip()
    chart = partial(render_chart_panel, business_context=biz_ctx)

    if "overview_kpis" in visible:
        kpis = overview_kpis(overview)
        cols = st.columns(len(kpis))
        for col, (label, value) in zip(cols, kpis.items()):
            col.metric(label, value)

    highlight_num, highlight_cat = select_highlight_columns(
        info.df, overview.get("columns", [])
    )
    heatmap_cols = [str(c) for c in info.df.columns]
    if not show_all_columns and len(heatmap_cols) > 20:
        heatmap_cols = highlight_num + highlight_cat
        st.info(
            f"A mostrar **{len(heatmap_cols)}** colunas mais relevantes "
            f"(variabilidade e qualidade). Ative «Ver todas as colunas» para o conjunto completo."
        )

    if "overview_dtypes" in visible:
        chart(
            fig_dtype_distribution(overview),
            CHART_CATALOG["dtypes"],
            "dtypes",
            df=info.df,
            chart_type="dtypes",
            interpretation_kwargs={"overview": overview},
            brand_color=brand_color,
            show_export=technical,
            enable_ai_interpretation=technical,
        )

    if "overview_missing_heatmap" in visible and heatmap_cols:
        labels = [display_label(c, label_map, technical) for c in heatmap_cols]
        chart(
            fig_missing_heatmap(info.df, heatmap_cols, row_labels=labels),
            CHART_CATALOG["missing_heatmap"],
            "missing_heatmap",
            df=info.df,
            chart_type="missing_heatmap",
            interpretation_kwargs={"columns": heatmap_cols},
            brand_color=brand_color,
            show_export=technical,
            enable_ai_interpretation=technical,
        )

    num_cols = numeric_columns(info.df) if show_all_columns else highlight_num
    if "overview_numeric_histograms" in visible and num_cols:
        st.markdown("#### Distribuições numéricas")
        for col in num_cols:
            col_label = display_label(col, label_map, technical)
            chart(
                fig_histogram(info.df, col, label_map, technical),
                CHART_CATALOG["histogram"],
                f"hist_{col}",
                df=info.df,
                chart_type="histogram",
                interpretation_kwargs={"column": col},
                column_label=col_label,
                brand_color=brand_color,
                show_export=technical,
                enable_ai_interpretation=technical,
            )

    cat_cols = categorical_columns(info.df) if show_all_columns else highlight_cat
    if "overview_categorical_bars" in visible and cat_cols:
        st.markdown("#### Frequência de categorias")
        for col in cat_cols:
            col_label = display_label(col, label_map, technical)
            if info.df[col].nunique() > 100:
                st.caption(f"{col_label}: a mostrar top 10 de {info.df[col].nunique()} categorias")
            chart(
                fig_top_categories(info.df, col, label_map, technical),
                CHART_CATALOG["categories"],
                f"cat_{col}",
                df=info.df,
                chart_type="categories",
                interpretation_kwargs={"column": col},
                column_label=col_label,
                brand_color=brand_color,
                show_export=technical,
                enable_ai_interpretation=technical,
            )


def _column_selector(
    label: str,
    options: list[str],
    label_map: dict[str, str],
    technical: bool,
    key: str,
) -> str:
    if not options:
        return ""
    display_options = options if technical else [
        display_label(c, label_map, False) for c in options
    ]
    idx = st.selectbox(
        label,
        range(len(options)),
        format_func=lambda i: display_options[i],
        key=key,
    )
    return options[idx]


def render_drilldown_dashboard(
    info: DatasetInfo,
    session: AppSession,
    *,
    technical: bool = True,
    visible_widgets: set[str] | None = None,
    brand_color: str = "#1f4e79",
) -> pd.DataFrame:
    df = info.df
    label_map = build_label_map(info.data_dictionary)
    visible = visible_widgets or set(WIDGET_LABELS.keys())
    filter_key = f"filters_{info.name}"

    if technical:
        render_business_context_editor(info)
    biz_ctx = (info.business_context or "").strip()
    chart = partial(render_chart_panel, business_context=biz_ctx)

    if filter_key not in st.session_state:
        st.session_state[filter_key] = {}

    with st.expander("Filtros dinâmicos", expanded=False):
        st.caption("Refine a análise — todos os gráficos abaixo atualizam automaticamente.")
        filter_cols = st.multiselect(
            "Colunas para filtrar",
            [str(c) for c in df.columns],
            format_func=lambda c: display_label(c, label_map, technical),
            key=f"filter_cols_{info.name}",
        )
        active_filters: dict[str, Any] = {}
        for col in filter_cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                lo, hi = float(series.min()), float(series.max())
                if lo == hi:
                    continue
                selected = st.slider(
                    display_label(col, label_map, technical),
                    lo, hi, (lo, hi),
                    key=f"filter_num_{info.name}_{col}",
                )
                active_filters[col] = {"type": "numeric_range", "min": selected[0], "max": selected[1]}
            else:
                options = series.dropna().astype(str).unique().tolist()[:50]
                chosen = st.multiselect(
                    display_label(col, label_map, technical),
                    options, default=options,
                    key=f"filter_cat_{info.name}_{col}",
                )
                active_filters[col] = {"type": "categorical", "values": chosen}
        st.session_state[filter_key] = active_filters

    filtered = apply_filters(df, st.session_state[filter_key])
    st.metric("Registos em análise", f"{len(filtered):,}", delta=f"{len(filtered) - len(df):+,}" if len(filtered) != len(df) else None)

    num_cols = numeric_columns(filtered)
    all_cols = [str(c) for c in filtered.columns]

    if "drilldown_correlation" in visible and len(num_cols) >= 2:
        st.markdown("#### Matriz de correlação")
        st.caption(CORRELATION_DISCLAIMER)
        corr_cols = st.multiselect(
            "Selecionar colunas numéricas",
            num_cols,
            default=num_cols[: min(8, len(num_cols))],
            format_func=lambda c: display_label(c, label_map, technical),
            key=f"corr_cols_{info.name}",
        )
        if len(corr_cols) >= 2:
            chart(
                fig_correlation_matrix(filtered, corr_cols, label_map, technical),
                CHART_CATALOG["correlation"],
                f"corr_{info.name}_{'_'.join(corr_cols[:3])}",
                df=filtered,
                chart_type="correlation",
                interpretation_kwargs={"columns": corr_cols},
                brand_color=brand_color,
                show_export=technical,
                enable_ai_interpretation=technical,
            )

    if "drilldown_distribution" in visible and all_cols:
        st.markdown("#### Explorar coluna")
        col = _column_selector(
            "Coluna",
            all_cols,
            label_map,
            technical,
            f"drill_col_{info.name}",
        )
        if col:
            col_label = display_label(col, label_map, technical)
            if col in numeric_columns(filtered):
                chart(
                    fig_histogram(filtered, col, label_map, technical),
                    CHART_CATALOG["histogram"],
                    f"drill_hist_{col}",
                    df=filtered,
                    chart_type="histogram",
                    interpretation_kwargs={"column": col},
                    column_label=col_label,
                    brand_color=brand_color,
                    show_export=technical,
                    enable_ai_interpretation=technical,
                )
            else:
                chart(
                    fig_top_categories(filtered, col, label_map, technical),
                    CHART_CATALOG["categories"],
                    f"drill_cat_{col}",
                    df=filtered,
                    chart_type="categories",
                    interpretation_kwargs={"column": col},
                    column_label=col_label,
                    brand_color=brand_color,
                    show_export=technical,
                    enable_ai_interpretation=technical,
                )

    if "drilldown_boxplot" in visible and num_cols:
        st.markdown("#### Outliers")
        box_col = _column_selector("Coluna numérica", num_cols, label_map, technical, f"box_col_{info.name}")
        if box_col:
            chart(
                fig_boxplot(filtered, box_col, label_map, technical),
                CHART_CATALOG["boxplot"],
                f"box_{box_col}",
                df=filtered,
                chart_type="boxplot",
                interpretation_kwargs={"column": box_col},
                column_label=display_label(box_col, label_map, technical),
                brand_color=brand_color,
                show_export=technical,
                enable_ai_interpretation=technical,
            )

    if "drilldown_scatter" in visible and len(num_cols) >= 2:
        st.markdown("#### Relação entre variáveis")
        c1, c2 = st.columns(2)
        with c1:
            x_col = _column_selector("Eixo X", num_cols, label_map, technical, f"scatter_x_{info.name}")
        with c2:
            y_options = [c for c in num_cols if c != x_col] or num_cols
            y_col = _column_selector("Eixo Y", y_options, label_map, technical, f"scatter_y_{info.name}")
        if x_col and y_col and x_col != y_col:
            chart(
                fig_scatter(filtered, x_col, y_col, label_map, technical),
                CHART_CATALOG["scatter"],
                f"scatter_{x_col}_{y_col}",
                df=filtered,
                chart_type="scatter",
                interpretation_kwargs={"x_col": x_col, "y_col": y_col},
                column_label=f"{display_label(x_col, label_map, technical)} × {display_label(y_col, label_map, technical)}",
                brand_color=brand_color,
                show_export=technical,
                enable_ai_interpretation=technical,
            )

    if "drilldown_comparison" in visible and len(all_cols) >= 2:
        st.markdown("#### Comparação de colunas")
        cc1, cc2 = st.columns(2)
        with cc1:
            col_a = _column_selector("Coluna A", all_cols, label_map, technical, f"cmp_a_{info.name}")
        with cc2:
            col_b_opts = [c for c in all_cols if c != col_a] or all_cols
            col_b = _column_selector("Coluna B", col_b_opts, label_map, technical, f"cmp_b_{info.name}")
        if col_a and col_b:
            chart(
                fig_column_comparison(filtered, col_a, col_b, label_map, technical),
                CHART_CATALOG["comparison"],
                f"cmp_{col_a}_{col_b}",
                df=filtered,
                chart_type="comparison",
                interpretation_kwargs={"col_a": col_a, "col_b": col_b},
                brand_color=brand_color,
                show_export=technical,
                enable_ai_interpretation=technical,
            )

        if len(session.datasets) >= 2:
            st.markdown("#### Comparação entre datasets")
            names = list(session.datasets.keys())
            dc1, dc2 = st.columns(2)
            with dc1:
                ds_a = st.selectbox("Dataset A", names, key=f"ds_a_{info.name}")
            with dc2:
                ds_b = st.selectbox("Dataset B", [n for n in names if n != ds_a] or names, key=f"ds_b_{info.name}")
            common = [c for c in session.datasets[ds_a].df.columns if c in session.datasets[ds_b].df.columns]
            if common:
                shared = _column_selector("Coluna comum", [str(c) for c in common], label_map, technical, f"ds_shared_{info.name}")
                df_a = session.datasets[ds_a].df
                df_b = session.datasets[ds_b].df
                chart(
                    fig_dataset_column_comparison(
                        df_a, df_b, shared, ds_a, ds_b, label_map, technical,
                    ),
                    CHART_CATALOG["dataset_comparison"],
                    f"ds_cmp_{ds_a}_{ds_b}_{shared}",
                    df=df_a,
                    chart_type="dataset_comparison",
                    interpretation_kwargs={
                        "column": shared,
                        "dataset_a": ds_a,
                        "dataset_b": ds_b,
                        "stats_b": _summarize_column(df_b, shared),
                        "rows_b": len(df_b),
                    },
                    column_label=display_label(shared, label_map, technical),
                    brand_color=brand_color,
                    show_export=technical,
                    enable_ai_interpretation=technical,
                )

    return filtered


def render_presentation_config_sidebar(config: PresentationConfig) -> PresentationConfig:
    config.presentation_title = st.text_input("Título", value=config.presentation_title)
    config.company_name = st.text_input("Nome / empresa", value=config.company_name)
    config.brand_color = st.color_picker("Cor corporativa", value=config.brand_color)
    config.visible_widgets = st.multiselect(
        "Gráficos visíveis",
        options=list(WIDGET_LABELS.keys()),
        default=config.visible_widgets,
        format_func=lambda k: WIDGET_LABELS.get(k, k),
    )
    return config


def render_presentation_mode(
    info: DatasetInfo,
    session: AppSession,
    config: PresentationConfig,
) -> None:
    from datamind_ai.ui.theme import inject_professional_theme

    inject_professional_theme(config.brand_color)
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        .main .block-container {padding-top: 1rem; max-width: 1200px;}
        .presentation-header {
            border-left: 6px solid var(--primary-color);
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
            border-radius: 0 8px 8px 0;
        }
        .presentation-header h1 {
            color: var(--text-color);
            margin: 0;
            font-size: 2rem;
        }
        .presentation-header p {
            color: var(--text-color);
            opacity: 0.85;
            margin: 0.5rem 0 0 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("← Voltar ao modo de trabalho"):
        st.session_state.presentation_mode = False
        st.rerun()

    st.markdown(
        f'<div class="presentation-header">'
        f'<h1>{config.presentation_title}</h1>'
        f'<p>{config.company_name}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    visible = set(config.visible_widgets)
    render_overview_dashboard(
        info, technical=False, show_all_columns=False,
        visible_widgets=visible, brand_color=config.brand_color,
    )
    if visible & {"drilldown_correlation", "drilldown_distribution", "drilldown_boxplot", "drilldown_scatter", "drilldown_comparison"}:
        st.divider()
        st.markdown("### Análise detalhada")
        render_drilldown_dashboard(
            info, session, technical=False, visible_widgets=visible, brand_color=config.brand_color,
        )

    st.divider()
    if st.button("Exportar apresentação (PDF)"):
        try:
            from datamind_ai.export.formats import export_to_pdf
            pdf = export_to_pdf(info)
            st.download_button("Descarregar PDF", pdf, file_name="apresentacao_datamind.pdf", mime="application/pdf")
        except Exception as exc:
            st.error(f"Não foi possível gerar PDF: {exc}")
