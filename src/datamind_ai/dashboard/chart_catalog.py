"""Metadados dos gráficos — título, descrição e interpretação."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChartMeta:
    title: str
    description: str
    interpretation: str
    x_label: str = ""
    y_label: str = ""


CHART_CATALOG: dict[str, ChartMeta] = {
    "dtypes": ChartMeta(
        title="Composição por tipo de dados",
        description="Proporção de colunas numéricas, texto, datas e outros tipos no dataset.",
        interpretation=(
            "Um dataset com muitas colunas de texto pode exigir normalização antes de análises "
            "estatísticas. Colunas de data permitem análises temporais."
        ),
        y_label="N.º de colunas",
    ),
    "missing_heatmap": ChartMeta(
        title="Mapa de completude dos dados",
        description="Visualização dos valores em falta por coluna e por registo (amostra).",
        interpretation=(
            "Células vermelhas indicam valores em falta. Padrões horizontais revelam colunas "
            "problemáticas; padrões verticais podem indicar registos incompletos."
        ),
        x_label="Registo (índice)",
        y_label="Coluna",
    ),
    "histogram": ChartMeta(
        title="Distribuição de valores",
        description="Frequência dos valores numa coluna numérica, com resumo estatístico (box plot).",
        interpretation=(
            "Permite identificar concentração de valores, assimetria e possíveis outliers. "
            "Compare a caixa central com a cauda para detetar extremos."
        ),
        x_label="Valor",
        y_label="Frequência",
    ),
    "categories": ChartMeta(
        title="Valores mais frequentes",
        description="As categorias com maior volume de registos nesta coluna.",
        interpretation=(
            "Dominância de uma categoria pode indicar regra de negócio implícita ou "
            "desbalanceamento nos dados de origem."
        ),
        x_label="Categoria",
        y_label="N.º de registos",
    ),
    "correlation": ChartMeta(
        title="Matriz de correlação",
        description="Grau de associação linear entre pares de colunas numéricas (−1 a +1).",
        interpretation=(
            "Valores próximos de +1 ou −1 indicam associação forte. "
            "**Correlação não implica causalidade** — validar sempre com contexto de negócio."
        ),
    ),
    "boxplot": ChartMeta(
        title="Distribuição e outliers",
        description="Mediana, quartis e valores extremos da coluna selecionada.",
        interpretation=(
            "Pontos fora dos bigodes são candidatos a outliers. Verifique se representam "
            "erros de dados ou casos de negócio legítimos."
        ),
        y_label="Valor",
    ),
    "scatter": ChartMeta(
        title="Relação entre duas variáveis",
        description="Cada ponto representa um registo; eixos são duas colunas numéricas.",
        interpretation=(
            "Padrões lineares sugerem associação; dispersão ampla indica relação fraca. "
            "Não assume relação de causa-efeito."
        ),
    ),
    "comparison": ChartMeta(
        title="Comparação de colunas",
        description="Distribuições lado a lado para comparar forma e escala.",
        interpretation="Útil para validar mapeamentos entre datasets ou colunas equivalentes.",
    ),
    "dataset_comparison": ChartMeta(
        title="Comparação entre datasets",
        description="Mesma coluna analisada em dois datasets diferentes.",
        interpretation="Diferenças de forma ou escala podem indicar transformações necessárias na integração.",
    ),
}
