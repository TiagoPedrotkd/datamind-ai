# DataMind AI

Assistente de análise de dados para uso **local no PC da empresa**. Todos os módulos de IA correm via **Ollama** — os dados dos clientes **nunca saem da máquina**.

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.com/) instalado e a correr

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env
```

### Configurar Ollama

```bash
ollama pull llama3

# Modelos especializados (recomendado)
ollama create datamind-chat -f modelfiles/datamind-chat.Modelfile
ollama create datamind-dictionary -f modelfiles/datamind-dictionary.Modelfile
ollama create datamind-sql -f modelfiles/datamind-sql.Modelfile
ollama create datamind-business-rules -f modelfiles/datamind-business-rules.Modelfile
ollama create datamind-kpi -f modelfiles/datamind-kpi.Modelfile
```

## Executar

```bash
streamlit run app.py
```

## Modo confidencial

Por defeito, `CONFIDENTIAL_MODE=true` no `.env`:

- **Apenas Ollama** — OpenAI bloqueado
- Dados processados **localmente** no PC
- Banner na app a confirmar que nada sai da máquina

## Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| Upload | CSV, Excel (todas as folhas), JSON |
| Projeto | Guardar/carregar análises em `projects/` |
| Visão geral | Estrutura, tipos, nulls, duplicados |
| Qualidade | Deteção automática de problemas |
| Dicionário | Gerado por IA, **editável** |
| Mapeamento | Comparação origem/destino, **editável** |
| SQL Assistant | Gerar, explicar, otimizar (DuckDB local) |
| Chat | Perguntas sobre os dados (Ollama) |
| Export | Markdown, Excel, PDF, Word |
| Dashboard Visual (FR-021) | KPIs, heatmap, histogramas interativos |
| Análise Detalhada (FR-022) | Filtros, correlação, drill-down, export PNG |
| Modo Apresentação (FR-023) | Vista executiva, gráficos selecionáveis |

## Datasets grandes

- Análises estatísticas usam o dataset completo
- Módulos de IA usam amostra de 500 linhas (configurável via `LLM_SAMPLE_ROWS`)
- Aviso automático acima de 100 000 linhas

## Estrutura

```
app.py
projects/              # Projetos guardados (não versionar — contém dados)
modelfiles/            # Perfis Ollama
src/datamind_ai/
```
