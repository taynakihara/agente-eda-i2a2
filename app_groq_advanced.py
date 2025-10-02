
from __future__ import annotations

import os
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Opcional (usado apenas no heatmap). Se não desejar a dependência, ver TODO abaixo
import seaborn as sns  # noqa: F401

try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None 

# =========================
# Configurações e Constantes
# =========================

APP_TITLE = "🤖 Agente de Análise de Dados (CSV)"
APP_SUBTITLE = "Ferramenta inteligente para análise de qualquer arquivo CSV com IA"
PAGE_CONFIG = dict(page_title="Análise de Dados CSV", page_icon="📊", layout="wide")

# Limites de renderização para não travar a UI em datasets grandes
MAX_NUMERIC_HISTS = 12
MAX_CATEGORICAL_BARS = 6
MAX_BOX_PLOTS = 8

# Amostragem para operações pesadas
SAMPLE_FOR_PLOTS = 100_000  # se o dataset for maior, faz uma amostra para gráficos
SAMPLE_FOR_STATS = 250_000  # amostra para estatísticas e correlação

DARK_BG = "#0E1117"


# =========================

st.markdown(
    """
    <h1 style='text-align: center; color: #00BFFF;'>
        🤖 Agente de Análise de Dados (CSV)
    </h1>
    <h3 style='text-align: center; color: #AAAAAA;'>
        Ferramenta inteligente para análise de qualquer arquivo CSV com IA
    </h3>
    """,
    unsafe_allow_html=True
)


# CSS customizado
st.markdown(
    """
    <style>
    /* Layout padrão (antes do upload) → centralizado */
    .block-container {
        max-width: 1100px;
        margin: auto;
    }

    /* Quando tiver tabelas/dataframes (após upload) → ocupa a tela toda */
    .stDataFrame, .stTable {
        max-width: 100% !important;
    }

    /* Ajusta responsividade das abas */
    div[data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
    }

    /* Ajusta largura dos elementos grandes */
    .element-container {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# =========================
# Funções utilitárias
# =========================

def _maybe_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Retorna amostra do DataFrame caso ele exceda max_rows (mantendo aleatoriedade)."""
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)
    return df


@st.cache_data(show_spinner=False)
def load_csv(file) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Lê CSV com detecção de separador e fallback de encoding.
    Evita o problema de carregar tudo como uma coluna só.
    """
    try:
        df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8", low_memory=False)
        return df, None
    except Exception:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine="python", encoding="latin1", low_memory=False)
            return df, None
        except Exception as e2:
            return pd.DataFrame(), f"Erro ao carregar CSV: {e2}"




@st.cache_data(show_spinner=False)
def get_overview(data: pd.DataFrame) -> Dict[str, object]:
    """Retorna dicionário com visão geral e tabelas auxiliares."""
    memory_mb = data.memory_usage(deep=True).sum() / 1024 ** 2
    tipos_dados = pd.DataFrame({
        "Coluna": data.columns,
        "Tipo": data.dtypes.astype(str).values,
        "Valores Únicos": [data[c].nunique(dropna=True) for c in data.columns],
        "Valores Nulos": [int(data[c].isna().sum()) for c in data.columns],
        "% Nulos": [f"{(data[c].isna().mean() * 100):.1f}%" for c in data.columns],
    })

    desc = None
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # amostra para acelerar describe em datasets gigantes
        data_stats = _maybe_sample(data[numeric_cols], SAMPLE_FOR_STATS)
        desc = data_stats.describe()

    return {
        "shape": data.shape,
        "memory_mb": memory_mb,
        "tipos_dados": tipos_dados,
        "desc": desc,
        "numeric_cols": numeric_cols,
        "categorical_cols": data.select_dtypes(include=["object", "category"]).columns.tolist(),
    }


def _setup_dark_axes(ax):
    """Aplica tema dark consistente em cada eixo."""
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, alpha=0.3)


def _new_fig(size=(12, 6)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(DARK_BG)
    return fig, ax


def _time_columns(data: pd.DataFrame) -> List[str]:
    """Detecta colunas temporais por nome e parse confiável."""
    candidates = [c for c in data.columns if any(w in c.lower() for w in ("time", "date", "timestamp", "data", "dt_"))]
    parsed = []
    for c in candidates:
        try:
            pd.to_datetime(data[c], errors="raise")  # valida parse
            parsed.append(c)
        except Exception:
            # tenta parse brando
            if pd.to_datetime(data[c], errors="coerce").notna().any():
                parsed.append(c)
    return list(dict.fromkeys(parsed))


def _safe_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação protegida contra constantes/NaN."""
    df2 = df.select_dtypes(include=[np.number]).copy()
    df2 = df2.loc[:, df2.nunique(dropna=True) > 1]  # remove colunas constantes
    if df2.empty or df2.shape[1] < 2:
        return pd.DataFrame()
    # amostra para acelerar
    df2 = _maybe_sample(df2, SAMPLE_FOR_STATS)
    return df2.corr(numeric_only=True)


def _top_frequencies(s: pd.Series, k: int = 10) -> pd.Series:
    """Top-k frequências (com limpeza de NaN)."""
    return s.dropna().astype(str).value_counts().head(k)


# =========================
# Seções de UI (tabs)
# =========================

def render_tab_overview(data: pd.DataFrame, overview: Dict[str, object]) -> None:
    st.header("📋 Visão Geral dos Dados")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Informações Básicas")
        st.write(f"**Número de linhas:** {overview['shape'][0]:,}")
        st.write(f"**Número de colunas:** {overview['shape'][1]:,}")
        st.write(f"**Tamanho em memória:** {overview['memory_mb']:.2f} MB")

        st.subheader("Tipos de Dados")
        st.dataframe(overview["tipos_dados"], use_container_width=True)

    with c2:
        st.subheader("Primeiras 10 Linhas")
        st.dataframe(data.head(10), use_container_width=True)

        st.subheader("Estatísticas Descritivas")
        if overview["desc"] is not None:
            st.dataframe(overview["desc"], use_container_width=True)
        else:
            st.info("Não há variáveis numéricas para descrever.")


def render_tab_distributions(data: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> None:
    st.header("📊 Distribuição das Variáveis")

    # -------- Numéricas (uma de cada vez) --------
    if numeric_cols:
        st.subheader("Variáveis Numéricas")
        col_select_num = st.selectbox(
            "Selecione uma variável numérica:",
            numeric_cols,
            key="dist_num_select"  # chave única
        )
        with st.spinner(f"⏳ Gerando histograma de {col_select_num}..."):
            data_plot = _maybe_sample(data[[col_select_num]], SAMPLE_FOR_PLOTS)
            Q1, Q3 = data_plot[col_select_num].quantile([0.01, 0.99])
            filtered = data_plot[col_select_num].clip(lower=Q1, upper=Q3)

            fig, ax = _new_fig((10, 6))
            ax.hist(filtered.dropna(), bins=30, alpha=0.7, edgecolor="white")
            ax.set_title(f"Distribuição: {col_select_num}", color="white", fontsize=14)
            _setup_dark_axes(ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Não há variáveis numéricas no dataset.")

    # -------- Categóricas (uma de cada vez) --------
    if categorical_cols:
        st.subheader("Variáveis Categóricas")
        col_select_cat = st.selectbox(
            "Selecione uma variável categórica:",
            categorical_cols,
            key="dist_cat_select"  # chave única
        )

        with st.spinner(f"⏳ Gerando barras de {col_select_cat}..."):
            vc = _top_frequencies(data[col_select_cat])

            fig, ax = _new_fig((10, 6))
            bars = ax.bar(range(len(vc)), vc.values, alpha=0.85)
            ax.set_title(f"Distribuição: {col_select_cat}", color="white", fontsize=14)
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels(vc.index, rotation=45, ha="right", color="white")
            _setup_dark_axes(ax)

            if len(vc) > 0:
                ymax = float(vc.values.max())
                for b, v in zip(bars, vc.values):
                    ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.01 * ymax, f"{v:,}",
                            ha="center", va="bottom", color="white", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Não há variáveis categóricas no dataset.")


        # Categóricas
    if categorical_cols:
        st.subheader("Variáveis Categóricas")
        col_select_cat = st.selectbox("Selecione uma variável categórica:", categorical_cols)
        vc = _top_frequencies(data[col_select_cat])

        fig, ax = _new_fig((10, 6))
        bars = ax.bar(range(len(vc)), vc.values, alpha=0.85)
        ax.set_title(f"Distribuição: {col_select_cat}", color="white", fontsize=14)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=45, ha="right", color="white")
        _setup_dark_axes(ax)

        for b, v in zip(bars, vc.values):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height() + 0.01 * vc.values.max(),
                f"{v:,}",
                ha="center", va="bottom", color="white", fontsize=9
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()




def render_tab_correlations(data: pd.DataFrame, numeric_cols: List[str]) -> None:
    st.header("🔍 Correlações entre Variáveis")

    if len(numeric_cols) < 2:
        st.info("É necessário ter pelo menos 2 variáveis numéricas para calcular correlações.")
        return

    corr = _safe_corr(data[numeric_cols])
    if corr.empty:
        st.info("Não foram encontradas correlações calculáveis (colunas constantes ou insuficientes).")
        return

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(DARK_BG)

    # TODO: Se quiser remover seaborn da stack, substitua por pcolormesh do matplotlib.
    sns.heatmap(
        corr,
        annot=True,
        cmap="RdYlBu_r",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Matriz de Correlação", color="white", fontsize=16, pad=20)
    ax.set_facecolor(DARK_BG)
    plt.xticks(rotation=45, ha="right", color="white")
    plt.yticks(rotation=0, color="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Top correlações
    st.subheader("Correlações Mais Significativas")
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and abs(v) > 0.1:
                pairs.append({
                    "Variável 1": cols[i],
                    "Variável 2": cols[j],
                    "Correlação": float(v),
                    "Força": "Forte" if abs(v) > 0.7 else ("Moderada" if abs(v) > 0.3 else "Fraca")
                })
    if pairs:
        df_pairs = pd.DataFrame(pairs).sort_values("Correlação", key=lambda s: s.abs(), ascending=False)
        st.dataframe(df_pairs, use_container_width=True)
    else:
        st.info("Não foram encontradas correlações significativas.")


def render_tab_trends(data: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> None:
    st.header("📈 Análise de Tendências")

    # -------- Tendências Temporais --------
    tcols = _time_columns(data)
    if tcols:
        st.subheader("Tendências Temporais")

        time_col = st.selectbox(
            "Selecione a coluna temporal:",
            tcols,
            key="trend_time_select"  # chave única
        )

        numeric_choices = [c for c in numeric_cols if c != time_col]
        if not numeric_choices:
            st.info("Não há variável numérica disponível diferente da coluna temporal selecionada.")
        else:
            numeric_col = st.selectbox(
                "Selecione a variável para análise temporal:",
                numeric_choices,
                key="trend_num_select"  # chave única
            )

            d = data.loc[:, [time_col, numeric_col]].copy()

            # Garante Series mesmo se houver duplicata de nome
            time_obj = d[time_col]
            if isinstance(time_obj, pd.DataFrame):
                time_obj = time_obj.iloc[:, 0]

            # Converte tempo (numérico -> segundos desde base; string/datetime -> to_datetime)
            if pd.api.types.is_numeric_dtype(time_obj):
                base = pd.Timestamp("2000-01-01")
                d[time_col] = base + pd.to_timedelta(pd.to_numeric(time_obj, errors="coerce"), unit="s")
            else:
                d[time_col] = pd.to_datetime(time_obj, errors="coerce")

            d = d.dropna(subset=[time_col]).sort_values(time_col)
            d = _maybe_sample(d, SAMPLE_FOR_PLOTS)

            with st.spinner(f"⏳ Plotando série temporal de {numeric_col}..."):
                fig, ax = _new_fig((12, 6))
                ax.plot(d[time_col], d[numeric_col], alpha=0.8)
                ax.set_title(f"Tendência Temporal: {numeric_col}", color="white", fontsize=14)
                ax.set_xlabel("Tempo", color="white")
                ax.set_ylabel(numeric_col, color="white")
                _setup_dark_axes(ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    else:
        st.info("Não foram identificadas colunas temporais no dataset.")

    # -------- Padrões em Categóricas (com key única) --------
    if categorical_cols:
        st.subheader("Padrões em Variáveis Categóricas")
        cat_col = st.selectbox(
            "Selecione uma variável categórica:",
            categorical_cols,
            key="trend_cat_select"  # chave única (evita conflito com aba Distribuições)
        )
        vc = data[cat_col].value_counts(dropna=True)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Valores Mais Frequentes:**")
            st.dataframe(vc.head(10).reset_index(names=[cat_col, "contagem"]), use_container_width=True)
        with c2:
            st.write("**Valores Menos Frequentes:**")
            st.dataframe(vc.tail(10).reset_index(names=[cat_col, "contagem"]), use_container_width=True)



    # Colunas temporais
    tcols = _time_columns(data)
    if tcols:
        st.subheader("Tendências Temporais")

        time_col = st.selectbox("Selecione a coluna temporal:", tcols)
        numeric_col = st.selectbox("Selecione a variável para análise temporal:", numeric_cols) if numeric_cols else None

        if time_col and numeric_col:
            # Ordena e converte tempo de forma confiável
            d = data[[time_col, numeric_col]].copy()
            d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
            d = d.dropna(subset=[time_col])
            d = d.sort_values(time_col)

            # Amostra para gráfico
            d = _maybe_sample(d, SAMPLE_FOR_PLOTS)

            fig, ax = _new_fig((12, 6))
            ax.plot(d[time_col], d[numeric_col], alpha=0.8)
            ax.set_title(f"Tendência Temporal: {numeric_col}", color="white", fontsize=14)
            ax.set_xlabel("Tempo", color="white")
            ax.set_ylabel(numeric_col, color="white")
            _setup_dark_axes(ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Não foram identificadas colunas temporais no dataset.")

    # Padrões categóricos
    if categorical_cols:
        st.subheader("Padrões em Variáveis Categóricas")
        cat_col = st.selectbox("Selecione uma variável categórica:", categorical_cols)
        if cat_col:
            vc = data[cat_col].value_counts(dropna=True)
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Valores Mais Frequentes:**")
                st.dataframe(vc.head(10).reset_index(names=[cat_col, "contagem"]), use_container_width=True)
            with c2:
                st.write("**Valores Menos Frequentes:**")
                st.dataframe(vc.tail(10).reset_index(names=[cat_col, "contagem"]), use_container_width=True)


def render_tab_outliers(data: pd.DataFrame, numeric_cols: List[str]) -> None:
    st.header("⚠️ Detecção de Anomalias")

    if not numeric_cols:
        st.info("Não há variáveis numéricas para análise de outliers.")
        return

    st.subheader("Outliers por Variável")

    # Amostra para IQR em datasets muito grandes
    d = _maybe_sample(data[numeric_cols], SAMPLE_FOR_STATS)

    summary = []
    for col in numeric_cols:
        s = d[col].dropna()
        if s.empty:
            continue
        Q1, Q3 = s.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (s < lower) | (s > upper)
        count = int(mask.sum())
        pct = (count / len(s)) * 100

        summary.append({
            "Variável": col,
            "Total de Outliers": count,
            "Percentual": f"{pct:.2f}%",
            "Limite Inferior": f"{lower:.2f}",
            "Limite Superior": f"{upper:.2f}",
            "Valor Mínimo": f"{s.min():.2f}",
            "Valor Máximo": f"{s.max():.2f}",
        })

    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    # Boxplots
    st.subheader("Visualização de Outliers (Boxplots)")
    cols_to_plot = numeric_cols[:MAX_BOX_PLOTS]
    cols_per_row = 4
    rows = (len(cols_to_plot) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 4 * rows))
    fig.patch.set_facecolor(DARK_BG)
    if rows == 1:
        axes = axes.reshape(1, -1) if len(cols_to_plot) > 1 else [axes]

    for i, col in enumerate(cols_to_plot):
        r, c = i // cols_per_row, i % cols_per_row
        ax = axes[r][c] if rows > 1 else axes[c]
        bp = ax.boxplot(d[col].dropna(), patch_artist=True)
        bp["boxes"][0].set_alpha(0.7)
        _setup_dark_axes(ax)
        ax.set_title(col, color="white", fontsize=10)

    for i in range(len(cols_to_plot), rows * cols_per_row):
        r, c = i // cols_per_row, i % cols_per_row
        if rows > 1:
            fig.delaxes(axes[r][c])
        else:
            fig.delaxes(axes[c])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_tab_ai(data: pd.DataFrame, overview: Dict[str, object]) -> None:
    st.header("🤖 Consulta Inteligente com IA Groq")
    st.markdown("Faça perguntas sobre seus dados e obtenha insights com modelos avançados.")

    # Chave da API
    c1, c2 = st.columns([2, 1])
    with c1:
        # Prioriza st.secrets se existir (mais seguro)
        default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
        api_key = st.text_input("🔑 Insira sua chave da API da Groq:", type="password", value=default_key,
                                help="Use st.secrets['GROQ_API_KEY'] para evitar digitar toda vez.")
    with c2:
        model_options = {
            "llama-3.3-70b-versatile": "🦙 Llama 3.3 70B (Recomendado)",
            "llama-3.1-8b-instant": "🦙 Llama 3.1 8B (Rápido)",
            "openai/gpt-oss-120b": "🧠 GPT OSS 120B (Poderoso)",
            "openai/gpt-oss-20b": "🧠 GPT OSS 20B (Eficiente)",
        }
        selected_model = st.selectbox(
            "🧠 Escolha o modelo:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0,
        )

    if not api_key:
        st.info("🔑 Insira sua chave da API da Groq para usar a consulta inteligente.")
        st.markdown(
            """
            **Como obter sua chave da API:**
            1. console.groq.com → API Keys → New key
            2. Copie e cole acima

            **Boas práticas:** use `st.secrets` para não digitar a chave sempre e evitar expor em repositórios.
            """
        )
        return

    if Groq is None:
        st.error("Biblioteca `groq` não instalada. Rode `pip install groq`.")
        return

    # Contexto “compacto” (evita enviar dados linha a linha)
    numeric_cols: List[str] = overview["numeric_cols"]  # type: ignore
    context_summary = {
        "num_linhas": int(overview["shape"][0]),      # type: ignore
        "num_colunas": int(overview["shape"][1]),     # type: ignore
        "colunas": list(map(str, data.columns.tolist())),
        "tipos": {c: str(t) for c, t in data.dtypes.items()},
        "nulos": {c: int(data[c].isna().sum()) for c in data.columns},
        "unicos": {c: int(data[c].nunique(dropna=True)) for c in data.columns},
        "describe": overview["desc"].to_dict() if overview["desc"] is not None else "sem_numericas",
    }

    question = st.text_area(
        "💭 Sua pergunta sobre os dados:",
        placeholder="Ex.: Quais variáveis mais se correlacionam? Há indícios de sazonalidade?",
        height=100,
    )

    with st.expander("⚙️ Configurações Avançadas"):
        c1, c2 = st.columns(2)
        with c1:
            max_tokens = st.slider("Máximo de tokens:", 100, 2000, 1000)
            temperature = st.slider("Criatividade (temperature):", 0.0, 1.0, 0.7, 0.1)
        with c2:
            system_prompt = st.text_area(
                "Prompt do sistema (opcional):",
                value="Você é um(a) especialista em análise de dados e ciência de dados.",
                height=100,
            )

    if st.button("🚀 Analisar com IA", type="primary"):
        if not question.strip():
            st.warning("⚠️ Digite uma pergunta antes de analisar.")
            return

        with st.spinner(f"🤖 Analisando com {model_options[selected_model]}..."):
            try:
                client = Groq(api_key=api_key)
                prompt = (
                    "Analise o dataset a partir do resumo e responda as pergunta com clareza,"
                    "Interaja com o usuário de forma objetiva e sucinta,"
                    "Responda de forma independente, e aja com inteligência,"
                    "Fale somente baseado em dados e estatísticas, sem sair do contexto do dataset,"
                    "citando possíveis limitações dos dados quando pertinente.\n\n"
                    f"RESUMO DO DATASET (compacto):\n{context_summary}\n\n"
                    f"PERGUNTA DO USUÁRIO: {question}\n"
                    "Sugira análises complementares se fizer sentido."
                )

                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                st.success("✅ Análise concluída!")
                st.markdown("### 🎯 Resposta da IA:")
                st.markdown(response.choices[0].message.content)

                if hasattr(response, "usage"):
                    with st.expander("📊 Informações de Uso"):
                        st.write(f"**Tokens usados:** {getattr(response.usage, 'total_tokens', 'N/D')}")
                        st.write(f"**Modelo:** {selected_model}")

            except Exception as e:
                st.error(f"❌ Erro ao consultar a API da Groq: {e}")
                st.info("Verifique a chave e a disponibilidade da API.")


# =========================
# Main (layout e fluxo)
# =========================

uploaded = st.file_uploader(
    "Carregue seu arquivo CSV para análise",
    type=["csv"],
    help="Selecione um arquivo CSV para análise exploratória",
)

# CSS condicional
if uploaded is None:
    # Antes do upload → centralizado
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1000px;
            margin: auto;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    # Depois do upload → full width responsivo
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 100% !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .stDataFrame, .stTable {
            max-width: 100% !important;
        }
        div[data-baseweb="tab-list"] {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if uploaded is not None:
    with st.spinner("⏳ Carregando arquivo CSV..."):
        data, err = load_csv(uploaded)

    if err:
        st.error(f"❌ {err}")
        st.info("Verifique se o arquivo está íntegro, separador e encoding corretos.")
    elif data.empty:
        st.warning("Arquivo vazio ou sem colunas legíveis.")
    else:
        st.success(f"✅ Arquivo carregado! {data.shape[0]:,} linhas x {data.shape[1]:,} colunas.")
        overview = get_overview(data)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📋 Visão Geral", "📊 Distribuições", "🔍 Correlações", "📈 Tendências", "⚠️ Anomalias", "🤖 Consulta IA"]
        )

        with tab1:
            render_tab_overview(data, overview)
        with tab2:
            render_tab_distributions(data, overview["numeric_cols"], overview["categorical_cols"])  # type: ignore
        with tab3:
            render_tab_correlations(data, overview["numeric_cols"])  # type: ignore
        with tab4:
            render_tab_trends(data, overview["numeric_cols"], overview["categorical_cols"])  # type: ignore
        with tab5:
            render_tab_outliers(data, overview["numeric_cols"])  # type: ignore
        with tab6:
            render_tab_ai(data, overview)

else:
    st.markdown(
        """
        ## Bem-vindo ao Agente de Análise de Dados com IA!
        Carregue um CSV e explore as abas de análise. Use a IA para perguntas específicas sobre o dataset.
        """
    )


