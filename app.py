from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Passos Mágicos | Risco de Defasagem",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_BASE = BASE_DIR / "base_datathon_consolidada.csv"
ARQUIVO_MODELO = BASE_DIR / "modelo_risco.joblib"

THRESHOLD_ALERTA = 0.35


# ============================================================
# CARGA DE DADOS E MODELO
# ============================================================

@st.cache_data
def carregar_base():
    return pd.read_csv(ARQUIVO_BASE)


@st.cache_resource
def carregar_modelo():
    return joblib.load(ARQUIVO_MODELO)


base = carregar_base()
modelo = carregar_modelo()


# ============================================================
# VARIÁVEIS DO MODELO
# ============================================================

FEATURES_NUMERICAS = [
    "idade",
    "anos_na_pm",
    "fase",
    "fase_ideal",
    "inde",
    "ian",
    "ida",
    "ieg",
    "iaa",
    "ips",
    "ipv",
    "matematica",
    "portugues",
    "ingles",
    "defasagem",
]

FEATURES_CATEGORICAS = [
    "genero",
    "instituicao_ensino",
    "pedra",
]

FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def classificar_risco(probabilidade):
    if probabilidade < 0.20:
        return "Baixo"
    if probabilidade < THRESHOLD_ALERTA:
        return "Atenção"
    return "Alto"


def preparar_base_2024():
    dados = base[base["ano_referencia"] == 2024].copy()
    dados = dados.dropna(subset=["ra"])

    dados["probabilidade_risco"] = modelo.predict_proba(
        dados[FEATURES]
    )[:, 1]

    dados["classificacao_risco"] = dados[
        "probabilidade_risco"
    ].apply(classificar_risco)

    return dados


base_2024 = preparar_base_2024()


# ============================================================
# CABEÇALHO
# ============================================================

st.title("📚 Passos Mágicos — Monitor de Risco de Defasagem")

st.markdown(
    """
    Aplicação desenvolvida para o **Datathon FIAP — Fase 5**, utilizando os
    dados PEDE de 2022, 2023 e 2024.

    O objetivo é combinar análise histórica e Machine Learning para apoiar
    a identificação antecipada de alunos com maior probabilidade de
    **piora da defasagem no período seguinte**.
    """
)

st.divider()


# ============================================================
# ABAS
# ============================================================

tab_visao, tab_risco, tab_efetividade, tab_modelo = st.tabs(
    [
        "📊 Visão Geral",
        "🎯 Risco por Aluno",
        "💎 Efetividade",
        "🤖 Sobre o Modelo",
    ]
)


# ============================================================
# ABA 1 — VISÃO GERAL
# ============================================================

with tab_visao:

    st.subheader("Visão geral dos dados")

    dados_2024 = base[base["ano_referencia"] == 2024]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Alunos em 2024",
        f"{dados_2024['ra'].nunique():,.0f}".replace(",", "."),
    )

    c2.metric(
        "INDE médio",
        f"{dados_2024['inde'].mean():.2f}",
    )

    c3.metric(
        "Alunos em defasagem",
        f"{100 * dados_2024['em_defasagem'].mean():.1f}%",
    )

    c4.metric(
        "Defasagem média",
        f"{dados_2024['defasagem'].mean():.2f}",
    )

    st.subheader("Evolução dos principais indicadores")

    resumo_anual = (
        base.groupby("ano_referencia")
        .agg(
            INDE=("inde", "mean"),
            IDA=("ida", "mean"),
            IEG=("ieg", "mean"),
            IPV=("ipv", "mean"),
        )
        .round(2)
    )

    st.line_chart(resumo_anual)

    st.subheader("Evolução da defasagem")

    defasagem_anual = (
        base.groupby("ano_referencia")["em_defasagem"]
        .mean()
        .mul(100)
        .rename("% em defasagem")
    )

    st.bar_chart(defasagem_anual)

    st.caption(
        "A proporção de alunos em defasagem cai de aproximadamente "
        "69,9% em 2022 para 46,2% em 2024."
    )

    st.subheader("Distribuição das Pedras em 2024")

    pedras_2024 = dados_2024["pedra"].value_counts()

    st.bar_chart(pedras_2024)


# ============================================================
# ABA 2 — RISCO POR ALUNO
# ============================================================

with tab_risco:

    st.subheader("Score preditivo de risco")

    st.info(
        "O modelo estima a probabilidade de piora da defasagem. "
        "O limiar operacional adotado é 35%, priorizando Recall para "
        "reduzir a chance de deixar um aluno realmente em risco sem alerta."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Alto risco",
        int((base_2024["classificacao_risco"] == "Alto").sum()),
    )

    c2.metric(
        "Atenção",
        int((base_2024["classificacao_risco"] == "Atenção").sum()),
    )

    c3.metric(
        "Baixo risco",
        int((base_2024["classificacao_risco"] == "Baixo").sum()),
    )

    filtro_risco = st.selectbox(
        "Filtrar por faixa de risco",
        ["Todos", "Alto", "Atenção", "Baixo"],
    )

    if filtro_risco == "Todos":
        dados_filtrados = base_2024.copy()
    else:
        dados_filtrados = base_2024[
            base_2024["classificacao_risco"] == filtro_risco
        ].copy()

    colunas_tabela = [
        "ra",
        "nome",
        "fase",
        "pedra",
        "inde",
        "ian",
        "ida",
        "ieg",
        "ipv",
        "defasagem",
        "probabilidade_risco",
        "classificacao_risco",
    ]

    st.dataframe(
        dados_filtrados[colunas_tabela]
        .sort_values("probabilidade_risco", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "probabilidade_risco": st.column_config.ProgressColumn(
                "Probabilidade de risco",
                format="%.1%%",
                min_value=0.0,
                max_value=1.0,
            )
        },
    )

    if not dados_filtrados.empty:

        st.subheader("Detalhamento individual")

        opcoes_ra = dados_filtrados["ra"].astype(str).tolist()

        ra_selecionado = st.selectbox(
            "Selecione um aluno pelo RA",
            opcoes_ra,
        )

        aluno = base_2024[
            base_2024["ra"].astype(str) == str(ra_selecionado)
        ].iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Risco estimado",
            f"{aluno['probabilidade_risco']:.1%}",
        )

        c2.metric(
            "Faixa",
            aluno["classificacao_risco"],
        )

        c3.metric(
            "Pedra",
            str(aluno["pedra"]),
        )

        c4.metric(
            "Defasagem atual",
            f"{aluno['defasagem']:.0f}",
        )

        indicadores = pd.DataFrame(
            {
                "Indicador": [
                    "INDE",
                    "IAN",
                    "IDA",
                    "IEG",
                    "IAA",
                    "IPS",
                    "IPV",
                ],
                "Valor": [
                    aluno["inde"],
                    aluno["ian"],
                    aluno["ida"],
                    aluno["ieg"],
                    aluno["iaa"],
                    aluno["ips"],
                    aluno["ipv"],
                ],
            }
        ).set_index("Indicador")

        st.bar_chart(indicadores)

        historico = (
            base[base["ra"].astype(str) == str(ra_selecionado)]
            .sort_values("ano_referencia")
            .set_index("ano_referencia")
        )

        if len(historico) > 1:
            st.subheader("Trajetória histórica")

            colunas_historico = [
                coluna
                for coluna in ["inde", "ida", "ieg", "ipv"]
                if coluna in historico.columns
            ]

            st.line_chart(
                historico[colunas_historico]
            )


# ============================================================
# ABA 3 — EFETIVIDADE
# ============================================================

with tab_efetividade:

    st.subheader("Efetividade do programa")

    indicador = st.selectbox(
        "Escolha um indicador para comparar as Pedras",
        ["inde", "ida", "ieg", "ipv", "defasagem"],
    )

    pedras_validas = [
        "Quartzo",
        "Ágata",
        "Ametista",
        "Topázio",
    ]

    dados_pedra = base[
        base["pedra"].isin(pedras_validas)
    ]

    evolucao_pedra = (
        dados_pedra.groupby(
            ["ano_referencia", "pedra"]
        )[indicador]
        .mean()
        .unstack()
    )

    st.line_chart(evolucao_pedra)

    st.subheader("Coorte longitudinal")

    anos_por_aluno = (
        base.groupby("ra")["ano_referencia"]
        .nunique()
    )

    alunos_tres_anos = anos_por_aluno[
        anos_por_aluno == 3
    ].index

    coorte = base[
        base["ra"].isin(alunos_tres_anos)
    ]

    resumo_coorte = (
        coorte.groupby("ano_referencia")
        .agg(
            Alunos=("ra", "nunique"),
            Defasagem_Media=("defasagem", "mean"),
            Percentual_Em_Defasagem=("em_defasagem", "mean"),
            INDE=("inde", "mean"),
            IDA=("ida", "mean"),
            IEG=("ieg", "mean"),
        )
    )

    resumo_coorte["Percentual_Em_Defasagem"] *= 100

    st.dataframe(
        resumo_coorte.round(2),
        use_container_width=True,
    )

    st.markdown(
        """
        **Leitura executiva**

        - A análise longitudinal acompanha os mesmos alunos ao longo do tempo.
        - Foram identificados **468 alunos presentes em 2022, 2023 e 2024**.
        - Nessa coorte, a defasagem média melhora de aproximadamente
          **-0,85 para -0,23**.
        - O percentual em defasagem cai de aproximadamente
          **67,3% para 34,8%**.
        - A evolução não é uniforme em todos os indicadores; o IDA merece
          acompanhamento específico.
        """
    )


# ============================================================
# ABA 4 — SOBRE O MODELO
# ============================================================

with tab_modelo:

    st.subheader("Modelo preditivo")

    st.markdown(
        """
        ### O que o modelo prevê?

        O alvo foi definido como:

        **`risco_piora = 1` quando a defasagem do aluno no ano seguinte é
        pior que a observada no ano atual.**

        Exemplos:

        - `0 → -1`: piora;
        - `-1 → -2`: piora;
        - `-2 → -1`: melhora;
        - `-1 → -1`: permanece estável.

        ### Estratégia de validação

        Para reduzir *data leakage* e aproximar a avaliação de um cenário
        real de utilização:

        - **Treino:** indicadores de 2022 → resultado observado em 2023;
        - **Teste:** indicadores de 2023 → resultado observado em 2024.

        O modelo escolhido foi o **Random Forest**.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ROC-AUC",
        "0,855",
    )

    c2.metric(
        "PR-AUC",
        "0,570",
    )

    c3.metric(
        "Recall",
        "77,3%",
    )

    c4.metric(
        "Threshold",
        "35%",
    )

    st.markdown(
        """
        ### Por que priorizar Recall?

        Em um cenário educacional, um falso negativo significa que um aluno
        realmente sujeito à piora pode não receber atenção preventiva.
        Por isso, o threshold de 35% foi escolhido para aumentar a capacidade
        de detecção do modelo.

        Com esse ponto de corte, o modelo alcançou aproximadamente:

        - **Recall:** 77,3%;
        - **Precision:** 44,5%;
        - **F1-Score:** 56,5%.

        ### Como interpretar o resultado?

        A probabilidade não representa uma decisão automática sobre o aluno.
        O objetivo do modelo é funcionar como **apoio à priorização de
        acompanhamento**, combinando o score com a análise da equipe da
        Passos Mágicos.
        """
    )

    st.warning(
        "Este protótipo é uma ferramenta de apoio à decisão. "
        "As probabilidades devem ser interpretadas em conjunto com os "
        "indicadores educacionais e a avaliação da equipe responsável."
    )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()
st.caption(
    "Datathon FIAP — Passos Mágicos | Análise PEDE 2022–2024"
)
