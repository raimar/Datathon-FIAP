from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Passos Mágicos | Monitor de Risco",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_BASE = BASE_DIR / "base_datathon_consolidada.csv"
ARQUIVO_MODELO = BASE_DIR / "modelo_risco.joblib"

THRESHOLD_ALERTA = 0.35


# ============================================================
# CARGA DOS ARQUIVOS
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
# FEATURES DO MODELO
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
# FUNÇÕES
# ============================================================

def classificar_risco(probabilidade):

    if probabilidade < 0.20:
        return "Baixo"

    elif probabilidade < THRESHOLD_ALERTA:
        return "Atenção"

    return "Alto"


def calcular_risco(dados):

    probabilidade = modelo.predict_proba(
        dados[FEATURES]
    )[:, 1][0]

    classificacao = classificar_risco(probabilidade)

    return probabilidade, classificacao


def preparar_base_2024():

    dados = base[
        base["ano_referencia"] == 2024
    ].copy()

    dados = dados.dropna(
        subset=["ra"]
    )

    dados["probabilidade_risco"] = modelo.predict_proba(
        dados[FEATURES]
    )[:, 1]

    dados["classificacao_risco"] = (
        dados["probabilidade_risco"]
        .apply(classificar_risco)
    )

    return dados


base_2024 = preparar_base_2024()


# ============================================================
# CABEÇALHO
# ============================================================

st.title("📚 Passos Mágicos — Monitor de Risco de Defasagem")

st.markdown(
    """
    Aplicação desenvolvida para o **Datathon FIAP — Fase 5**.

    O objetivo é utilizar os indicadores educacionais dos alunos
    para apoiar a identificação antecipada de estudantes com maior
    probabilidade de **piora da defasagem no período seguinte**.
    """
)

st.divider()


# ============================================================
# ABAS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Visão Geral",
        "🎯 Previsão de Risco",
        "👥 Risco por Aluno",
        "💎 Efetividade",
        "🤖 Sobre o Modelo",
    ]
)


# ============================================================
# ABA 1 — VISÃO GERAL
# ============================================================

with tab1:

    st.header("Visão Geral")

    dados_2024 = base[
        base["ano_referencia"] == 2024
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Alunos em 2024",
        f"{dados_2024['ra'].nunique():,.0f}".replace(",", ".")
    )

    c2.metric(
        "INDE médio",
        f"{dados_2024['inde'].mean():.2f}"
    )

    c3.metric(
        "Em defasagem",
        f"{dados_2024['em_defasagem'].mean():.1%}"
    )

    c4.metric(
        "Defasagem média",
        f"{dados_2024['defasagem'].mean():.2f}"
    )

    st.divider()

    st.subheader("Evolução dos principais indicadores")

    resumo = (
        base
        .groupby("ano_referencia")
        .agg(
            INDE=("inde", "mean"),
            IDA=("ida", "mean"),
            IEG=("ieg", "mean"),
            IPV=("ipv", "mean"),
        )
        .round(2)
    )

    st.line_chart(resumo)

    st.subheader("Evolução da defasagem")

    defasagem = (
        base
        .groupby("ano_referencia")["em_defasagem"]
        .mean()
        .mul(100)
        .rename("% em defasagem")
    )

    st.bar_chart(defasagem)

    st.subheader("Distribuição das Pedras — 2024")

    pedras = (
        dados_2024["pedra"]
        .value_counts()
    )

    st.bar_chart(pedras)

    st.info(
        "Entre 2022 e 2024, a proporção de alunos em defasagem "
        "apresenta queda, indicando evolução positiva do programa."
    )


# ============================================================
# ABA 2 — PREVISÃO DE RISCO
# ============================================================

with tab2:

    st.header("🎯 Previsão de Risco Individual")

    st.markdown(
        """
        Informe os indicadores atuais de um aluno para estimar
        a probabilidade de **piora da defasagem no período seguinte**.
        """
    )

    st.warning(
        "A previsão é uma ferramenta de apoio à decisão. "
        "O resultado deve ser analisado junto aos indicadores "
        "educacionais e à equipe da Passos Mágicos."
    )

    st.subheader("Dados do aluno")

    c1, c2, c3 = st.columns(3)

    with c1:

        idade = st.number_input(
            "Idade",
            min_value=5,
            max_value=30,
            value=14
        )

        anos_na_pm = st.number_input(
            "Anos na Passos Mágicos",
            min_value=0,
            max_value=20,
            value=2
        )

        fase = st.number_input(
            "Fase atual",
            min_value=1.0,
            max_value=10.0,
            value=3.0
        )

        fase_ideal = st.number_input(
            "Fase ideal",
            min_value=1.0,
            max_value=10.0,
            value=3.0
        )

        genero = st.selectbox(
            "Gênero",
            sorted(
                base["genero"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with c2:

        inde = st.number_input(
            "INDE",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1
        )

        ian = st.number_input(
            "IAN",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

        ida = st.number_input(
            "IDA",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

        ieg = st.number_input(
            "IEG",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1
        )

        iaa = st.number_input(
            "IAA",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1
        )

    with c3:

        ips = st.number_input(
            "IPS",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1
        )

        ipv = st.number_input(
            "IPV",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1
        )

        matematica = st.number_input(
            "Matemática",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

        portugues = st.number_input(
            "Português",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

        ingles = st.number_input(
            "Inglês",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

    c4, c5, c6 = st.columns(3)

    with c4:

        defasagem = st.number_input(
            "Defasagem atual",
            min_value=-10.0,
            max_value=10.0,
            value=-1.0,
            step=1.0
        )

    with c5:

        pedra = st.selectbox(
            "Pedra",
            sorted(
                base["pedra"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with c6:

        instituicao = st.selectbox(
            "Instituição de ensino",
            sorted(
                base["instituicao_ensino"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    st.divider()

    if st.button(
        "🔮 Calcular risco",
        type="primary",
        use_container_width=True
    ):

        entrada = pd.DataFrame(
            [{
                "idade": idade,
                "anos_na_pm": anos_na_pm,
                "fase": fase,
                "fase_ideal": fase_ideal,
                "inde": inde,
                "ian": ian,
                "ida": ida,
                "ieg": ieg,
                "iaa": iaa,
                "ips": ips,
                "ipv": ipv,
                "matematica": matematica,
                "portugues": portugues,
                "ingles": ingles,
                "defasagem": defasagem,
                "genero": genero,
                "instituicao_ensino": instituicao,
                "pedra": pedra,
            }]
        )

        probabilidade, classificacao = calcular_risco(
            entrada
        )

        st.subheader("Resultado da previsão")

        r1, r2 = st.columns(2)

        r1.metric(
            "Probabilidade estimada",
            f"{probabilidade:.1%}"
        )

        r2.metric(
            "Classificação",
            classificacao
        )

        st.progress(
            float(probabilidade)
        )

        if classificacao == "Alto":

            st.error(
                "🔴 Alto risco: recomenda-se priorizar "
                "o acompanhamento deste aluno."
            )

        elif classificacao == "Atenção":

            st.warning(
                "🟡 Atenção: o aluno apresenta sinal que "
                "merece acompanhamento."
            )

        else:

            st.success(
                "🟢 Baixo risco: não foi identificado "
                "alto risco pelo modelo."
            )

        st.caption(
            "Threshold operacional utilizado: 35%."
        )


# ============================================================
# ABA 3 — RISCO POR ALUNO
# ============================================================

with tab3:

    st.header("👥 Risco por Aluno")

    st.markdown(
        """
        Ranking dos alunos de 2024 segundo a probabilidade
        estimada pelo modelo.
        """
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🔴 Alto risco",
        int(
            (
                base_2024["classificacao_risco"]
                == "Alto"
            ).sum()
        )
    )

    c2.metric(
        "🟡 Atenção",
        int(
            (
                base_2024["classificacao_risco"]
                == "Atenção"
            ).sum()
        )
    )

    c3.metric(
        "🟢 Baixo risco",
        int(
            (
                base_2024["classificacao_risco"]
                == "Baixo"
            ).sum()
        )
    )

    filtro = st.selectbox(
        "Filtrar classificação",
        [
            "Todos",
            "Alto",
            "Atenção",
            "Baixo",
        ]
    )

    if filtro == "Todos":

        dados = base_2024.copy()

    else:

        dados = base_2024[
            base_2024[
                "classificacao_risco"
            ] == filtro
        ].copy()

    colunas = [
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

    tabela = (
        dados[colunas]
        .sort_values(
            "probabilidade_risco",
            ascending=False
        )
    )

    st.dataframe(
        tabela,
        use_container_width=True,
        hide_index=True,
        column_config={
            "probabilidade_risco":
                st.column_config.ProgressColumn(
                    "Probabilidade",
                    format="%.1%%",
                    min_value=0,
                    max_value=1,
                )
        }
    )

    st.divider()

    if not dados.empty:

        st.subheader(
            "🔎 Detalhamento do aluno"
        )

        ra = st.selectbox(
            "Selecione o RA",
            dados["ra"]
            .astype(str)
            .tolist()
        )

        aluno = base_2024[
            base_2024["ra"]
            .astype(str) == str(ra)
        ].iloc[0]

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Risco",
            f"{aluno['probabilidade_risco']:.1%}"
        )

        a2.metric(
            "Classificação",
            aluno["classificacao_risco"]
        )

        a3.metric(
            "Pedra",
            str(aluno["pedra"])
        )

        a4.metric(
            "Defasagem",
            f"{aluno['defasagem']:.0f}"
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

        st.subheader(
            "Indicadores atuais"
        )

        st.bar_chart(
            indicadores
        )

        historico = (
            base[
                base["ra"]
                .astype(str) == str(ra)
            ]
            .sort_values(
                "ano_referencia"
            )
            .set_index(
                "ano_referencia"
            )
        )

        if len(historico) > 1:

            st.subheader(
                "Evolução histórica"
            )

            historico_indicadores = [
                "inde",
                "ida",
                "ieg",
                "ipv",
            ]

            st.line_chart(
                historico[
                    historico_indicadores
                ]
            )


# ============================================================
# ABA 4 — EFETIVIDADE
# ============================================================

with tab4:

    st.header("💎 Efetividade do Programa")

    indicador = st.selectbox(
        "Indicador",
        [
            "inde",
            "ian",
            "ida",
            "ieg",
            "iaa",
            "ips",
            "ipv",
            "defasagem",
        ]
    )

    pedras = [
        "Quartzo",
        "Ágata",
        "Ametista",
        "Topázio",
    ]

    dados_pedra = base[
        base["pedra"].isin(pedras)
    ]

    evolucao = (
        dados_pedra
        .groupby(
            [
                "ano_referencia",
                "pedra"
            ]
        )[indicador]
        .mean()
        .unstack()
    )

    st.subheader(
        f"Evolução de {indicador.upper()} por Pedra"
    )

    st.line_chart(
        evolucao
    )

    st.divider()

    st.subheader(
        "Coorte longitudinal"
    )

    anos_por_aluno = (
        base
        .groupby("ra")[
            "ano_referencia"
        ]
        .nunique()
    )

    alunos_tres_anos = (
        anos_por_aluno[
            anos_por_aluno == 3
        ]
        .index
    )

    coorte = base[
        base["ra"].isin(
            alunos_tres_anos
        )
    ]

    resumo_coorte = (
        coorte
        .groupby("ano_referencia")
        .agg(
            Alunos=(
                "ra",
                "nunique"
            ),
            Defasagem_Media=(
                "defasagem",
                "mean"
            ),
            Percentual_Defasagem=(
                "em_defasagem",
                "mean"
            ),
            INDE=(
                "inde",
                "mean"
            ),
            IDA=(
                "ida",
                "mean"
            ),
            IEG=(
                "ieg",
                "mean"
            ),
        )
    )

    resumo_coorte[
        "Percentual_Defasagem"
    ] *= 100

    st.dataframe(
        resumo_coorte.round(2),
        use_container_width=True
    )

    st.info(
        "A análise longitudinal acompanha os mesmos alunos "
        "ao longo dos três anos, permitindo observar a evolução "
        "real da trajetória educacional."
    )


# ============================================================
# ABA 5 — MODELO
# ============================================================

with tab5:

    st.header("🤖 Sobre o Modelo Preditivo")

    st.subheader(
        "O que o modelo prevê?"
    )

    st.markdown(
        """
        O modelo estima a probabilidade de **piora da defasagem
        do aluno no período seguinte**.

        Exemplos:

        - `0 → -1` = piora
        - `-1 → -2` = piora
        - `-2 → -1` = melhora
        - `-1 → -1` = estabilidade
        """
    )

    st.subheader(
        "Modelo utilizado"
    )

    st.write(
        "**Random Forest Classifier**"
    )

    st.subheader(
        "Validação temporal"
    )

    st.markdown(
        """
        **Treinamento**

        2022 → 2023

        **Teste**

        2023 → 2024
        """
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ROC-AUC",
        "0,855"
    )

    c2.metric(
        "PR-AUC",
        "0,570"
    )

    c3.metric(
        "Recall",
        "77,3%"
    )

    c4.metric(
        "Threshold",
        "35%"
    )

    st.subheader(
        "Desempenho no threshold de 35%"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Precision",
        "44,5%"
    )

    c2.metric(
        "F1-Score",
        "56,5%"
    )

    st.info(
        """
        O threshold de 35% prioriza o Recall, buscando reduzir
        a quantidade de alunos em risco que poderiam deixar de
        receber atenção preventiva.
        """
    )

    st.warning(
        """
        A probabilidade gerada pelo modelo não representa uma
        decisão automática sobre o aluno. O resultado deve ser
        utilizado como apoio à priorização de acompanhamento,
        junto aos indicadores educacionais e à avaliação da
        equipe da Passos Mágicos.
        """
    )


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Datathon FIAP — Passos Mágicos | PEDE 2022–2024"
)
