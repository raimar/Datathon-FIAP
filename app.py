import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Passos Mágicos - Risco de Defasagem", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
base = pd.read_csv(BASE_DIR / "base_datathon_consolidada.csv")
with open(BASE_DIR / "modelo_risco.pkl", "rb") as arquivo_modelo:
    modelo = pickle.load(arquivo_modelo)

features_num = ["idade","anos_na_pm","fase","fase_ideal","inde","ian","ida","ieg","iaa","ips","ipv",
                "matematica","portugues","ingles","defasagem"]
features_cat = ["genero","instituicao_ensino","pedra"]
features = features_num + features_cat

st.title("Passos Mágicos - Monitor de Risco de Defasagem")
st.caption("Protótipo preditivo baseado nos dados PEDE 2022-2024")

tab1, tab2, tab3 = st.tabs(["Visão geral", "Risco por aluno", "Efetividade"])

with tab1:
    c1,c2,c3,c4 = st.columns(4)
    b24 = base[base["ano_referencia"]==2024]
    c1.metric("Alunos 2024", f"{b24['ra'].nunique():,}".replace(",","."))
    c2.metric("INDE médio", f"{b24['inde'].mean():.2f}")
    c3.metric("Em defasagem", f"{100*b24['em_defasagem'].mean():.1f}%")
    c4.metric("Defasagem média", f"{b24['defasagem'].mean():.2f}")

    st.subheader("Evolução anual")
    anual = base.groupby("ano_referencia").agg(
        INDE=("inde","mean"),
        IDA=("ida","mean"),
        IEG=("ieg","mean"),
        Defasagem=("defasagem","mean")
    )
    st.line_chart(anual)

    st.subheader("Distribuição por Pedra")
    pedras = b24["pedra"].value_counts()
    st.bar_chart(pedras)

with tab2:
    st.subheader("Score preditivo")
    st.info("O limiar operacional sugerido é 35%, priorizando Recall para reduzir falsos negativos.")

    b24 = base[base["ano_referencia"]==2024].copy()
    candidatos = b24.dropna(subset=["ra"]).copy()
    probs = modelo.predict_proba(candidatos[features])[:,1]
    candidatos["prob_risco"] = probs
    candidatos["faixa_risco"] = pd.cut(
        candidatos["prob_risco"],
        bins=[-0.01,0.20,0.35,1.0],
        labels=["Baixo","Atenção","Alto"]
    )

    filtro = st.selectbox("Faixa de risco", ["Todos","Alto","Atenção","Baixo"])
    view = candidatos if filtro=="Todos" else candidatos[candidatos["faixa_risco"].astype(str)==filtro]

    st.dataframe(
        view[["ra","nome","fase","pedra","inde","ian","ida","ieg","ipv","defasagem","prob_risco","faixa_risco"]]
        .sort_values("prob_risco",ascending=False),
        use_container_width=True
    )

    ra_sel = st.selectbox("Selecionar aluno", view["ra"].astype(str).tolist() if len(view) else [])
    if ra_sel:
        aluno = candidatos[candidatos["ra"].astype(str)==ra_sel].iloc[0]
        st.write(f"**Probabilidade de piora:** {aluno['prob_risco']:.1%}")
        radar = pd.DataFrame({
            "Indicador":["INDE","IAN","IDA","IEG","IAA","IPS","IPV"],
            "Valor":[aluno["inde"],aluno["ian"],aluno["ida"],aluno["ieg"],aluno["iaa"],aluno["ips"],aluno["ipv"]]
        }).set_index("Indicador")
        st.bar_chart(radar)

with tab3:
    st.subheader("Evolução por Pedra")
    pedras = base[base["pedra"].isin(["Quartzo","Ágata","Ametista","Topázio"])]
    indicador = st.selectbox("Indicador", ["inde","ida","ieg","ipv","defasagem"])
    piv = pedras.groupby(["ano_referencia","pedra"])[indicador].mean().unstack()
    st.line_chart(piv)

    st.subheader("Leitura executiva")
    st.markdown("""
    - A proporção de alunos em defasagem caiu de **69,9% em 2022** para **46,2% em 2024**.
    - Na coorte presente nos três anos, a defasagem média melhorou de **-0,85 para -0,23**.
    - **Topázio** concentra os melhores indicadores globais, enquanto **Quartzo** demanda maior atenção.
    - O desempenho acadêmico (IDA) não melhora linearmente, portanto a efetividade deve ser analisada de forma multidimensional.
    """)
