#Datathon FIAP — Passos Mágicos

Integrantes:

Raimar de Assis Modesto

Luis Gustavo Barbosa Ribeiro

Turma: Pós Tech - 11DTAT

Data Analytics — FIAP

Projeto desenvolvido para o **Datathon da Fase 5 da Pós-Tech FIAP**, utilizando
os dados PEDE de 2022, 2023 e 2024 da Passos Mágicos.

## Objetivo

Analisar a evolução dos alunos, identificar fatores relacionados à defasagem e
desenvolver um modelo preditivo capaz de estimar a probabilidade de piora da
defasagem no período seguinte.

## Modelo preditivo

O alvo utilizado foi:

`risco_piora = 1` quando a defasagem do aluno no ano seguinte é pior que a
observada no ano atual.

A validação foi temporal:

- Treino: **2022 → 2023**
- Teste: **2023 → 2024**

Modelo escolhido: **Random Forest**

### Resultados

| Métrica | Resultado |
|---|---:|
| ROC-AUC | 0,855 |
| PR-AUC | 0,570 |
| Recall (threshold 0,35) | 77,3% |
| Precision (threshold 0,35) | 44,5% |
| F1-Score (threshold 0,35) | 56,5% |

## Principais resultados

- Alunos em defasagem: **69,9% em 2022 → 46,2% em 2024**
- INDE médio: **7,04 → 7,40**
- Coorte de **468 alunos** presentes nos três anos
- Nessa coorte, a defasagem média evolui de aproximadamente **-0,85 → -0,23**
- O percentual em defasagem da coorte cai de aproximadamente **67,3% → 34,8%**

## Aplicação Streamlit

A aplicação possui quatro áreas:

1. **Visão Geral** — evolução dos principais indicadores;
2. **Risco por Aluno** — ranking e detalhamento do score preditivo;
3. **Efetividade** — análise por Pedra e coorte longitudinal;
4. **Sobre o Modelo** — metodologia e métricas do Random Forest.

### Link da aplicação

> Após o deploy no Streamlit Community Cloud, coloque aqui o link público.

`https://SEU-APP.streamlit.app`

## Estrutura do repositório

```text
.
├── app.py
├── requirements.txt
├── modelo_risco.joblib
├── base_datathon_consolidada.csv
├── README.md
├── .gitignore
│
├── notebook/
│   └── Datathon_Passos_Magicos_Analise_ML_V2.ipynb
│
└── docs/
    └── Datathon_Passos_Magicos_Storytelling.pdf
```

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Observação

O modelo é uma ferramenta de **apoio à decisão**. A probabilidade de risco não
deve ser utilizada como decisão automática sobre o aluno e deve ser analisada
em conjunto com os indicadores e com a equipe da Passos Mágicos.
