import streamlit as st

import _bootstrap  # noqa: F401
from state import EXEMPLOS, get_dataset, mostrar_perfil_dataset, sidebar_dados
from theme import aplicar_tema, cabecalho, rodape

st.set_page_config(page_title="Econometria", page_icon="📐", layout="wide")
aplicar_tema()
sidebar_dados()

cabecalho(
    "Sistema Econometria",
    "Uma central de calculo econometrico, estatistico e financeiro para decisoes pessoais e "
    "empresariais: carregue seus dados, escolha a analise e gere relatorios automaticos com "
    "interpretacao pronta para uso.",
)

df = get_dataset()
if df is None:
    st.markdown(
        "### 👋 Comece por aqui\n"
        "Use a barra lateral para **enviar seus proprios dados** (CSV, Excel, JSON ou Parquet) "
        "ou carregar um dos **datasets de exemplo** e explorar o sistema imediatamente."
    )
else:
    st.markdown("### ✅ Dataset ativo")
    mostrar_perfil_dataset(df)

st.markdown("### 🧭 Frentes disponiveis")
st.caption("Cada card abre uma pagina dedicada, com formularios, graficos, diagnosticos e exportacao de relatorio.")

cartoes = [
    ("📊", "Estatistica Descritiva", "Medidas de posicao/dispersao, correlacoes, normalidade e testes de hipotese (t, ANOVA, qui-quadrado).", "pages/1_📊_Estatistica_Descritiva.py"),
    ("📈", "Regressao Linear", "OLS/WLS multipla, erros robustos e bateria completa de diagnosticos (VIF, heterocedasticidade, autocorrelacao, normalidade, RESET).", "pages/2_📈_Regressao_Linear.py"),
    ("🔀", "Escolha Discreta", "Logit e Probit para variaveis binarias, efeitos marginais, razao de chances e classificacao.", "pages/3_🔀_Escolha_Discreta.py"),
    ("⏳", "Series Temporais", "Estacionariedade (ADF/KPSS), decomposicao STL, ARIMA/SARIMA automatico e volatilidade GARCH.", "pages/4_⏳_Series_Temporais.py"),
    ("🧮", "Dados em Painel", "Pooled OLS, Efeitos Fixos e Aleatorios com teste de Hausman para escolher o modelo certo.", "pages/5_🧮_Dados_em_Painel.py"),
    ("🔮", "Previsao", "Compare metodos (ingenuo, media movel, Holt-Winters, ARIMA) por validacao fora da amostra e projete o futuro.", "pages/6_🔮_Previsao.py"),
    ("💰", "Financas Pessoais", "Juros compostos, financiamentos (Price/SAC), aposentadoria e analise de orcamento.", "pages/7_💰_Financas_Pessoais.py"),
    ("🏢", "Financas Empresariais", "VPL, TIR, payback, ponto de equilibrio, indicadores financeiros e veredito de viabilidade.", "pages/8_🏢_Financas_Empresariais.py"),
]

for linha in range(0, len(cartoes), 4):
    cols = st.columns(4)
    for col, (icone, titulo, desc, pagina) in zip(cols, cartoes[linha : linha + 4]):
        with col:
            st.page_link(pagina, label=f"**{titulo}**", icon=icone, width="stretch")
            st.caption(desc)

st.divider()
st.markdown("### 💡 Datasets de exemplo inclusos")
col1, col2 = st.columns(2)
itens = list(EXEMPLOS.items())
for i, (nome, arquivo) in enumerate(itens):
    (col1 if i % 2 == 0 else col2).markdown(f"- **{nome}** (`{arquivo}`)")

rodape()
