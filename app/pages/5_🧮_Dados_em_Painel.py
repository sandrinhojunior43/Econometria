import streamlit as st

import _bootstrap  # noqa: F401
from components import botoes_download_relatorio, kpis
from state import exigir_dataset, sidebar_dados
from theme import aplicar_tema, cabecalho, caixa, rodape

from econometria.panel import fixed_effects, hausman_test, pooled_ols, random_effects
from econometria.reports import ReportBuilder

st.set_page_config(page_title="Dados em Painel - Econometria", page_icon="🧮", layout="wide")
aplicar_tema()
sidebar_dados()
cabecalho(
    "🧮 Dados em Painel",
    "Combine a dimensao de corte transversal (empresas, pessoas, paises...) com a dimensao temporal. "
    "Compare Pooled OLS, Efeitos Fixos e Efeitos Aleatorios, e deixe o teste de Hausman indicar o modelo mais adequado.",
)
df = exigir_dataset()
numericas = df.select_dtypes(include="number").columns.tolist()
todas_colunas = df.columns.tolist()

st.markdown("#### ⚙️ Estrutura do painel")
c1, c2 = st.columns(2)
entidade = c1.selectbox("Coluna de identificacao da entidade (ex.: empresa, pais, individuo)", todas_colunas)
tempo = c2.selectbox("Coluna de tempo (ex.: ano, data)", todas_colunas)

y = st.selectbox("Variavel dependente (Y)", [c for c in numericas if c not in (entidade, tempo)])
x = st.multiselect("Variaveis independentes (X)", [c for c in numericas if c not in (entidade, tempo, y)], default=[c for c in numericas if c not in (entidade, tempo, y)][:2])

if not x:
    st.info("Selecione ao menos uma variavel independente.")
    st.stop()

if st.button("▶️ Estimar os tres modelos", type="primary"):
    try:
        pooled = pooled_ols(df, entidade, tempo, y, x)
        fe = fixed_effects(df, entidade, tempo, y, x)
        re_ = random_effects(df, entidade, tempo, y, x)
        haus = hausman_test(fe, re_)
        st.session_state["painel_resultados"] = (pooled, fe, re_, haus)
    except Exception as exc:
        st.error(str(exc))
        st.session_state.pop("painel_resultados", None)

resultados = st.session_state.get("painel_resultados")
if resultados is not None:
    pooled, fe, re_, haus = resultados
    st.divider()
    kpis(
        [
            ("R² - Pooled", f"{pooled.r2:.4f}", ""),
            ("R² within - Efeitos Fixos", f"{fe.r2_within:.4f}" if fe.r2_within is not None else "-", ""),
            ("R² - Efeitos Aleatorios", f"{re_.r2:.4f}", ""),
        ]
    )

    st.markdown("#### 🧪 Teste de Hausman (Efeitos Fixos vs. Aleatorios)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estatistica qui²", f"{haus['estatistica_qui2']:.4f}" if haus["estatistica_qui2"] == haus["estatistica_qui2"] else "-")
    c2.metric("Graus de liberdade", haus["graus_liberdade"])
    c3.metric("P-valor", f"{haus['p_valor']:.4g}" if haus["p_valor"] == haus["p_valor"] else "-")
    caixa(haus["conclusao"], "sucesso")

    aba_pooled, aba_fe, aba_re = st.tabs(["Pooled OLS", "Efeitos Fixos", "Efeitos Aleatorios"])
    for aba, resultado in [(aba_pooled, pooled), (aba_fe, fe), (aba_re, re_)]:
        with aba:
            st.dataframe(resultado.tabela_coeficientes.style.format({c: "{:.4f}" for c in resultado.tabela_coeficientes.select_dtypes("number").columns}), width="stretch")
            st.caption(resultado.resumo_texto())

    st.divider()
    if st.button("📄 Gerar relatorio comparativo de painel"):
        builder = ReportBuilder("Relatorio de Dados em Painel", f"{y} ~ {' + '.join(x)} (entidade: {entidade}, tempo: {tempo})")
        builder.add_card("R² Pooled", f"{pooled.r2:.4f}")
        builder.add_card("R² within (FE)", f"{fe.r2_within:.4f}" if fe.r2_within is not None else "-")
        builder.add_card("R² Efeitos Aleatorios", f"{re_.r2:.4f}")
        builder.add_section(
            "Pooled OLS", pooled.resumo_texto(), tabelas=[("Coeficientes - Pooled", pooled.tabela_coeficientes)]
        )
        builder.add_section(
            "Efeitos Fixos", fe.resumo_texto(), tabelas=[("Coeficientes - Efeitos Fixos", fe.tabela_coeficientes)]
        )
        builder.add_section(
            "Efeitos Aleatorios", re_.resumo_texto(), tabelas=[("Coeficientes - Efeitos Aleatorios", re_.tabela_coeficientes)]
        )
        builder.add_section("Teste de Hausman", haus["conclusao"])
        botoes_download_relatorio(builder, "dados_em_painel")

rodape()
