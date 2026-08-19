import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

import _bootstrap  # noqa: F401
from components import botoes_download_relatorio, estilizar, kpis, selecionar_colunas_numericas
from state import exigir_dataset, mostrar_perfil_dataset, sidebar_dados
from theme import aplicar_tema, cabecalho, caixa, rodape

from econometria.reports import ReportBuilder, fig_to_base64
from econometria.stats import (
    anova_one_way,
    chi_square_independence,
    correlation_matrix,
    describe,
    normality_test,
    t_test_one_sample,
    t_test_paired,
    t_test_two_sample,
)
from econometria.utils import fmt_pvalue

st.set_page_config(page_title="Estatistica Descritiva - Econometria", page_icon="📊", layout="wide")
aplicar_tema()
sidebar_dados()
cabecalho(
    "📊 Estatistica Descritiva & Testes de Hipotese",
    "Resuma qualquer variavel numerica, avalie correlacoes e normalidade, e rode os testes de "
    "hipotese classicos (t, ANOVA, qui-quadrado) com interpretacao automatica.",
)
df = exigir_dataset()
mostrar_perfil_dataset(df)

aba_desc, aba_corr, aba_testes = st.tabs(["📐 Resumo descritivo", "🔗 Correlacao & normalidade", "🧪 Testes de hipotese"])

# ---------------------------------------------------------------- resumo descritivo
with aba_desc:
    colunas = selecionar_colunas_numericas(
        df, "Colunas numericas a analisar", padrao=df.select_dtypes(include="number").columns.tolist()[:5]
    )
    if not colunas:
        st.info("Selecione ao menos uma coluna numerica.")
    else:
        resultado = describe(df, colunas)
        st.dataframe(resultado.tabela.style.format("{:.4f}"), width="stretch")

        for texto in resultado.interpretacao():
            st.markdown(f"- {texto}")

        coluna_grafico = st.selectbox("Ver distribuicao de:", colunas, key="hist_col")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=coluna_grafico, nbins=30, marginal="box", title=f"Distribuicao de {coluna_grafico}")
            st.plotly_chart(estilizar(fig), width="stretch")
        with col2:
            fig2 = px.box(df, y=coluna_grafico, title=f"Boxplot de {coluna_grafico}", points="outliers")
            st.plotly_chart(estilizar(fig2), width="stretch")

        if st.button("📄 Gerar relatorio deste resumo descritivo", key="btn_report_desc"):
            fig_mpl, ax = plt.subplots(figsize=(7, 4))
            df[colunas].hist(ax=ax if len(colunas) == 1 else None, bins=25, figsize=(9, 5))
            fig_hist = plt.gcf()
            builder = ReportBuilder(
                "Relatorio de Estatistica Descritiva",
                f"Analise de {len(colunas)} variavel(is) numerica(s) sobre {resultado.n_observacoes} observacoes.",
            )
            builder.add_card("Observacoes", str(resultado.n_observacoes))
            builder.add_card("Variaveis", str(len(colunas)))
            builder.add_section(
                "Estatisticas descritivas",
                "\n\n".join(resultado.interpretacao()),
                tabelas=[("Tabela de estatisticas", resultado.tabela)],
                figuras=[("Histogramas", fig_to_base64(fig_hist))],
            )
            plt.close(fig_hist)
            botoes_download_relatorio(builder, "estatistica_descritiva")

# ---------------------------------------------------------------- correlacao e normalidade
with aba_corr:
    colunas_corr = selecionar_colunas_numericas(
        df, "Colunas para matriz de correlacao", padrao=df.select_dtypes(include="number").columns.tolist(), key="corr_cols"
    )
    metodo = st.radio("Metodo", ["pearson", "spearman", "kendall"], horizontal=True)
    if len(colunas_corr) >= 2:
        matriz = correlation_matrix(df, colunas_corr, metodo=metodo)
        fig = px.imshow(matriz, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(estilizar(fig, f"Matriz de correlacao ({metodo})"), width="stretch")
        st.caption("Correlacoes com |r| > 0.7 podem indicar redundancia entre variaveis (cuidado com multicolinearidade em regressoes).")
    else:
        st.info("Selecione ao menos 2 colunas.")

    st.markdown("#### Teste de normalidade (Jarque-Bera / Shapiro-Wilk)")
    coluna_normal = st.selectbox("Coluna", df.select_dtypes(include="number").columns.tolist(), key="col_normal")
    if st.button("Testar normalidade"):
        try:
            resultado_norm = normality_test(df[coluna_normal])
            c1, c2 = st.columns(2)
            c1.metric("Jarque-Bera (p-valor)", fmt_pvalue(resultado_norm["jarque_bera_p_valor"]))
            if "shapiro_p_valor" in resultado_norm:
                c2.metric("Shapiro-Wilk (p-valor)", fmt_pvalue(resultado_norm["shapiro_p_valor"]))
            normal = resultado_norm.get("normal_shapiro_5%", resultado_norm["normal_jarque_bera_5%"])
            caixa(
                f"A 5% de significancia, os dados {'<b>parecem</b>' if normal else '<b>nao parecem</b>'} seguir uma distribuicao normal.",
                "sucesso" if normal else "alerta",
            )
        except Exception as exc:
            st.error(str(exc))

# ---------------------------------------------------------------- testes de hipotese
with aba_testes:
    tipo_teste = st.selectbox(
        "Tipo de teste",
        [
            "Teste t - uma amostra (media = valor)",
            "Teste t - duas amostras independentes",
            "Teste t - amostras pareadas",
            "ANOVA one-way (3+ grupos)",
            "Qui-quadrado de independencia (categoricas)",
        ],
    )
    numericas = df.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in df.columns if c not in numericas]
    alfa = st.slider("Nivel de significancia (alfa)", 0.01, 0.10, 0.05, 0.01)

    resultado_teste = None
    if tipo_teste.startswith("Teste t - uma"):
        col = st.selectbox("Variavel", numericas)
        mu0 = st.number_input("Valor de referencia (H0: media = ...)", value=float(df[col].mean()) if col else 0.0)
        if st.button("Rodar teste"):
            resultado_teste = t_test_one_sample(df[col], mu0, alfa)
    elif tipo_teste.startswith("Teste t - duas"):
        col_valor = st.selectbox("Variavel numerica", numericas)
        col_grupo = st.selectbox("Variavel de grupo (2 categorias)", categoricas or numericas)
        if st.button("Rodar teste"):
            grupos = df[col_grupo].dropna().unique()
            if len(grupos) != 2:
                st.error(f"A variavel de grupo precisa ter exatamente 2 categorias (encontradas: {len(grupos)}).")
            else:
                a = df.loc[df[col_grupo] == grupos[0], col_valor]
                b = df.loc[df[col_grupo] == grupos[1], col_valor]
                resultado_teste = t_test_two_sample(a, b, alfa=alfa)
    elif tipo_teste.startswith("Teste t - amostras pareadas"):
        col_a = st.selectbox("Variavel A", numericas, key="pareado_a")
        col_b = st.selectbox("Variavel B", numericas, key="pareado_b")
        if st.button("Rodar teste"):
            resultado_teste = t_test_paired(df[col_a], df[col_b], alfa)
    elif tipo_teste.startswith("ANOVA"):
        col_valor = st.selectbox("Variavel numerica", numericas, key="anova_valor")
        col_grupo = st.selectbox("Variavel de grupo", categoricas or numericas, key="anova_grupo")
        if st.button("Rodar teste"):
            resultado_teste = anova_one_way(df, col_valor, col_grupo, alfa)
    else:
        col_a = st.selectbox("Variavel categorica A", categoricas or df.columns.tolist(), key="qui_a")
        col_b = st.selectbox("Variavel categorica B", categoricas or df.columns.tolist(), key="qui_b")
        if st.button("Rodar teste"):
            resultado_teste = chi_square_independence(df, col_a, col_b, alfa)

    if resultado_teste is not None:
        kpis(
            [
                ("Estatistica", f"{resultado_teste.estatistica:.4f}", ""),
                ("P-valor", fmt_pvalue(resultado_teste.p_valor), ""),
                ("Decisao", "Rejeita H0" if resultado_teste.rejeita_h0 else "Nao rejeita H0", ""),
            ]
        )
        caixa(resultado_teste.conclusao(), "sucesso" if resultado_teste.rejeita_h0 else "info")

rodape()
