import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

import _bootstrap  # noqa: F401
from components import botoes_download_relatorio, estilizar, kpis, selecionar_colunas_numericas
from state import exigir_dataset, sidebar_dados
from theme import aplicar_tema, cabecalho, caixa, rodape

from econometria.reports import ReportBuilder, fig_to_base64
from econometria.regression import ols_regression, run_diagnostics, wls_regression

st.set_page_config(page_title="Regressao Linear - Econometria", page_icon="📈", layout="wide")
aplicar_tema()
sidebar_dados()
cabecalho(
    "📈 Regressao Linear Multipla",
    "Ajuste um modelo OLS/WLS, veja coeficientes com significancia estatistica e rode a bateria "
    "completa de diagnosticos (multicolinearidade, heterocedasticidade, autocorrelacao, normalidade e forma funcional).",
)
df = exigir_dataset()
numericas = df.select_dtypes(include="number").columns.tolist()

if len(numericas) < 2:
    st.warning("Sao necessarias ao menos 2 colunas numericas (1 dependente + 1 independente).")
    st.stop()

col_config, col_result = st.columns([1, 2], gap="large")

with col_config:
    st.markdown("#### ⚙️ Configuracao do modelo")
    y = st.selectbox("Variavel dependente (Y)", numericas)
    x = st.multiselect("Variaveis independentes (X)", [c for c in numericas if c != y], default=[c for c in numericas if c != y][:2])
    cov_type = st.selectbox(
        "Erros padrao",
        ["nonrobust", "HC0", "HC1", "HC2", "HC3"],
        help="HC0-HC3: erros padrao robustos a heterocedasticidade (White). Use quando o teste de Breusch-Pagan indicar heterocedasticidade.",
    )
    usar_wls = st.checkbox("Usar WLS (minimos quadrados ponderados)", value=False)
    peso = None
    if usar_wls:
        peso = st.selectbox("Coluna de peso (inverso da variancia, se conhecido)", [c for c in numericas if c not in ([y] + x)])
    rodar = st.button("▶️ Rodar regressao", type="primary", width="stretch")

with col_result:
    if not x:
        st.info("Selecione ao menos uma variavel independente.")
    elif rodar:
        try:
            if usar_wls and peso:
                resultado = wls_regression(df, y, x, peso)
            else:
                resultado = ols_regression(df, y, x, cov_type=cov_type)
            st.session_state["reg_resultado"] = resultado
            st.session_state["reg_df"] = df
        except Exception as exc:
            st.error(str(exc))
            st.session_state.pop("reg_resultado", None)

resultado = st.session_state.get("reg_resultado")
if resultado is not None and resultado.variavel_dependente == y and set(resultado.variaveis_independentes) == set(x):
    st.divider()
    kpis(
        [
            ("R²", f"{resultado.r2:.4f}", ""),
            ("R² ajustado", f"{resultado.r2_ajustado:.4f}", ""),
            ("F (p-valor)", f"{resultado.f_p_valor:.4g}", ""),
            ("AIC", f"{resultado.aic:.1f}", ""),
            ("N", f"{resultado.n_observacoes}", ""),
        ]
    )
    caixa(resultado.resumo_texto().replace("\n\n", "<br><br>"), "info")

    aba_coef, aba_resid, aba_diag = st.tabs(["📋 Coeficientes", "📉 Residuos", "🩺 Diagnosticos"])

    with aba_coef:
        st.dataframe(
            resultado.tabela_coeficientes.style.format(
                {"coeficiente": "{:.4f}", "erro_padrao": "{:.4f}", "estatistica_t": "{:.3f}", "p_valor": "{:.4g}", "ic_inferior_95": "{:.4f}", "ic_superior_95": "{:.4f}"}
            ),
            width="stretch",
        )
        st.caption("Significancia: *** p<0.01, ** p<0.05, * p<0.10")

        fig_pred = px.scatter(
            x=resultado.valores_ajustados, y=df.loc[resultado.residuos.index, y],
            labels={"x": "Valores ajustados", "y": f"{y} observado"},
        )
        fig_pred.add_shape(
            type="line",
            x0=resultado.valores_ajustados.min(), x1=resultado.valores_ajustados.max(),
            y0=resultado.valores_ajustados.min(), y1=resultado.valores_ajustados.max(),
            line=dict(dash="dash", color="#d93025"),
        )
        st.plotly_chart(estilizar(fig_pred, "Observado vs. Ajustado"), width="stretch")

    with aba_resid:
        fig_resid = px.scatter(x=resultado.valores_ajustados, y=resultado.residuos, labels={"x": "Valores ajustados", "y": "Residuos"})
        fig_resid.add_hline(y=0, line_dash="dash", line_color="#d93025")
        st.plotly_chart(estilizar(fig_resid, "Residuos vs. Ajustados (checar heterocedasticidade)"), width="stretch")

        fig_hist = px.histogram(resultado.residuos, nbins=30, labels={"value": "Residuo"})
        st.plotly_chart(estilizar(fig_hist, "Distribuicao dos residuos"), width="stretch")

    with aba_diag:
        try:
            diag = run_diagnostics(resultado, df)
            caixa(diag.resumo_texto().replace("\n", "<br>"), "alerta" if diag.alertas else "sucesso")
            if diag.vif is not None:
                st.markdown("**VIF (multicolinearidade)**")
                st.dataframe(diag.vif, width="stretch")
            c1, c2 = st.columns(2)
            c1.metric("Breusch-Pagan (p)", f"{diag.breusch_pagan['p_valor_lm']:.4g}")
            c1.metric("Durbin-Watson", f"{diag.durbin_watson:.3f}")
            c2.metric("Jarque-Bera residuos (p)", f"{diag.jarque_bera_residuos['p_valor']:.4g}")
            if "p_valor" in diag.reset_test:
                c2.metric("RESET Ramsey (p)", f"{diag.reset_test['p_valor']:.4g}")
        except Exception as exc:
            st.warning(f"Nao foi possivel rodar todos os diagnosticos: {exc}")
            diag = None

    st.divider()
    if st.button("📄 Gerar relatorio de regressao", key="btn_report_reg"):
        fig_mpl, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(resultado.valores_ajustados, resultado.residuos, alpha=0.6)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Valores ajustados")
        axes[0].set_ylabel("Residuos")
        axes[0].set_title("Residuos vs. Ajustados")
        axes[1].hist(resultado.residuos, bins=25, color="#2563eb")
        axes[1].set_title("Distribuicao dos residuos")
        fig_mpl.tight_layout()

        builder = ReportBuilder(
            "Relatorio de Regressao Linear",
            f"{y} ~ {' + '.join(x)}",
        )
        builder.add_card("R²", f"{resultado.r2:.4f}")
        builder.add_card("R² ajustado", f"{resultado.r2_ajustado:.4f}")
        builder.add_card("N observacoes", str(resultado.n_observacoes))
        builder.add_section(
            "Resultado do modelo",
            resultado.resumo_texto(),
            tabelas=[("Coeficientes", resultado.tabela_coeficientes)],
            figuras=[("Diagnostico visual de residuos", fig_to_base64(fig_mpl))],
        )
        try:
            diag = run_diagnostics(resultado, df)
            tabelas_diag = [("VIF", diag.vif)] if diag.vif is not None else []
            builder.add_section("Diagnosticos do modelo", diag.resumo_texto(), tabelas=tabelas_diag)
        except Exception:
            pass
        plt.close(fig_mpl)
        botoes_download_relatorio(builder, "regressao_linear")

rodape()
