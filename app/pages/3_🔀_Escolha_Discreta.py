import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

import _bootstrap  # noqa: F401
from components import botoes_download_relatorio, estilizar, kpis
from state import exigir_dataset, sidebar_dados
from theme import aplicar_tema, cabecalho, caixa, rodape

from econometria.reports import ReportBuilder, fig_to_base64
from econometria.regression import logit_regression, probit_regression

st.set_page_config(page_title="Escolha Discreta - Econometria", page_icon="🔀", layout="wide")
aplicar_tema()
sidebar_dados()
cabecalho(
    "🔀 Modelos de Escolha Discreta",
    "Modele uma decisao binaria (comprar/nao comprar, inadimplir/nao inadimplir, converter/nao "
    "converter...) com Logit ou Probit, veja efeitos marginais, razoes de chance e a acuracia do modelo.",
)
df = exigir_dataset()
numericas = df.select_dtypes(include="number").columns.tolist()
candidatas_y = [c for c in numericas if df[c].dropna().nunique() == 2]

if not candidatas_y:
    st.warning(
        "Nenhuma coluna binaria (0/1 ou apenas 2 valores distintos) foi encontrada nos dados. "
        "A variavel dependente de um modelo de escolha discreta precisa ser binaria."
    )
    st.stop()

col_config, col_result = st.columns([1, 2], gap="large")
with col_config:
    st.markdown("#### ⚙️ Configuracao do modelo")
    y = st.selectbox("Variavel dependente binaria (Y)", candidatas_y)
    x = st.multiselect("Variaveis explicativas (X)", [c for c in numericas if c != y], default=[c for c in numericas if c != y][:3])
    tipo = st.radio("Tipo de modelo", ["Logit", "Probit"], horizontal=True)
    rodar = st.button("▶️ Rodar modelo", type="primary", width="stretch")

with col_result:
    if not x:
        st.info("Selecione ao menos uma variavel explicativa.")
    elif rodar:
        try:
            fn = logit_regression if tipo == "Logit" else probit_regression
            resultado = fn(df, y, x)
            st.session_state["disc_resultado"] = resultado
        except Exception as exc:
            st.error(str(exc))
            st.session_state.pop("disc_resultado", None)

resultado = st.session_state.get("disc_resultado")
if resultado is not None and resultado.variavel_dependente == y and set(resultado.variaveis_independentes) == set(x):
    st.divider()
    kpis(
        [
            ("Pseudo-R² (McFadden)", f"{resultado.pseudo_r2:.4f}", ""),
            ("Log-verossimilhanca", f"{resultado.log_likelihood:.1f}", ""),
            ("AIC", f"{resultado.aic:.1f}", ""),
            ("Acuracia (corte 0,5)", f"{resultado.acuracia * 100:.1f}%", ""),
        ]
    )
    caixa(resultado.resumo_texto().replace("\n\n", "<br><br>"), "info")

    aba_coef, aba_class = st.tabs(["📋 Coeficientes & efeitos marginais", "🎯 Classificacao"])
    with aba_coef:
        st.dataframe(
            resultado.tabela_coeficientes.style.format(
                {c: "{:.4f}" for c in resultado.tabela_coeficientes.select_dtypes("number").columns}
            ),
            width="stretch",
        )
        if resultado.efeitos_marginais is not None:
            st.markdown("**Efeitos marginais medios** (impacto de +1 unidade em X sobre a probabilidade de Y=1)")
            st.dataframe(resultado.efeitos_marginais.style.format("{:.4f}"), width="stretch")

    with aba_class:
        fig = px.imshow(
            resultado.matriz_confusao,
            text_auto=True,
            labels=dict(x="Previsto", y="Real", color="contagem"),
            color_continuous_scale="Blues",
        )
        st.plotly_chart(estilizar(fig, "Matriz de confusao"), width="stretch")
        vp = resultado.matriz_confusao.iloc[1, 1] if resultado.matriz_confusao.shape == (2, 2) else None
        if vp is not None:
            vn, fp = resultado.matriz_confusao.iloc[0, 0], resultado.matriz_confusao.iloc[0, 1]
            fn_ = resultado.matriz_confusao.iloc[1, 0]
            precisao = vp / (vp + fp) if (vp + fp) else float("nan")
            revocacao = vp / (vp + fn_) if (vp + fn_) else float("nan")
            kpis([("Precisao", f"{precisao*100:.1f}%", ""), ("Revocacao (recall)", f"{revocacao*100:.1f}%", "")])

    st.divider()
    if st.button("📄 Gerar relatorio do modelo", key="btn_report_disc"):
        fig_mpl, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(resultado.matriz_confusao, cmap="Blues")
        mat = resultado.matriz_confusao.to_numpy()
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de confusao")

        builder = ReportBuilder(f"Relatorio de Modelo {tipo}", f"{y} ~ {' + '.join(x)}")
        builder.add_card("Pseudo-R²", f"{resultado.pseudo_r2:.4f}")
        builder.add_card("Acuracia", f"{resultado.acuracia*100:.1f}%")
        builder.add_card("N observacoes", str(resultado.n_observacoes))
        tabelas = [("Coeficientes", resultado.tabela_coeficientes)]
        if resultado.efeitos_marginais is not None:
            tabelas.append(("Efeitos marginais", resultado.efeitos_marginais))
        builder.add_section(
            "Resultado do modelo",
            resultado.resumo_texto(),
            tabelas=tabelas,
            figuras=[("Matriz de confusao", fig_to_base64(fig_mpl))],
        )
        plt.close(fig_mpl)
        botoes_download_relatorio(builder, f"{tipo.lower()}_escolha_discreta")

rodape()
