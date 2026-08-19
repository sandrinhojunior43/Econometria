import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import _bootstrap  # noqa: F401
from components import botoes_download_relatorio, estilizar, kpis
from state import exigir_dataset, sidebar_dados
from theme import aplicar_tema, cabecalho, caixa, rodape

from econometria.reports import ReportBuilder, fig_to_base64
from econometria.timeseries import auto_arima, decompose_series, fit_garch, forecast_arima, test_stationarity

st.set_page_config(page_title="Series Temporais - Econometria", page_icon="⏳", layout="wide")
aplicar_tema()
sidebar_dados()
cabecalho(
    "⏳ Series Temporais",
    "Teste estacionariedade, decomponha tendencia/sazonalidade, ajuste um ARIMA automatico com "
    "previsao e modele volatilidade (GARCH) para series financeiras.",
)
df = exigir_dataset()

colunas_data = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
numericas = df.select_dtypes(include="number").columns.tolist()

col1, col2 = st.columns(2)
with col1:
    coluna_tempo = st.selectbox("Coluna de tempo (opcional - deixe em branco para usar a ordem das linhas)", ["(ordem das linhas)"] + colunas_data)
with col2:
    coluna_valor = st.selectbox("Coluna de valor (serie a analisar)", numericas)

if not coluna_valor:
    st.stop()

dados_serie = df[[coluna_valor] + ([coluna_tempo] if coluna_tempo != "(ordem das linhas)" else [])].dropna()
if coluna_tempo != "(ordem das linhas)":
    dados_serie = dados_serie.sort_values(coluna_tempo)
    serie = pd.Series(dados_serie[coluna_valor].to_numpy(), index=pd.DatetimeIndex(dados_serie[coluna_tempo]))
    freq_inferida = pd.infer_freq(serie.index)
    if freq_inferida:
        serie = serie.asfreq(freq_inferida)
        serie = serie.interpolate()
else:
    serie = pd.Series(dados_serie[coluna_valor].to_numpy())

fig_serie = px.line(x=serie.index, y=serie.values, labels={"x": "tempo", "y": coluna_valor})
st.plotly_chart(estilizar(fig_serie, f"Serie: {coluna_valor}"), width="stretch")

aba_estac, aba_decomp, aba_arima, aba_garch = st.tabs(
    ["📐 Estacionariedade", "🧩 Decomposicao", "🔮 ARIMA & Previsao", "📊 Volatilidade (GARCH)"]
)

with aba_estac:
    if st.button("Rodar testes de estacionariedade (ADF + KPSS)"):
        try:
            resultado = test_stationarity(serie)
            kpis(
                [
                    ("ADF - p-valor", f"{resultado.adf_p_valor:.4g}", "H0: raiz unitaria (nao estacionaria)"),
                    ("KPSS - p-valor", f"{resultado.kpss_p_valor:.4g}", "H0: estacionaria"),
                ]
            )
            caixa(resultado.resumo_texto().replace("\n\n", "<br><br>"), "sucesso" if resultado.estacionaria else "alerta")
        except Exception as exc:
            st.error(str(exc))

with aba_decomp:
    periodo = st.number_input("Periodo sazonal (ex.: 12 = mensal com sazonalidade anual, 7 = diaria semanal)", min_value=2, value=12)
    metodo_decomp = st.radio("Metodo", ["stl", "classica"], horizontal=True)
    if st.button("Decompor serie"):
        try:
            dec = decompose_series(serie, periodo=int(periodo), metodo=metodo_decomp)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=dec.observado, name="Observado"))
            fig.add_trace(go.Scatter(y=dec.tendencia, name="Tendencia"))
            st.plotly_chart(estilizar(fig, "Observado vs. Tendencia"), width="stretch")
            fig2 = px.line(y=dec.sazonalidade, labels={"y": "sazonalidade"}, title=None)
            st.plotly_chart(estilizar(fig2, "Componente sazonal"), width="stretch")
            fig3 = px.scatter(y=dec.residuo, labels={"y": "residuo"})
            st.plotly_chart(estilizar(fig3, "Residuo"), width="stretch")
            caixa(dec.resumo_texto().replace("\n\n", "<br><br>"), "info")
            st.session_state["ts_decomp"] = dec
        except Exception as exc:
            st.error(str(exc))

with aba_arima:
    c1, c2, c3 = st.columns(3)
    p_max = c1.number_input("p maximo", 0, 6, 3)
    d_max = c2.number_input("d maximo", 0, 2, 2)
    q_max = c3.number_input("q maximo", 0, 6, 3)
    passos = st.slider("Passos a prever", 1, 36, 12)
    if st.button("Ajustar ARIMA automatico e prever"):
        try:
            with st.spinner("Buscando a melhor ordem (p,d,q) por AIC..."):
                arima_res = auto_arima(serie, p_max=int(p_max), d_max=int(d_max), q_max=int(q_max))
                previsao = forecast_arima(arima_res, passos)
            caixa(arima_res.resumo_texto(), "info")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=serie.values, x=list(range(len(serie))), name="Historico"))
            offset = len(serie)
            idx_fut = list(range(offset, offset + passos))
            fig.add_trace(go.Scatter(x=idx_fut, y=previsao["previsao"], name="Previsao", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=idx_fut, y=previsao["ic_superior"], name="IC superior", line=dict(width=0), showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=idx_fut, y=previsao["ic_inferior"], name="Intervalo de confianca",
                    fill="tonexty", line=dict(width=0), fillcolor="rgba(37,99,235,0.15)",
                )
            )
            st.plotly_chart(estilizar(fig, "Previsao ARIMA com intervalo de confianca"), width="stretch")
            st.dataframe(previsao.style.format("{:.4f}"), width="stretch")
            st.session_state["ts_arima"] = (arima_res, previsao)
        except Exception as exc:
            st.error(str(exc))

with aba_garch:
    st.caption("Use retornos (variacoes percentuais), nao precos em nivel. Se sua coluna e um preco, calcule antes: `retorno = preco.pct_change()`.")
    p = st.number_input("Ordem p (ARCH)", 1, 3, 1)
    q = st.number_input("Ordem q (GARCH)", 1, 3, 1)
    if st.button("Ajustar GARCH"):
        try:
            garch = fit_garch(serie.dropna(), p=int(p), q=int(q))
            caixa(garch.resumo_texto(), "info")
            fig = px.line(y=garch.vol_condicional.values, labels={"y": "volatilidade condicional"})
            st.plotly_chart(estilizar(fig, "Volatilidade condicional (desvio padrao)"), width="stretch")
            prev_vol = garch.prever_volatilidade(10)
            st.dataframe(prev_vol.style.format("{:.6f}"), width="stretch")
            st.session_state["ts_garch"] = garch
        except Exception as exc:
            st.error(str(exc))

st.divider()
if st.button("📄 Gerar relatorio consolidado de series temporais"):
    builder = ReportBuilder("Relatorio de Series Temporais", f"Serie analisada: {coluna_valor}")
    fig_mpl, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(serie.values)
    ax.set_title(f"Serie: {coluna_valor}")
    builder.add_section("Serie observada", "", figuras=[("Serie ao longo do tempo", fig_to_base64(fig_mpl))])
    plt.close(fig_mpl)

    if "ts_decomp" in st.session_state:
        dec = st.session_state["ts_decomp"]
        builder.add_section("Decomposicao", dec.resumo_texto())
    if "ts_arima" in st.session_state:
        arima_res, previsao = st.session_state["ts_arima"]
        builder.add_section("Modelo ARIMA", arima_res.resumo_texto(), tabelas=[("Previsao", previsao)])
    if "ts_garch" in st.session_state:
        garch = st.session_state["ts_garch"]
        builder.add_section("Volatilidade (GARCH)", garch.resumo_texto())
    botoes_download_relatorio(builder, "series_temporais")

rodape()
