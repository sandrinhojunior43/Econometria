import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import _bootstrap  # noqa: F401
from components import botoes_download_relatorio, estilizar
from state import exigir_dataset, sidebar_dados
from theme import aplicar_tema, cabecalho, caixa, rodape

from econometria.forecasting import (
    arima_forecast,
    compare_models,
    exponential_smoothing_forecast,
    moving_average_forecast,
    naive_forecast,
)
from econometria.reports import ReportBuilder

st.set_page_config(page_title="Previsao - Econometria", page_icon="🔮", layout="wide")
aplicar_tema()
sidebar_dados()
cabecalho(
    "🔮 Previsao",
    "Compare metodos de previsao (ingenuo, media movel, Holt-Winters, ARIMA) por validacao fora "
    "da amostra e escolha o melhor antes de projetar o futuro.",
)
df = exigir_dataset()

colunas_data = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
numericas = df.select_dtypes(include="number").columns.tolist()

c1, c2 = st.columns(2)
coluna_tempo = c1.selectbox("Coluna de tempo (opcional)", ["(ordem das linhas)"] + colunas_data)
coluna_valor = c2.selectbox("Coluna a prever", numericas)

dados_serie = df[[coluna_valor] + ([coluna_tempo] if coluna_tempo != "(ordem das linhas)" else [])].dropna()
if coluna_tempo != "(ordem das linhas)":
    dados_serie = dados_serie.sort_values(coluna_tempo)
    serie = pd.Series(dados_serie[coluna_valor].to_numpy(), index=pd.DatetimeIndex(dados_serie[coluna_tempo]))
    freq_inferida = pd.infer_freq(serie.index)
    if freq_inferida:
        serie = serie.asfreq(freq_inferida).interpolate()
else:
    serie = pd.Series(dados_serie[coluna_valor].to_numpy())

passos = st.slider("Passos a prever", 1, 36, 6)

if st.button("🏁 Comparar metodos (backtest) e escolher o melhor", type="primary"):
    try:
        with st.spinner("Rodando validacao fora da amostra para cada metodo..."):
            comparacao = compare_models(serie, passos, n_janelas_backtest=min(8, max(3, len(serie) // 10)))
        st.session_state["fc_comparacao"] = comparacao
    except Exception as exc:
        st.error(str(exc))

comparacao = st.session_state.get("fc_comparacao")
if comparacao is not None:
    st.markdown("#### 🏆 Ranking dos metodos (menor RMSE em backtest = melhor)")
    st.dataframe(comparacao.style.format({"mae": "{:.3f}", "mse": "{:.3f}", "rmse": "{:.3f}", "mape_%": "{:.2f}"}), width="stretch")
    validos = comparacao.dropna(subset=["rmse"])
    if not validos.empty:
        melhor = validos.index[0]
        caixa(f"Melhor metodo no backtest: <b>{melhor}</b> (RMSE = {validos.loc[melhor, 'rmse']:.3f}).", "sucesso")

st.divider()
st.markdown("#### 📈 Gerar previsao final")
metodo_escolhido = st.selectbox("Metodo para a previsao final", ["ingenuo", "media_movel", "holt_winters", "arima"])

if st.button("Gerar previsao"):
    try:
        if metodo_escolhido == "ingenuo":
            resultado = naive_forecast(serie, passos)
        elif metodo_escolhido == "media_movel":
            resultado = moving_average_forecast(serie, passos, janela=min(3, len(serie) - 1) or 1)
        elif metodo_escolhido == "holt_winters":
            resultado = exponential_smoothing_forecast(serie, passos, tendencia="add")
        else:
            resultado = arima_forecast(serie, passos)
        st.session_state["fc_resultado"] = resultado
    except Exception as exc:
        st.error(str(exc))

resultado = st.session_state.get("fc_resultado")
if resultado is not None:
    caixa(resultado.resumo_texto(), "info")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=resultado.serie_historica.values, x=list(range(len(resultado.serie_historica))), name="Historico"))
    offset = len(resultado.serie_historica)
    idx_fut = list(range(offset, offset + passos))
    fig.add_trace(go.Scatter(x=idx_fut, y=resultado.previsao["previsao"], name="Previsao", line=dict(dash="dash")))
    if "ic_superior" in resultado.previsao.columns:
        fig.add_trace(go.Scatter(x=idx_fut, y=resultado.previsao["ic_superior"], showlegend=False, line=dict(width=0)))
        fig.add_trace(
            go.Scatter(
                x=idx_fut, y=resultado.previsao["ic_inferior"], name="Intervalo de confianca",
                fill="tonexty", line=dict(width=0), fillcolor="rgba(37,99,235,0.15)",
            )
        )
    st.plotly_chart(estilizar(fig, f"Previsao - {resultado.metodo}"), width="stretch")
    st.dataframe(resultado.previsao.style.format("{:.4f}"), width="stretch")

    if st.button("📄 Gerar relatorio de previsao"):
        builder = ReportBuilder("Relatorio de Previsao", f"Serie: {coluna_valor} | Metodo: {resultado.metodo}")
        builder.add_card("Metodo", resultado.metodo)
        builder.add_card("Passos previstos", str(passos))
        secoes_tabelas = [("Previsao", resultado.previsao)]
        if comparacao is not None:
            secoes_tabelas.append(("Comparacao de metodos (backtest)", comparacao))
        builder.add_section("Resultado da previsao", resultado.resumo_texto(), tabelas=secoes_tabelas)
        botoes_download_relatorio(builder, "previsao")

rodape()
