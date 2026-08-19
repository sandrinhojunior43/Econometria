"""Interface unificada de previsao: varios metodos, metricas de erro comuns e backtesting.

A ideia deste modulo e permitir comparar, com a mesma "linguagem" de entrada/saida,
metodos simples (ingenuo, media movel, suavizacao exponencial) com o ARIMA do
modulo `econometria.timeseries`, e escolher automaticamente o melhor via
validacao fora da amostra (backtest walk-forward).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ..timeseries.arima import auto_arima, forecast_arima


@dataclass
class ForecastResult:
    metodo: str
    previsao: pd.DataFrame  # colunas: previsao [, ic_inferior, ic_superior]
    serie_historica: pd.Series
    detalhes: dict = field(default_factory=dict)

    def resumo_texto(self) -> str:
        ultimo_real = self.serie_historica.iloc[-1]
        primeiro_previsto = self.previsao["previsao"].iloc[0]
        variacao = (primeiro_previsto / ultimo_real - 1) * 100 if ultimo_real != 0 else float("nan")
        direcao = "alta" if variacao > 0 else ("queda" if variacao < 0 else "estabilidade")
        return (
            f"Metodo: **{self.metodo}**. Ultimo valor observado: {ultimo_real:.2f}. "
            f"Primeira previsao: {primeiro_previsto:.2f} ({direcao} de {abs(variacao):.1f}% em relacao ao ultimo valor)."
        )


def error_metrics(y_real: pd.Series | np.ndarray, y_previsto: pd.Series | np.ndarray) -> dict:
    """MAE, RMSE, MAPE e MSE entre valores reais e previstos (mesma dimensao)."""
    y_real = np.asarray(y_real, dtype=float)
    y_previsto = np.asarray(y_previsto, dtype=float)
    if y_real.shape != y_previsto.shape:
        raise ValueError("y_real e y_previsto precisam ter o mesmo tamanho.")
    erro = y_real - y_previsto
    mae = float(np.mean(np.abs(erro)))
    mse = float(np.mean(erro**2))
    rmse = float(np.sqrt(mse))
    nao_zero = y_real != 0
    mape = float(np.mean(np.abs(erro[nao_zero] / y_real[nao_zero])) * 100) if nao_zero.any() else float("nan")
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape_%": mape}


def naive_forecast(serie: pd.Series, passos: int) -> ForecastResult:
    """Previsao ingenua: repete o ultimo valor observado (benchmark de referencia)."""
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    ultimo = valores.iloc[-1]
    idx = _future_index(valores, passos)
    tabela = pd.DataFrame({"previsao": [ultimo] * passos}, index=idx)
    return ForecastResult(metodo="Ingenuo (naive)", previsao=tabela, serie_historica=valores)


def moving_average_forecast(serie: pd.Series, passos: int, janela: int = 3) -> ForecastResult:
    """Previsao por media movel simples das ultimas `janela` observacoes."""
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    if len(valores) < janela:
        raise ValueError(f"A serie precisa ter ao menos {janela} observacoes (tamanho da janela).")
    media = valores.tail(janela).mean()
    idx = _future_index(valores, passos)
    tabela = pd.DataFrame({"previsao": [media] * passos}, index=idx)
    return ForecastResult(metodo=f"Media movel (janela={janela})", previsao=tabela, serie_historica=valores, detalhes={"janela": janela})


def exponential_smoothing_forecast(
    serie: pd.Series,
    passos: int,
    tendencia: str | None = "add",
    sazonalidade: str | None = None,
    periodo_sazonal: int | None = None,
) -> ForecastResult:
    """Suavizacao exponencial de Holt-Winters (nivel + tendencia + sazonalidade opcionais)."""
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    if sazonalidade and not periodo_sazonal:
        raise ValueError("Informe periodo_sazonal quando usar sazonalidade.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            valores,
            trend=tendencia,
            seasonal=sazonalidade,
            seasonal_periods=periodo_sazonal,
            initialization_method="estimated",
        )
        fit = modelo.fit()
    previsao = fit.forecast(passos)
    idx = _future_index(valores, passos)
    tabela = pd.DataFrame({"previsao": previsao.values}, index=idx)
    nome = "Holt-Winters"
    if tendencia:
        nome += f" (tendencia={tendencia}"
        nome += f", sazonalidade={sazonalidade})" if sazonalidade else ")"
    return ForecastResult(metodo=nome, previsao=tabela, serie_historica=valores, detalhes={"aic": float(fit.aic)})


def arima_forecast(serie: pd.Series, passos: int, p_max: int = 3, d_max: int = 2, q_max: int = 3) -> ForecastResult:
    """Previsao via ARIMA com ordem selecionada automaticamente por AIC."""
    resultado = auto_arima(serie, p_max=p_max, d_max=d_max, q_max=q_max)
    tabela = forecast_arima(resultado, passos)
    idx = _future_index(resultado.serie_original, passos)
    tabela.index = idx
    return ForecastResult(
        metodo=f"ARIMA{resultado.ordem}",
        previsao=tabela,
        serie_historica=resultado.serie_original,
        detalhes={"aic": resultado.aic, "bic": resultado.bic, "ordem": resultado.ordem},
    )


def _future_index(serie: pd.Series, passos: int) -> pd.Index:
    """Gera indice futuro coerente com o indice historico (datas ou inteiros sequenciais)."""
    if isinstance(serie.index, pd.DatetimeIndex) and serie.index.freq is not None:
        freq = serie.index.freq
        return pd.date_range(serie.index[-1] + freq, periods=passos, freq=freq)
    if isinstance(serie.index, pd.DatetimeIndex):
        inferido = pd.infer_freq(serie.index)
        if inferido:
            return pd.date_range(serie.index[-1], periods=passos + 1, freq=inferido)[1:]
        passo_medio = (serie.index[-1] - serie.index[0]) / max(1, len(serie) - 1)
        return pd.DatetimeIndex([serie.index[-1] + passo_medio * (i + 1) for i in range(passos)])
    if pd.api.types.is_integer_dtype(serie.index) or pd.api.types.is_float_dtype(serie.index):
        ultimo = serie.index[-1]
        return pd.Index([ultimo + i + 1 for i in range(passos)])
    return pd.RangeIndex(len(serie), len(serie) + passos)


MetodoForecast = Callable[[pd.Series, int], ForecastResult]

_METODOS_PADRAO: dict[str, MetodoForecast] = {
    "ingenuo": naive_forecast,
    "media_movel": lambda s, h: moving_average_forecast(s, h, janela=min(3, len(s) - 1) or 1),
    "holt_winters": lambda s, h: exponential_smoothing_forecast(s, h, tendencia="add", sazonalidade=None),
    "arima": lambda s, h: arima_forecast(s, h),
}


def backtest(serie: pd.Series, metodo: MetodoForecast, horizonte: int = 1, n_janelas: int = 5) -> dict:
    """Validacao walk-forward: treina em uma janela crescente, preve `horizonte` passos,
    compara com o valor real e repete deslizando a origem `n_janelas` vezes.
    Retorna as metricas de erro agregadas (MAE/RMSE/MAPE) fora da amostra.
    """
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    minimo_treino = max(10, horizonte + 5)
    if len(valores) < minimo_treino + n_janelas:
        raise ValueError(
            f"Serie muito curta para backtest: sao necessarias ao menos {minimo_treino + n_janelas} observacoes."
        )

    erros_reais, erros_previstos = [], []
    inicio = len(valores) - n_janelas - horizonte + 1
    for origem in range(inicio, len(valores) - horizonte + 1):
        treino = valores.iloc[:origem]
        teste = valores.iloc[origem : origem + horizonte]
        try:
            resultado = metodo(treino, horizonte)
        except Exception:
            continue
        previsto = resultado.previsao["previsao"].to_numpy()[: len(teste)]
        erros_reais.extend(teste.to_numpy())
        erros_previstos.extend(previsto)

    if not erros_reais:
        raise RuntimeError("Nao foi possivel completar nenhuma janela de backtest (modelo nao convergiu).")

    return error_metrics(np.array(erros_reais), np.array(erros_previstos))


def compare_models(serie: pd.Series, passos: int, n_janelas_backtest: int = 5) -> pd.DataFrame:
    """Roda ingenuo, media movel, Holt-Winters e ARIMA; ordena por RMSE em backtest."""
    linhas = []
    for nome, fn in _METODOS_PADRAO.items():
        entrada = {"mae": np.nan, "mse": np.nan, "rmse": np.nan, "mape_%": np.nan, "erro": None}
        try:
            metricas = backtest(serie, fn, horizonte=1, n_janelas=n_janelas_backtest)
            entrada.update(metricas)
        except Exception as exc:
            entrada["erro"] = str(exc)
        entrada["metodo"] = nome
        linhas.append(entrada)
    tabela = pd.DataFrame(linhas).set_index("metodo")
    return tabela.sort_values("rmse", na_position="last")
