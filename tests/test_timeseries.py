import numpy as np
import pandas as pd
import pytest

from econometria.timeseries import (
    auto_arima,
    decompose_series,
    fit_garch,
    forecast_arima,
)
# Importado com alias: um nome comecando com "test_" no escopo do modulo seria
# coletado pelo pytest como se fosse um caso de teste.
from econometria.timeseries import test_stationarity as verificar_estacionariedade


def test_stationarity_detecta_serie_nao_estacionaria(serie_temporal):
    # serie_temporal tem tendencia forte -> nao deve ser estacionaria em nivel
    resultado = verificar_estacionariedade(serie_temporal)
    assert resultado.adf_p_valor > 0.01


def test_stationarity_detecta_serie_estacionaria(rng):
    ruido_branco = pd.Series(rng.normal(0, 1, 200))
    resultado = verificar_estacionariedade(ruido_branco)
    assert resultado.estacionaria is True


def test_stationarity_poucos_dados_gera_erro():
    with pytest.raises(ValueError):
        verificar_estacionariedade(pd.Series([1, 2, 3]))


def test_decompose_series_recupera_sazonalidade(rng):
    datas = pd.date_range("2018-01-01", periods=72, freq="MS")
    sazonal_verdadeira = 10 * np.sin(2 * np.pi * datas.month / 12)
    serie = pd.Series(50 + np.arange(72) * 0.2 + sazonal_verdadeira + rng.normal(0, 0.5, 72), index=datas)
    dec = decompose_series(serie, periodo=12)
    assert dec.forca_sazonalidade > 0.7  # sazonalidade e dominante e deve ser bem capturada
    assert dec.forca_tendencia > 0.5


def test_auto_arima_e_forecast(serie_temporal):
    resultado = auto_arima(serie_temporal, p_max=2, d_max=1, q_max=2)
    assert resultado.aic == resultado.aic  # nao e NaN
    previsao = forecast_arima(resultado, 6)
    assert len(previsao) == 6
    assert (previsao["ic_superior"] >= previsao["ic_inferior"]).all()


def test_garch_volatilidade_positiva(rng):
    retornos = pd.Series(rng.normal(0, 0.02, 600))
    resultado = fit_garch(retornos)
    assert (resultado.vol_condicional > 0).all()
    previsao = resultado.prever_volatilidade(5)
    assert len(previsao) == 5
    assert (previsao["desvio_padrao_previsto"] > 0).all()


def test_garch_exige_minimo_observacoes():
    with pytest.raises(ValueError):
        fit_garch(pd.Series([0.01, -0.02, 0.03]))
