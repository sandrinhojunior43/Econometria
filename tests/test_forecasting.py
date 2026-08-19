import numpy as np
import pandas as pd
import pytest

from econometria.forecasting import (
    error_metrics,
    exponential_smoothing_forecast,
    moving_average_forecast,
    naive_forecast,
)


def test_error_metrics_zero_quando_previsao_perfeita():
    y = np.array([1.0, 2.0, 3.0])
    metricas = error_metrics(y, y)
    assert metricas["mae"] == pytest.approx(0)
    assert metricas["rmse"] == pytest.approx(0)
    assert metricas["mape_%"] == pytest.approx(0)


def test_error_metrics_tamanhos_diferentes_gera_erro():
    with pytest.raises(ValueError):
        error_metrics([1, 2, 3], [1, 2])


def test_naive_forecast_repete_ultimo_valor():
    serie = pd.Series([10, 20, 30, 40])
    resultado = naive_forecast(serie, 3)
    assert (resultado.previsao["previsao"] == 40).all()


def test_moving_average_forecast():
    serie = pd.Series([10, 20, 30])
    resultado = moving_average_forecast(serie, 2, janela=3)
    assert resultado.previsao["previsao"].iloc[0] == pytest.approx(20)


def test_exponential_smoothing_forecast_tamanho_correto(serie_temporal):
    resultado = exponential_smoothing_forecast(serie_temporal, 12, tendencia="add")
    assert len(resultado.previsao) == 12
    assert resultado.previsao["previsao"].notna().all()
