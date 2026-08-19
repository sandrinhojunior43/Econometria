import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def regressao_df(rng):
    n = 300
    x1 = rng.normal(10, 2, n)
    x2 = rng.normal(5, 1, n)
    ruido = rng.normal(0, 1, n)
    y = 3 + 2 * x1 - 1.5 * x2 + ruido
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def binario_df(rng):
    n = 400
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = -0.5 + 2.0 * x1 - 1.0 * x2
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def serie_temporal(rng):
    datas = pd.date_range("2016-01-01", periods=96, freq="MS")
    tendencia = np.linspace(50, 120, len(datas))
    sazonal = 8 * np.sin(2 * np.pi * datas.month / 12)
    ruido = rng.normal(0, 2, len(datas))
    valores = tendencia + sazonal + ruido
    return pd.Series(valores, index=datas)


@pytest.fixture
def painel_df(rng):
    empresas = [f"E{i}" for i in range(15)]
    anos = pd.date_range("2015-01-01", periods=8, freq="YS")
    linhas = []
    for empresa in empresas:
        efeito = rng.normal(0, 3)
        for ano in anos:
            x = rng.normal(10, 3)
            y = efeito + 1.2 * x + rng.normal(0, 1)
            linhas.append({"empresa": empresa, "ano": ano, "x": x, "y": y})
    return pd.DataFrame(linhas)
