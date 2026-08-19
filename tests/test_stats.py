import numpy as np
import pandas as pd
import pytest

from econometria.stats import (
    anova_one_way,
    chi_square_independence,
    correlation_matrix,
    describe,
    normality_test,
    t_test_paired,
    t_test_two_sample,
)
from econometria.utils import ValidationError


def test_describe_valores_basicos():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    resultado = describe(df, ["x"])
    assert resultado.tabela.loc["x", "media"] == pytest.approx(3.0)
    assert resultado.tabela.loc["x", "mediana"] == pytest.approx(3.0)
    assert resultado.tabela.loc["x", "minimo"] == pytest.approx(1.0)
    assert resultado.tabela.loc["x", "maximo"] == pytest.approx(5.0)


def test_describe_coluna_inexistente_gera_erro():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValidationError):
        describe(df, ["y"])


def test_correlation_matrix_diagonal_e_um():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    matriz = correlation_matrix(df, ["a", "b"])
    assert matriz.loc["a", "a"] == pytest.approx(1.0)
    assert matriz.loc["a", "b"] == pytest.approx(-1.0, abs=1e-9)


def test_normality_test_distribuicao_normal(rng):
    dados = pd.Series(rng.normal(0, 1, 500))
    resultado = normality_test(dados)
    assert resultado["jarque_bera_p_valor"] > 0.01


def test_t_test_two_sample_detecta_diferenca(rng):
    a = pd.Series(rng.normal(0, 1, 200))
    b = pd.Series(rng.normal(3, 1, 200))
    resultado = t_test_two_sample(a, b)
    assert resultado.rejeita_h0 is True
    assert resultado.p_valor < 0.001


def test_t_test_two_sample_nao_detecta_quando_mesma_distribuicao(rng):
    a = pd.Series(rng.normal(0, 1, 300))
    b = pd.Series(rng.normal(0, 1, 300))
    resultado = t_test_two_sample(a, b)
    assert resultado.p_valor > 0.01


def test_t_test_paired():
    a = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    b = a - 2 + pd.Series([0.1, -0.2, 0.15, -0.1, 0.05])  # diferenca ~constante, com leve ruido
    resultado = t_test_paired(a, b)
    assert resultado.detalhes["diferenca_media"] == pytest.approx(2.0, abs=0.1)


def test_anova_detecta_diferenca_entre_grupos(rng):
    df = pd.DataFrame(
        {
            "valor": np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50), rng.normal(10, 1, 50)]),
            "grupo": ["a"] * 50 + ["b"] * 50 + ["c"] * 50,
        }
    )
    resultado = anova_one_way(df, "valor", "grupo")
    assert resultado.rejeita_h0 is True


def test_chi_square_independencia():
    df = pd.DataFrame(
        {
            "genero": ["M", "F"] * 50,
            "compra": (["sim"] * 40 + ["nao"] * 10) * 2,
        }
    )
    resultado = chi_square_independence(df, "genero", "compra")
    assert resultado.p_valor >= 0
