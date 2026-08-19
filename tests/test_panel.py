import pytest

from econometria.panel import fixed_effects, hausman_test, pooled_ols, random_effects


def test_fixed_effects_recupera_coeficiente_verdadeiro(painel_df):
    resultado = fixed_effects(painel_df, "empresa", "ano", "y", ["x"])
    assert resultado.tabela_coeficientes.loc["x", "coeficiente"] == pytest.approx(1.2, abs=0.2)


def test_pooled_ols_roda_sem_erro(painel_df):
    resultado = pooled_ols(painel_df, "empresa", "ano", "y", ["x"])
    assert resultado.n_entidades == 15
    assert resultado.tipo == "pooled"


def test_random_effects_roda_sem_erro(painel_df):
    resultado = random_effects(painel_df, "empresa", "ano", "y", ["x"])
    assert resultado.tipo == "efeitos_aleatorios"
    assert resultado.tabela_coeficientes.loc["x", "coeficiente"] == pytest.approx(1.2, abs=0.2)


def test_hausman_test_estrutura_correta(painel_df):
    fe = fixed_effects(painel_df, "empresa", "ano", "y", ["x"])
    re_ = random_effects(painel_df, "empresa", "ano", "y", ["x"])
    resultado = hausman_test(fe, re_)
    assert "p_valor" in resultado
    assert "conclusao" in resultado
    assert resultado["graus_liberdade"] == 1


def test_hausman_test_exige_tipos_corretos(painel_df):
    fe = fixed_effects(painel_df, "empresa", "ano", "y", ["x"])
    with pytest.raises(ValueError):
        hausman_test(fe, fe)
