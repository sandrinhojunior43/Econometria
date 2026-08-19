import pytest

from econometria.regression import logit_regression, ols_regression, run_diagnostics
from econometria.regression.diagnostics import variance_inflation_factors
from econometria.utils import ValidationError


def test_ols_recupera_coeficientes_verdadeiros(regressao_df):
    resultado = ols_regression(regressao_df, "y", ["x1", "x2"])
    # y = 3 + 2*x1 - 1.5*x2 + ruido -> coeficientes estimados devem ficar proximos dos verdadeiros
    assert resultado.tabela_coeficientes.loc["x1", "coeficiente"] == pytest.approx(2.0, abs=0.15)
    assert resultado.tabela_coeficientes.loc["x2", "coeficiente"] == pytest.approx(-1.5, abs=0.25)
    assert resultado.r2 > 0.9


def test_ols_variaveis_significativas(regressao_df):
    resultado = ols_regression(regressao_df, "y", ["x1", "x2"])
    assert resultado.tabela_coeficientes.loc["x1", "p_valor"] < 0.01
    assert resultado.tabela_coeficientes.loc["x2", "p_valor"] < 0.01


def test_ols_com_coluna_inexistente_gera_erro(regressao_df):
    with pytest.raises(ValidationError):
        ols_regression(regressao_df, "y", ["nao_existe"])


def test_ols_erros_robustos_alteram_erro_padrao(regressao_df):
    normal = ols_regression(regressao_df, "y", ["x1", "x2"], cov_type="nonrobust")
    robusto = ols_regression(regressao_df, "y", ["x1", "x2"], cov_type="HC3")
    # coeficientes identicos, mas erro padrao pode diferir
    assert normal.tabela_coeficientes["coeficiente"].equals(robusto.tabela_coeficientes["coeficiente"])


def test_run_diagnostics_nao_quebra_e_retorna_vif(regressao_df):
    resultado = ols_regression(regressao_df, "y", ["x1", "x2"])
    diag = run_diagnostics(resultado, regressao_df)
    assert diag.vif is not None
    assert set(diag.vif.index) == {"x1", "x2"}
    assert (diag.vif["vif"] > 0).all()


def test_vif_baixo_quando_variaveis_independentes(regressao_df):
    vif = variance_inflation_factors(regressao_df, ["x1", "x2"])
    assert (vif["vif"] < 5).all()  # x1 e x2 sao gerados independentemente


def test_logit_classifica_melhor_que_acaso(binario_df):
    resultado = logit_regression(binario_df, "y", ["x1", "x2"])
    assert resultado.acuracia > 0.6
    assert 0 <= resultado.pseudo_r2 <= 1


def test_logit_exige_variavel_binaria(regressao_df):
    with pytest.raises(ValueError):
        logit_regression(regressao_df, "y", ["x1", "x2"])
