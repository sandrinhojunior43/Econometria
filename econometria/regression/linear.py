"""Regressao linear multipla: OLS e WLS, com erros padrao robustos opcionais."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..utils.formatting import fmt_pvalue, significance_stars
from ..utils.validation import require_columns, require_min_rows, require_numeric

COV_TYPES_VALIDOS = {"nonrobust", "HC0", "HC1", "HC2", "HC3"}


@dataclass
class RegressionResult:
    modelo: object  # statsmodels RegressionResultsWrapper
    variavel_dependente: str
    variaveis_independentes: list[str]
    tabela_coeficientes: pd.DataFrame
    r2: float
    r2_ajustado: float
    f_estatistica: float
    f_p_valor: float
    aic: float
    bic: float
    n_observacoes: int
    cov_type: str
    residuos: pd.Series = field(repr=False)
    valores_ajustados: pd.Series = field(repr=False)

    def resumo_texto(self) -> str:
        linhas = [
            f"Regressao de **{self.variavel_dependente}** sobre {', '.join(self.variaveis_independentes)} "
            f"(n = {self.n_observacoes}, erros padrao: {self.cov_type}).",
            f"R² = {self.r2:.4f} | R² ajustado = {self.r2_ajustado:.4f} | "
            f"F = {self.f_estatistica:.2f} (p = {fmt_pvalue(self.f_p_valor)}).",
        ]
        significativos = self.tabela_coeficientes[self.tabela_coeficientes["p_valor"] < 0.05].index.tolist()
        significativos = [v for v in significativos if v not in ("const", "Intercept")]
        if significativos:
            linhas.append(f"Variaveis estatisticamente significativas a 5%: {', '.join(significativos)}.")
        else:
            linhas.append("Nenhuma variavel explicativa e estatisticamente significativa a 5%.")
        qualidade = "boa" if self.r2_ajustado > 0.7 else ("moderada" if self.r2_ajustado > 0.4 else "baixa")
        linhas.append(f"O modelo explica uma parcela {qualidade} da variabilidade de {self.variavel_dependente}.")
        return "\n\n".join(linhas)

    def prever(self, novos_dados: pd.DataFrame) -> pd.Series:
        X = sm.add_constant(novos_dados[self.variaveis_independentes], has_constant="add")
        X = X[self.modelo.params.index]
        return self.modelo.predict(X)


def _build_coef_table(fit_result) -> pd.DataFrame:
    conf = fit_result.conf_int()
    conf.columns = ["ic_inferior_95", "ic_superior_95"]
    tabela = pd.DataFrame(
        {
            "coeficiente": fit_result.params,
            "erro_padrao": fit_result.bse,
            "estatistica_t": fit_result.tvalues,
            "p_valor": fit_result.pvalues,
        }
    ).join(conf)
    tabela["significancia"] = tabela["p_valor"].apply(significance_stars)
    return tabela


def ols_regression(
    df: pd.DataFrame,
    y: str,
    x: list[str],
    incluir_intercepto: bool = True,
    cov_type: str = "nonrobust",
) -> RegressionResult:
    """Regressao linear multipla por Minimos Quadrados Ordinarios (OLS).

    Parameters
    ----------
    df: dados de entrada.
    y: nome da coluna dependente.
    x: lista de colunas independentes.
    incluir_intercepto: se True, adiciona constante ao modelo.
    cov_type: tipo de matriz de covariancia ('nonrobust' ou 'HC0'..'HC3' para
        erros padrao robustos a heterocedasticidade de White).
    """
    if cov_type not in COV_TYPES_VALIDOS:
        raise ValueError(f"cov_type invalido. Use um de: {sorted(COV_TYPES_VALIDOS)}")
    require_columns(df, [y] + x)
    require_numeric(df, [y] + x)
    dados = df[[y] + x].dropna()
    require_min_rows(dados, len(x) + 2, "regressao linear")

    X = dados[x]
    if incluir_intercepto:
        X = sm.add_constant(X)
    modelo = sm.OLS(dados[y], X)
    fit = modelo.fit(cov_type=cov_type) if cov_type != "nonrobust" else modelo.fit()

    return RegressionResult(
        modelo=fit,
        variavel_dependente=y,
        variaveis_independentes=x,
        tabela_coeficientes=_build_coef_table(fit),
        r2=float(fit.rsquared),
        r2_ajustado=float(fit.rsquared_adj),
        f_estatistica=float(fit.fvalue) if fit.fvalue is not None else float("nan"),
        f_p_valor=float(fit.f_pvalue) if fit.f_pvalue is not None else float("nan"),
        aic=float(fit.aic),
        bic=float(fit.bic),
        n_observacoes=int(fit.nobs),
        cov_type=cov_type,
        residuos=fit.resid,
        valores_ajustados=fit.fittedvalues,
    )


def wls_regression(
    df: pd.DataFrame,
    y: str,
    x: list[str],
    peso: str,
    incluir_intercepto: bool = True,
) -> RegressionResult:
    """Regressao por Minimos Quadrados Ponderados (WLS) - util quando ha heterocedasticidade conhecida."""
    require_columns(df, [y, peso] + x)
    require_numeric(df, [y, peso] + x)
    dados = df[[y, peso] + x].dropna()
    require_min_rows(dados, len(x) + 2, "regressao WLS")

    X = dados[x]
    if incluir_intercepto:
        X = sm.add_constant(X)
    modelo = sm.WLS(dados[y], X, weights=dados[peso])
    fit = modelo.fit()

    return RegressionResult(
        modelo=fit,
        variavel_dependente=y,
        variaveis_independentes=x,
        tabela_coeficientes=_build_coef_table(fit),
        r2=float(fit.rsquared),
        r2_ajustado=float(fit.rsquared_adj),
        f_estatistica=float(fit.fvalue) if fit.fvalue is not None else float("nan"),
        f_p_valor=float(fit.f_pvalue) if fit.f_pvalue is not None else float("nan"),
        aic=float(fit.aic),
        bic=float(fit.bic),
        n_observacoes=int(fit.nobs),
        cov_type="nonrobust (WLS)",
        residuos=fit.resid,
        valores_ajustados=fit.fittedvalues,
    )
