"""Modelagem ARIMA/SARIMA com selecao automatica de ordem por AIC e previsao."""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ArimaResult:
    modelo: object  # ARIMAResultsWrapper
    ordem: tuple[int, int, int]
    ordem_sazonal: tuple[int, int, int, int] | None
    aic: float
    bic: float
    n_observacoes: int
    serie_original: pd.Series

    def resumo_texto(self) -> str:
        p, d, q = self.ordem
        base = f"Modelo ARIMA({p},{d},{q})"
        if self.ordem_sazonal:
            P, D, Q, s = self.ordem_sazonal
            base += f"(P={P},D={D},Q={Q})[{s}]"
        return f"{base} ajustado com AIC = {self.aic:.2f} e BIC = {self.bic:.2f} (n = {self.n_observacoes})."


def fit_arima(
    serie: pd.Series,
    ordem: tuple[int, int, int],
    ordem_sazonal: tuple[int, int, int, int] | None = None,
) -> ArimaResult:
    """Ajusta um ARIMA(p,d,q) [SARIMA se ordem_sazonal for informada] com ordem especifica."""
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ARIMA(valores, order=ordem, seasonal_order=ordem_sazonal or (0, 0, 0, 0))
        fit = modelo.fit()
    return ArimaResult(
        modelo=fit,
        ordem=ordem,
        ordem_sazonal=ordem_sazonal,
        aic=float(fit.aic),
        bic=float(fit.bic),
        n_observacoes=int(fit.nobs),
        serie_original=valores,
    )


def auto_arima(
    serie: pd.Series,
    p_max: int = 3,
    d_max: int = 2,
    q_max: int = 3,
    ordem_sazonal: tuple[int, int, int, int] | None = None,
) -> ArimaResult:
    """Busca em grade (p,d,q) em [0,p_max]x[0,d_max]x[0,q_max] minimizando o AIC.

    Abordagem simples e transparente (sem dependencia externa tipo pmdarima):
    testa todas as combinacoes plausiveis e mantem a de menor AIC que convergir.
    Adequado para series de porte pequeno/medio, tipico de uso pessoal/empresarial.
    """
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    if len(valores) < 15:
        raise ValueError("Sao necessarias ao menos 15 observacoes para selecionar um modelo ARIMA automaticamente.")

    melhor: ArimaResult | None = None
    combinacoes = list(itertools.product(range(p_max + 1), range(d_max + 1), range(q_max + 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, d, q in combinacoes:
            if p == 0 and q == 0:
                continue
            try:
                modelo = ARIMA(valores, order=(p, d, q), seasonal_order=ordem_sazonal or (0, 0, 0, 0))
                fit = modelo.fit()
            except Exception:
                continue
            if melhor is None or fit.aic < melhor.aic:
                melhor = ArimaResult(
                    modelo=fit,
                    ordem=(p, d, q),
                    ordem_sazonal=ordem_sazonal,
                    aic=float(fit.aic),
                    bic=float(fit.bic),
                    n_observacoes=int(fit.nobs),
                    serie_original=valores,
                )
    if melhor is None:
        raise RuntimeError("Nenhum modelo ARIMA convergiu para os parametros informados. Tente reduzir p_max/q_max.")
    return melhor


def forecast_arima(resultado: ArimaResult, passos: int, alfa: float = 0.05) -> pd.DataFrame:
    """Gera previsao de `passos` periodos a frente com intervalo de confianca (1 - alfa)."""
    previsao = resultado.modelo.get_forecast(steps=passos)
    tabela = previsao.summary_frame(alpha=alfa)
    tabela = tabela.rename(
        columns={
            "mean": "previsao",
            "mean_se": "erro_padrao",
            "mean_ci_lower": "ic_inferior",
            "mean_ci_upper": "ic_superior",
        }
    )
    return tabela
