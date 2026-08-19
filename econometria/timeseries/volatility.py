"""Modelagem de volatilidade (GARCH/ARCH), tipica de series financeiras (retornos)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass
class VolatilityResult:
    modelo: object
    p: int
    q: int
    vol_condicional: pd.Series
    parametros: pd.DataFrame
    aic: float
    bic: float
    log_likelihood: float

    def resumo_texto(self) -> str:
        return (
            f"Modelo GARCH({self.p},{self.q}) ajustado. AIC = {self.aic:.2f}, BIC = {self.bic:.2f}. "
            f"Volatilidade condicional media (desvio padrao) = {self.vol_condicional.mean():.4f}."
        )

    def prever_volatilidade(self, passos: int = 10) -> pd.DataFrame:
        previsao = self.modelo.forecast(horizon=passos, reindex=False)
        variancia = previsao.variance.iloc[-1]
        desvio = np.sqrt(variancia)
        return pd.DataFrame({"horizonte": range(1, passos + 1), "variancia_prevista": variancia.values, "desvio_padrao_previsto": desvio.values})


def fit_garch(
    retornos: pd.Series,
    p: int = 1,
    q: int = 1,
    escalar: float = 100.0,
) -> VolatilityResult:
    """Ajusta um modelo GARCH(p,q) a uma serie de retornos.

    Parameters
    ----------
    retornos: serie de retornos (ex.: variacao percentual de precos). Nao usar precos em nivel.
    escalar: fator multiplicativo aplicado antes do ajuste (recomendado pelo pacote `arch`
        para melhorar a convergencia numerica quando os retornos sao muito pequenos, ex.: 0,01).
        O resultado e reescalado de volta automaticamente.
    """
    valores = pd.to_numeric(retornos, errors="coerce").dropna() * escalar
    if len(valores) < 30:
        raise ValueError("Sao necessarias ao menos 30 observacoes de retorno para ajustar um GARCH com confianca.")

    modelo = arch_model(valores, vol="Garch", p=p, q=q, dist="normal")
    fit = modelo.fit(disp="off")

    vol_condicional = fit.conditional_volatility / escalar

    return VolatilityResult(
        modelo=fit,
        p=p,
        q=q,
        vol_condicional=vol_condicional,
        parametros=fit.params.to_frame("valor"),
        aic=float(fit.aic),
        bic=float(fit.bic),
        log_likelihood=float(fit.loglikelihood),
    )
