"""Decomposicao de series temporais em tendencia, sazonalidade e residuo."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose


@dataclass
class DecompositionResult:
    tendencia: pd.Series
    sazonalidade: pd.Series
    residuo: pd.Series
    observado: pd.Series
    modelo: str
    periodo: int
    forca_tendencia: float
    forca_sazonalidade: float

    def resumo_texto(self) -> str:
        return (
            f"Decomposicao {self.modelo} com periodo sazonal = {self.periodo}.\n\n"
            f"Forca da tendencia: {self.forca_tendencia:.2f} (0 = nenhuma, 1 = dominante).\n\n"
            f"Forca da sazonalidade: {self.forca_sazonalidade:.2f} (0 = nenhuma, 1 = dominante)."
        )


def _forca(componente: pd.Series, residuo: pd.Series) -> float:
    """Medida de forca de Hyndman: 1 - Var(residuo) / Var(componente + residuo)."""
    import numpy as np

    combinado = componente.fillna(0) + residuo.fillna(0)
    var_resid = residuo.dropna().var()
    var_comb = combinado.dropna().var()
    if var_comb == 0 or pd.isna(var_comb):
        return 0.0
    return float(max(0.0, min(1.0, 1 - var_resid / var_comb)))


def decompose_series(
    serie: pd.Series,
    periodo: int,
    metodo: str = "stl",
    modelo: str = "additive",
) -> DecompositionResult:
    """Decompoe a serie em tendencia + sazonalidade + residuo.

    Parameters
    ----------
    periodo: numero de observacoes por ciclo sazonal (ex.: 12 para dados mensais com sazonalidade anual).
    metodo: 'stl' (robusto, recomendado) ou 'classica' (media movel classica).
    modelo: 'additive' ou 'multiplicative' (apenas para metodo 'classica').
    """
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    if len(valores) < 2 * periodo:
        raise ValueError(f"Sao necessarias ao menos {2 * periodo} observacoes para decompor com periodo {periodo}.")

    if metodo == "stl":
        resultado = STL(valores, period=periodo, robust=True).fit()
        tendencia, sazonalidade, residuo = resultado.trend, resultado.seasonal, resultado.resid
        nome_modelo = "STL (robusta)"
    elif metodo == "classica":
        resultado = seasonal_decompose(valores, model=modelo, period=periodo, extrapolate_trend="freq")
        tendencia, sazonalidade, residuo = resultado.trend, resultado.seasonal, resultado.resid
        nome_modelo = f"classica ({modelo})"
    else:
        raise ValueError("metodo deve ser 'stl' ou 'classica'.")

    return DecompositionResult(
        tendencia=tendencia,
        sazonalidade=sazonalidade,
        residuo=residuo,
        observado=valores,
        modelo=nome_modelo,
        periodo=periodo,
        forca_tendencia=_forca(tendencia, residuo),
        forca_sazonalidade=_forca(sazonalidade, residuo),
    )
