"""Testes de estacionariedade: Augmented Dickey-Fuller (ADF) e KPSS."""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    adf_estatistica: float
    adf_p_valor: float
    adf_valores_criticos: dict
    kpss_estatistica: float
    kpss_p_valor: float
    kpss_valores_criticos: dict
    conclusao: str
    estacionaria: bool

    def resumo_texto(self) -> str:
        return (
            f"ADF: estatistica = {self.adf_estatistica:.4f}, p-valor = {self.adf_p_valor:.4f} "
            f"(H0: raiz unitaria / nao estacionaria).\n\n"
            f"KPSS: estatistica = {self.kpss_estatistica:.4f}, p-valor = {self.kpss_p_valor:.4f} "
            f"(H0: estacionaria).\n\n"
            f"**Conclusao:** {self.conclusao}"
        )


def test_stationarity(serie: pd.Series, alfa: float = 0.05) -> StationarityResult:
    """Combina ADF e KPSS para uma conclusao mais robusta sobre estacionariedade.

    - ADF rejeita H0 (p < alfa) -> evidencia de estacionariedade.
    - KPSS rejeita H0 (p < alfa) -> evidencia de nao-estacionariedade.
    Quando os dois testes concordam, a conclusao e direta; quando divergem,
    o resultado e sinalizado como "inconclusivo" (comum na pratica com
    series curtas ou proximas do limiar).
    """
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    if len(valores) < 10:
        raise ValueError("Sao necessarias ao menos 10 observacoes para testar estacionariedade.")

    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(valores, autolag="AIC")
    with warnings.catch_warnings():
        # o statsmodels avisa quando a estatistica do KPSS esta fora da tabela de
        # p-valores tabelados; o p-valor retornado ja e o limite correto (< ou > tabela).
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, _, kpss_crit = kpss(valores, regression="c", nlags="auto")

    adf_estacionaria = adf_p < alfa
    kpss_estacionaria = kpss_p >= alfa

    if adf_estacionaria and kpss_estacionaria:
        conclusao = "os dois testes indicam que a serie e **estacionaria**."
        estacionaria = True
    elif not adf_estacionaria and not kpss_estacionaria:
        conclusao = "os dois testes indicam que a serie **nao e estacionaria** (considere diferenciar, ex.: `series.diff()`)."
        estacionaria = False
    else:
        conclusao = (
            "os testes **divergem** - resultado inconclusivo. Isso pode indicar uma serie "
            "estacionaria em tendencia (trend-stationary) ou a necessidade de mais observacoes."
        )
        estacionaria = adf_estacionaria  # heuristica: prioriza ADF, mas sinaliza divergencia

    return StationarityResult(
        adf_estatistica=float(adf_stat),
        adf_p_valor=float(adf_p),
        adf_valores_criticos={k: float(v) for k, v in adf_crit.items()},
        kpss_estatistica=float(kpss_stat),
        kpss_p_valor=float(kpss_p),
        kpss_valores_criticos={k: float(v) for k, v in kpss_crit.items()},
        conclusao=conclusao,
        estacionaria=estacionaria,
    )
