"""Funcoes de formatacao para exibicao (UI e relatorios)."""
from __future__ import annotations

import math


def fmt_number(x: float, casas: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:,.{casas}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_percent(x: float, casas: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return fmt_number(x * 100, casas) + "%"


def fmt_currency(x: float, moeda: str = "R$") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    sinal = "-" if x < 0 else ""
    return f"{sinal}{moeda} {fmt_number(abs(x), 2)}"


def fmt_pvalue(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    if p < 0.0001:
        return "<0,0001"
    return fmt_number(p, 4)


def significance_stars(p: float) -> str:
    """Notacao classica de significancia estatistica."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""
