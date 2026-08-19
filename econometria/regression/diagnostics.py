"""Diagnosticos de um modelo de regressao: multicolinearidade, heterocedasticidade,
autocorrelacao, normalidade dos residuos e forma funcional (RESET).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from .linear import RegressionResult


def variance_inflation_factors(df: pd.DataFrame, x: list[str]) -> pd.DataFrame:
    """VIF por variavel - valores > 10 costumam indicar multicolinearidade problematica."""
    dados = df[x].dropna()
    X = sm.add_constant(dados)
    vifs = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vifs.append({"variavel": col, "vif": variance_inflation_factor(X.values, i)})
    tabela = pd.DataFrame(vifs).set_index("variavel")
    tabela["alerta"] = tabela["vif"].apply(lambda v: "multicolinearidade alta" if v > 10 else ("moderada" if v > 5 else "ok"))
    return tabela


@dataclass
class DiagnosticsResult:
    vif: pd.DataFrame | None
    breusch_pagan: dict
    white: dict
    durbin_watson: float
    breusch_godfrey: dict
    jarque_bera_residuos: dict
    reset_test: dict
    alertas: list[str] = field(default_factory=list)

    def resumo_texto(self) -> str:
        linhas = []
        if self.alertas:
            linhas.append("**Alertas de diagnostico:**")
            linhas.extend(f"- {a}" for a in self.alertas)
        else:
            linhas.append("Nenhum problema grave identificado nos testes de diagnostico padrao.")
        return "\n".join(linhas)


def run_diagnostics(resultado: RegressionResult, df: pd.DataFrame, nlags_bg: int = 2) -> DiagnosticsResult:
    """Roda a bateria padrao de diagnosticos sobre um RegressionResult (ver `ols_regression`)."""
    fit = resultado.modelo
    resid = fit.resid
    exog = fit.model.exog
    exog_names = fit.model.exog_names

    alertas = []

    vif_tabela = None
    x_sem_const = [v for v in resultado.variaveis_independentes]
    if len(x_sem_const) >= 2:
        try:
            vif_tabela = variance_inflation_factors(df, x_sem_const)
            if (vif_tabela["vif"] > 10).any():
                piores = vif_tabela[vif_tabela["vif"] > 10].index.tolist()
                alertas.append(f"Multicolinearidade alta (VIF > 10) em: {', '.join(piores)}.")
        except Exception:
            vif_tabela = None

    bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(resid, exog)
    breusch_pagan = {"estatistica_lm": bp_stat, "p_valor_lm": bp_p, "estatistica_f": bp_f, "p_valor_f": bp_fp}
    if bp_p < 0.05:
        alertas.append(f"Heterocedasticidade detectada pelo teste de Breusch-Pagan (p = {bp_p:.4f}).")

    try:
        w_stat, w_p, w_f, w_fp = het_white(resid, exog)
        white = {"estatistica_lm": w_stat, "p_valor_lm": w_p, "estatistica_f": w_f, "p_valor_f": w_fp}
        if w_p < 0.05:
            alertas.append(f"Heterocedasticidade detectada pelo teste de White (p = {w_p:.4f}).")
    except Exception:
        white = {"erro": "Nao foi possivel calcular (numero de variaveis/observacoes insuficiente)."}

    dw = float(durbin_watson(resid))
    if dw < 1.5:
        alertas.append(f"Durbin-Watson = {dw:.2f} sugere autocorrelacao positiva nos residuos.")
    elif dw > 2.5:
        alertas.append(f"Durbin-Watson = {dw:.2f} sugere autocorrelacao negativa nos residuos.")

    try:
        bg_stat, bg_p, bg_f, bg_fp = acorr_breusch_godfrey(fit, nlags=nlags_bg)
        breusch_godfrey = {"estatistica_lm": bg_stat, "p_valor_lm": bg_p, "estatistica_f": bg_f, "p_valor_f": bg_fp, "nlags": nlags_bg}
        if bg_p < 0.05:
            alertas.append(f"Autocorrelacao serial detectada pelo teste de Breusch-Godfrey (p = {bg_p:.4f}).")
    except Exception:
        breusch_godfrey = {"erro": "Nao foi possivel calcular."}

    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    jarque_bera_res = {"estatistica": jb_stat, "p_valor": jb_p, "assimetria": skew, "curtose": kurt}
    if jb_p < 0.05:
        alertas.append(f"Residuos nao seguem distribuicao normal pelo teste de Jarque-Bera (p = {jb_p:.4f}).")

    try:
        reset = linear_reset(fit, power=3, use_f=True)
        reset_test = {"estatistica_f": float(reset.fvalue), "p_valor": float(reset.pvalue)}
        if reset.pvalue < 0.05:
            alertas.append(f"Teste RESET de Ramsey sugere ma especificacao da forma funcional (p = {reset.pvalue:.4f}).")
    except Exception:
        reset_test = {"erro": "Nao foi possivel calcular."}

    return DiagnosticsResult(
        vif=vif_tabela,
        breusch_pagan=breusch_pagan,
        white=white,
        durbin_watson=dw,
        breusch_godfrey=breusch_godfrey,
        jarque_bera_residuos=jarque_bera_res,
        reset_test=reset_test,
        alertas=alertas,
    )
