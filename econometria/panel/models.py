"""Dados em painel: Pooled OLS, Efeitos Fixos, Efeitos Aleatorios e teste de Hausman."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

from ..utils.formatting import significance_stars
from ..utils.validation import require_columns, require_min_rows, require_numeric


@dataclass
class PanelResult:
    modelo: object
    tipo: str  # "pooled", "efeitos_fixos", "efeitos_aleatorios"
    variavel_dependente: str
    variaveis_independentes: list[str]
    tabela_coeficientes: pd.DataFrame
    r2: float
    r2_within: float | None
    n_observacoes: int
    n_entidades: int
    n_periodos: int

    def resumo_texto(self) -> str:
        nomes = {"pooled": "Pooled OLS", "efeitos_fixos": "Efeitos Fixos", "efeitos_aleatorios": "Efeitos Aleatorios"}
        linhas = [
            f"Modelo de {nomes[self.tipo]} para **{self.variavel_dependente}** "
            f"({self.n_entidades} entidades x ate {self.n_periodos} periodos, n = {self.n_observacoes}).",
            f"R² = {self.r2:.4f}" + (f" | R² within = {self.r2_within:.4f}" if self.r2_within is not None else "") + ".",
        ]
        sig = self.tabela_coeficientes[self.tabela_coeficientes["p_valor"] < 0.05].index.tolist()
        sig = [v for v in sig if v not in ("const", "Intercept")]
        linhas.append(f"Variaveis significativas a 5%: {', '.join(sig)}." if sig else "Nenhuma variavel e significativa a 5%.")
        return "\n\n".join(linhas)


def _prepare_panel_data(df: pd.DataFrame, entidade: str, tempo: str, y: str, x: list[str]) -> pd.DataFrame:
    require_columns(df, [entidade, tempo, y] + x)
    require_numeric(df, [y] + x)
    dados = df[[entidade, tempo, y] + x].dropna()
    require_min_rows(dados, len(x) + 3, "modelo de dados em painel")
    if not pd.api.types.is_datetime64_any_dtype(dados[tempo]) and not pd.api.types.is_numeric_dtype(dados[tempo]):
        dados[tempo] = pd.to_datetime(dados[tempo], errors="coerce", format="mixed")
    dados = dados.set_index([entidade, tempo]).sort_index()
    return dados


def _build_coef_table(fit) -> pd.DataFrame:
    tabela = pd.DataFrame(
        {
            "coeficiente": fit.params,
            "erro_padrao": fit.std_errors,
            "estatistica_t": fit.tstats,
            "p_valor": fit.pvalues,
        }
    )
    tabela["significancia"] = tabela["p_valor"].apply(significance_stars)
    return tabela


def pooled_ols(df: pd.DataFrame, entidade: str, tempo: str, y: str, x: list[str]) -> PanelResult:
    """Pooled OLS - ignora a estrutura de painel (todas as observacoes tratadas igualmente)."""
    dados = _prepare_panel_data(df, entidade, tempo, y, x)
    exog = dados[x].assign(const=1.0)
    modelo = PooledOLS(dados[y], exog)
    fit = modelo.fit()
    return PanelResult(
        modelo=fit,
        tipo="pooled",
        variavel_dependente=y,
        variaveis_independentes=x,
        tabela_coeficientes=_build_coef_table(fit),
        r2=float(fit.rsquared),
        r2_within=None,
        n_observacoes=int(fit.nobs),
        n_entidades=dados.index.get_level_values(0).nunique(),
        n_periodos=dados.index.get_level_values(1).nunique(),
    )


def fixed_effects(
    df: pd.DataFrame,
    entidade: str,
    tempo: str,
    y: str,
    x: list[str],
    efeitos_entidade: bool = True,
    efeitos_tempo: bool = False,
) -> PanelResult:
    """Modelo de Efeitos Fixos (within estimator) via linearmodels.PanelOLS."""
    dados = _prepare_panel_data(df, entidade, tempo, y, x)
    modelo = PanelOLS(dados[y], dados[x], entity_effects=efeitos_entidade, time_effects=efeitos_tempo)
    fit = modelo.fit(cov_type="clustered", cluster_entity=True)
    return PanelResult(
        modelo=fit,
        tipo="efeitos_fixos",
        variavel_dependente=y,
        variaveis_independentes=x,
        tabela_coeficientes=_build_coef_table(fit),
        r2=float(fit.rsquared),
        r2_within=float(fit.rsquared_within),
        n_observacoes=int(fit.nobs),
        n_entidades=dados.index.get_level_values(0).nunique(),
        n_periodos=dados.index.get_level_values(1).nunique(),
    )


def random_effects(df: pd.DataFrame, entidade: str, tempo: str, y: str, x: list[str]) -> PanelResult:
    """Modelo de Efeitos Aleatorios via linearmodels.RandomEffects."""
    dados = _prepare_panel_data(df, entidade, tempo, y, x)
    exog = dados[x].assign(const=1.0)
    modelo = RandomEffects(dados[y], exog)
    fit = modelo.fit()
    return PanelResult(
        modelo=fit,
        tipo="efeitos_aleatorios",
        variavel_dependente=y,
        variaveis_independentes=x,
        tabela_coeficientes=_build_coef_table(fit),
        r2=float(fit.rsquared),
        r2_within=float(fit.rsquared_within) if hasattr(fit, "rsquared_within") else None,
        n_observacoes=int(fit.nobs),
        n_entidades=dados.index.get_level_values(0).nunique(),
        n_periodos=dados.index.get_level_values(1).nunique(),
    )


def hausman_test(fe_resultado: PanelResult, re_resultado: PanelResult, alfa: float = 0.05) -> dict:
    """Teste de Hausman: compara Efeitos Fixos (consistente sob H0 e H1) com
    Efeitos Aleatorios (eficiente sob H0, inconsistente sob H1).

    H0: os efeitos individuais nao sao correlacionados com os regressores
        (efeitos aleatorios e o modelo preferido, mais eficiente).
    H1: ha correlacao (efeitos fixos e necessario para consistencia).
    """
    if fe_resultado.tipo != "efeitos_fixos" or re_resultado.tipo != "efeitos_aleatorios":
        raise ValueError("Informe um resultado de fixed_effects() e um de random_effects().")

    fe = fe_resultado.modelo
    re = re_resultado.modelo

    comuns = [v for v in fe.params.index if v in re.params.index and v != "const"]
    if not comuns:
        raise ValueError("Nao ha variaveis em comum entre os dois modelos para comparar.")

    b_fe = fe.params[comuns].to_numpy()
    b_re = re.params[comuns].to_numpy()
    diff = b_fe - b_re

    cov_fe = fe.cov.loc[comuns, comuns].to_numpy()
    cov_re = re.cov.loc[comuns, comuns].to_numpy()
    cov_diff = cov_fe - cov_re

    try:
        cov_diff_inv = np.linalg.pinv(cov_diff)
        estatistica = float(diff.T @ cov_diff_inv @ diff)
        graus_liberdade = len(comuns)
        from scipy import stats

        p_valor = float(1 - stats.chi2.cdf(estatistica, graus_liberdade))
    except np.linalg.LinAlgError:
        estatistica, p_valor, graus_liberdade = float("nan"), float("nan"), len(comuns)

    if np.isnan(p_valor):
        conclusao = "Nao foi possivel calcular o teste (matriz de covariancia singular). Avalie manualmente as diferencas de coeficiente."
    elif p_valor < alfa:
        conclusao = "Rejeita-se H0: use **Efeitos Fixos** (ha correlacao entre efeitos individuais e regressores)."
    else:
        conclusao = "Nao se rejeita H0: **Efeitos Aleatorios** e preferivel (mais eficiente e nao ha evidencia de correlacao)."

    return {
        "estatistica_qui2": estatistica,
        "graus_liberdade": graus_liberdade,
        "p_valor": p_valor,
        "conclusao": conclusao,
    }
