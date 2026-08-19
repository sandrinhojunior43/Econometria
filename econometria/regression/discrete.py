"""Modelos de escolha discreta: Logit e Probit, com efeitos marginais e razoes de chance."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..utils.formatting import significance_stars
from ..utils.validation import require_columns, require_min_rows, require_numeric


@dataclass
class DiscreteChoiceResult:
    modelo: object
    tipo: str  # "logit" ou "probit"
    variavel_dependente: str
    variaveis_independentes: list[str]
    tabela_coeficientes: pd.DataFrame
    efeitos_marginais: pd.DataFrame | None
    pseudo_r2: float
    log_likelihood: float
    aic: float
    bic: float
    n_observacoes: int
    acuracia: float
    matriz_confusao: pd.DataFrame

    def resumo_texto(self) -> str:
        sig = self.tabela_coeficientes[self.tabela_coeficientes["p_valor"] < 0.05].index.tolist()
        sig = [v for v in sig if v not in ("const",)]
        linhas = [
            f"Modelo {self.tipo.capitalize()} para **{self.variavel_dependente}** "
            f"(n = {self.n_observacoes}). Pseudo-R² (McFadden) = {self.pseudo_r2:.4f}.",
            f"Acuracia de classificacao (corte 0,5): {self.acuracia * 100:.1f}%.",
        ]
        if sig:
            linhas.append(f"Variaveis significativas a 5%: {', '.join(sig)}.")
        else:
            linhas.append("Nenhuma variavel e estatisticamente significativa a 5%.")
        return "\n\n".join(linhas)


def _fit_discrete(df: pd.DataFrame, y: str, x: list[str], tipo: str) -> DiscreteChoiceResult:
    require_columns(df, [y] + x)
    require_numeric(df, [y] + x)
    dados = df[[y] + x].dropna()
    require_min_rows(dados, len(x) + 5, f"regressao {tipo}")

    valores_y = sorted(dados[y].unique())
    if len(valores_y) != 2:
        raise ValueError(
            f"A variavel dependente '{y}' precisa ser binaria (0/1). "
            f"Valores encontrados: {valores_y}."
        )
    if set(valores_y) != {0, 1}:
        dados = dados.copy()
        dados[y] = (dados[y] == valores_y[1]).astype(int)

    X = sm.add_constant(dados[x])
    modelo_cls = sm.Logit if tipo == "logit" else sm.Probit
    modelo = modelo_cls(dados[y], X)
    fit = modelo.fit(disp=0)

    conf = fit.conf_int()
    conf.columns = ["ic_inferior_95", "ic_superior_95"]
    tabela = pd.DataFrame(
        {
            "coeficiente": fit.params,
            "erro_padrao": fit.bse,
            "estatistica_z": fit.tvalues,
            "p_valor": fit.pvalues,
        }
    ).join(conf)
    tabela["significancia"] = tabela["p_valor"].apply(significance_stars)
    if tipo == "logit":
        tabela["razao_chance_odds_ratio"] = np.exp(tabela["coeficiente"])

    try:
        margeff = fit.get_margeff(at="overall")
        efeitos = pd.DataFrame(
            {"efeito_marginal": margeff.margeff, "erro_padrao": margeff.margeff_se, "p_valor": margeff.pvalues},
            index=x,
        )
    except Exception:
        efeitos = None

    preditos_prob = fit.predict(X)
    preditos_classe = (preditos_prob >= 0.5).astype(int)
    y_real = dados[y]
    acuracia = float((preditos_classe == y_real).mean())
    matriz_confusao = pd.crosstab(
        y_real.rename("real"), preditos_classe.rename("previsto"), dropna=False
    )

    return DiscreteChoiceResult(
        modelo=fit,
        tipo=tipo,
        variavel_dependente=y,
        variaveis_independentes=x,
        tabela_coeficientes=tabela,
        efeitos_marginais=efeitos,
        pseudo_r2=float(fit.prsquared),
        log_likelihood=float(fit.llf),
        aic=float(fit.aic),
        bic=float(fit.bic),
        n_observacoes=int(fit.nobs),
        acuracia=acuracia,
        matriz_confusao=matriz_confusao,
    )


def logit_regression(df: pd.DataFrame, y: str, x: list[str]) -> DiscreteChoiceResult:
    """Regressao Logit para variavel dependente binaria."""
    return _fit_discrete(df, y, x, "logit")


def probit_regression(df: pd.DataFrame, y: str, x: list[str]) -> DiscreteChoiceResult:
    """Regressao Probit para variavel dependente binaria."""
    return _fit_discrete(df, y, x, "probit")
