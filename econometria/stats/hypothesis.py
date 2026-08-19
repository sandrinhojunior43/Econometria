"""Testes de hipotese classicos: t (uma/duas amostras, pareado), ANOVA, qui-quadrado, proporcao."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.formatting import fmt_pvalue


@dataclass
class HypothesisTestResult:
    nome_teste: str
    estatistica: float
    p_valor: float
    graus_liberdade: float | None = None
    hipotese_nula: str = ""
    alfa: float = 0.05
    detalhes: dict = field(default_factory=dict)

    @property
    def rejeita_h0(self) -> bool:
        return self.p_valor < self.alfa

    def conclusao(self) -> str:
        veredito = "rejeitamos" if self.rejeita_h0 else "nao rejeitamos"
        return (
            f"{self.nome_teste}: estatistica = {self.estatistica:.4f}, "
            f"p-valor = {fmt_pvalue(self.p_valor)}. Ao nivel de significancia de "
            f"{self.alfa * 100:.0f}%, {veredito} H0 ({self.hipotese_nula})."
        )


def t_test_one_sample(serie: pd.Series, mu0: float, alfa: float = 0.05) -> HypothesisTestResult:
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    stat, p = stats.ttest_1samp(valores, popmean=mu0)
    return HypothesisTestResult(
        nome_teste="Teste t - uma amostra",
        estatistica=float(stat),
        p_valor=float(p),
        graus_liberdade=len(valores) - 1,
        hipotese_nula=f"a media populacional e igual a {mu0}",
        alfa=alfa,
        detalhes={"media_amostral": float(valores.mean()), "n": len(valores)},
    )


def t_test_two_sample(a: pd.Series, b: pd.Series, variancias_iguais: bool = False, alfa: float = 0.05) -> HypothesisTestResult:
    va = pd.to_numeric(a, errors="coerce").dropna()
    vb = pd.to_numeric(b, errors="coerce").dropna()
    stat, p = stats.ttest_ind(va, vb, equal_var=variancias_iguais)
    return HypothesisTestResult(
        nome_teste="Teste t - duas amostras independentes",
        estatistica=float(stat),
        p_valor=float(p),
        graus_liberdade=len(va) + len(vb) - 2,
        hipotese_nula="as medias dos dois grupos sao iguais",
        alfa=alfa,
        detalhes={
            "media_a": float(va.mean()),
            "media_b": float(vb.mean()),
            "variancias_iguais_assumidas": variancias_iguais,
        },
    )


def t_test_paired(a: pd.Series, b: pd.Series, alfa: float = 0.05) -> HypothesisTestResult:
    df = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    stat, p = stats.ttest_rel(df["a"], df["b"])
    return HypothesisTestResult(
        nome_teste="Teste t - amostras pareadas",
        estatistica=float(stat),
        p_valor=float(p),
        graus_liberdade=len(df) - 1,
        hipotese_nula="a diferenca media entre os pares e zero",
        alfa=alfa,
        detalhes={"diferenca_media": float((df["a"] - df["b"]).mean())},
    )


def anova_one_way(df: pd.DataFrame, coluna_valor: str, coluna_grupo: str, alfa: float = 0.05) -> HypothesisTestResult:
    dados = df[[coluna_valor, coluna_grupo]].dropna()
    grupos = [g[coluna_valor].to_numpy(dtype=float) for _, g in dados.groupby(coluna_grupo)]
    if len(grupos) < 2:
        raise ValueError("ANOVA requer ao menos 2 grupos distintos.")
    stat, p = stats.f_oneway(*grupos)
    return HypothesisTestResult(
        nome_teste="ANOVA one-way",
        estatistica=float(stat),
        p_valor=float(p),
        graus_liberdade=len(grupos) - 1,
        hipotese_nula=f"as medias de '{coluna_valor}' sao iguais entre os grupos de '{coluna_grupo}'",
        alfa=alfa,
        detalhes={"n_grupos": len(grupos), "medias_por_grupo": dados.groupby(coluna_grupo)[coluna_valor].mean().to_dict()},
    )


def chi_square_independence(df: pd.DataFrame, coluna_a: str, coluna_b: str, alfa: float = 0.05) -> HypothesisTestResult:
    tabela = pd.crosstab(df[coluna_a], df[coluna_b])
    stat, p, dof, esperado = stats.chi2_contingency(tabela)
    return HypothesisTestResult(
        nome_teste="Qui-quadrado de independencia",
        estatistica=float(stat),
        p_valor=float(p),
        graus_liberdade=dof,
        hipotese_nula=f"'{coluna_a}' e '{coluna_b}' sao independentes",
        alfa=alfa,
        detalhes={"tabela_contingencia": tabela, "frequencias_esperadas": pd.DataFrame(esperado, index=tabela.index, columns=tabela.columns)},
    )


def proportion_test(sucessos: int, n: int, p0: float, alfa: float = 0.05) -> HypothesisTestResult:
    """Teste z para uma proporcao."""
    if not (0 < p0 < 1):
        raise ValueError("p0 deve estar entre 0 e 1.")
    p_hat = sucessos / n
    erro_padrao = np.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / erro_padrao
    p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
    return HypothesisTestResult(
        nome_teste="Teste z - uma proporcao",
        estatistica=float(z),
        p_valor=float(p_valor),
        hipotese_nula=f"a proporcao populacional e igual a {p0}",
        alfa=alfa,
        detalhes={"proporcao_amostral": p_hat, "n": n},
    )
