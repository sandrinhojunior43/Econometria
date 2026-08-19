"""Estatistica descritiva: medidas de posicao, dispersao, forma, correlacao e normalidade."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.validation import require_columns, require_min_rows


@dataclass
class DescriptiveResult:
    tabela: pd.DataFrame
    n_observacoes: int
    colunas: list[str]

    def interpretacao(self) -> list[str]:
        bullets = []
        for col in self.colunas:
            row = self.tabela.loc[col]
            cv = row["desvio_padrao"] / row["media"] * 100 if row["media"] not in (0, None) and not np.isnan(row["media"]) else np.nan
            assimetria = row.get("assimetria", np.nan)
            forma = "aproximadamente simetrica"
            if not np.isnan(assimetria):
                if assimetria > 0.5:
                    forma = "com cauda a direita (assimetria positiva)"
                elif assimetria < -0.5:
                    forma = "com cauda a esquerda (assimetria negativa)"
            texto = f"**{col}**: media {row['media']:.2f}, mediana {row['mediana']:.2f}, distribuicao {forma}."
            if not np.isnan(cv):
                nivel = "alta" if cv > 50 else ("moderada" if cv > 15 else "baixa")
                texto += f" Dispersao relativa (CV) {nivel} ({cv:.1f}%)."
            bullets.append(texto)
        return bullets


def describe(df: pd.DataFrame, colunas: list[str] | None = None) -> DescriptiveResult:
    """Estatistica descritiva completa: posicao, dispersao, forma e quartis."""
    if colunas is None:
        colunas = df.select_dtypes(include=np.number).columns.tolist()
    require_columns(df, colunas)
    require_min_rows(df, 2, "estatistica descritiva")

    linhas = {}
    for col in colunas:
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        if serie.empty:
            continue
        linhas[col] = {
            "n": int(serie.shape[0]),
            "media": serie.mean(),
            "mediana": serie.median(),
            "moda": serie.mode().iloc[0] if not serie.mode().empty else np.nan,
            "desvio_padrao": serie.std(ddof=1),
            "variancia": serie.var(ddof=1),
            "erro_padrao": serie.sem(),
            "minimo": serie.min(),
            "maximo": serie.max(),
            "amplitude": serie.max() - serie.min(),
            "q1": serie.quantile(0.25),
            "q3": serie.quantile(0.75),
            "iqr": serie.quantile(0.75) - serie.quantile(0.25),
            "assimetria": stats.skew(serie, bias=False) if len(serie) > 2 else np.nan,
            "curtose": stats.kurtosis(serie, bias=False) if len(serie) > 3 else np.nan,
            "coef_variacao_%": (serie.std(ddof=1) / serie.mean() * 100) if serie.mean() != 0 else np.nan,
        }
    tabela = pd.DataFrame(linhas).T
    return DescriptiveResult(tabela=tabela, n_observacoes=len(df), colunas=list(linhas.keys()))


def correlation_matrix(df: pd.DataFrame, colunas: list[str] | None = None, metodo: str = "pearson") -> pd.DataFrame:
    """Matriz de correlacao (pearson, spearman ou kendall)."""
    if colunas is None:
        colunas = df.select_dtypes(include=np.number).columns.tolist()
    require_columns(df, colunas)
    if metodo not in ("pearson", "spearman", "kendall"):
        raise ValueError("metodo deve ser 'pearson', 'spearman' ou 'kendall'")
    return df[colunas].apply(pd.to_numeric, errors="coerce").corr(method=metodo)


def normality_test(serie: pd.Series) -> dict:
    """Testa normalidade via Jarque-Bera e Shapiro-Wilk (para amostras menores)."""
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    if len(valores) < 8:
        raise ValueError("Sao necessarias ao menos 8 observacoes para o teste de normalidade.")

    jb_stat, jb_p = stats.jarque_bera(valores)
    resultado = {
        "jarque_bera_estatistica": jb_stat,
        "jarque_bera_p_valor": jb_p,
        "normal_jarque_bera_5%": bool(jb_p > 0.05),
    }
    if len(valores) <= 5000:
        sw_stat, sw_p = stats.shapiro(valores)
        resultado.update(
            {
                "shapiro_estatistica": sw_stat,
                "shapiro_p_valor": sw_p,
                "normal_shapiro_5%": bool(sw_p > 0.05),
            }
        )
    return resultado
