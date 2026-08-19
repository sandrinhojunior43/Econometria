"""Validacoes compartilhadas por todos os modulos de calculo.

O objetivo destas funcoes e falhar cedo e com mensagens em portugues,
claras o suficiente para aparecer diretamente na interface (Streamlit)
sem precisar ser reescritas.
"""
from __future__ import annotations

import pandas as pd


class ValidationError(ValueError):
    """Erro de validacao de dados/parametros, pensado para ser exibido ao usuario final."""


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    faltantes = [c for c in columns if c not in df.columns]
    if faltantes:
        raise ValidationError(
            f"As seguintes colunas nao foram encontradas nos dados: {', '.join(faltantes)}. "
            f"Colunas disponiveis: {', '.join(map(str, df.columns))}."
        )


def require_min_rows(df: pd.DataFrame, n: int, contexto: str = "esta analise") -> None:
    linhas_validas = len(df.dropna())
    if linhas_validas < n:
        raise ValidationError(
            f"Dados insuficientes para {contexto}: sao necessarias pelo menos {n} observacoes "
            f"completas, mas ha apenas {linhas_validas} apos remover valores ausentes."
        )


def require_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    nao_numericas = [c for c in columns if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    if nao_numericas:
        raise ValidationError(
            "As seguintes colunas precisam ser numericas para esta analise: "
            f"{', '.join(nao_numericas)}. Verifique separadores decimais, texto misturado "
            "com numeros ou celulas vazias."
        )
