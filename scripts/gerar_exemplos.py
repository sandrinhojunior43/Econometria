"""Gera os datasets de exemplo usados no botao 'usar dados de exemplo' do app.

Rodar uma vez (`python scripts/gerar_exemplos.py`) sempre que os exemplos precisarem
ser regenerados. Os arquivos ficam versionados em `examples/` para o app funcionar
offline, sem depender de download externo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SAIDA = Path(__file__).resolve().parent.parent / "examples"
SAIDA.mkdir(exist_ok=True)


def gerar_vendas_mensais(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2019-01-01", periods=72, freq="MS")
    tendencia = np.linspace(80, 180, len(datas))
    sazonalidade = 18 * np.sin(2 * np.pi * (datas.month / 12))
    marketing = rng.normal(20, 5, len(datas)).clip(min=2)
    ruido = rng.normal(0, 6, len(datas))
    vendas = tendencia + sazonalidade + 1.8 * marketing + ruido
    temperatura = 22 + 8 * np.sin(2 * np.pi * (datas.month / 12) - 0.4) + rng.normal(0, 1.5, len(datas))
    return pd.DataFrame(
        {
            "data": datas,
            "vendas_mil_r$": vendas.round(2),
            "gasto_marketing_mil_r$": marketing.round(2),
            "temperatura_media_c": temperatura.round(1),
        }
    )


def gerar_imoveis(seed: int = 2, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.normal(90, 35, n).clip(min=25)
    quartos = rng.integers(1, 5, n)
    distancia_centro_km = rng.exponential(6, n).clip(max=40)
    idade_imovel = rng.integers(0, 40, n)
    padrao = rng.choice(["popular", "medio", "alto"], size=n, p=[0.4, 0.4, 0.2])
    premio_padrao = pd.Series(padrao).map({"popular": 0, "medio": 60000, "alto": 180000}).to_numpy()
    preco = (
        50000
        + 3200 * area
        + 15000 * quartos
        - 2800 * distancia_centro_km
        - 900 * idade_imovel
        + premio_padrao
        + rng.normal(0, 25000, n)
    ).clip(min=40000)
    return pd.DataFrame(
        {
            "preco_r$": preco.round(0),
            "area_m2": area.round(1),
            "quartos": quartos,
            "distancia_centro_km": distancia_centro_km.round(2),
            "idade_imovel_anos": idade_imovel,
            "padrao": padrao,
        }
    )


def gerar_clientes_churn(seed: int = 3, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idade = rng.integers(18, 75, n)
    mensalidade = rng.normal(120, 40, n).clip(min=30)
    tempo_contrato_meses = rng.integers(1, 72, n)
    chamados_suporte = rng.poisson(1.2, n)
    score_satisfacao = rng.normal(7, 1.8, n).clip(0, 10)

    logit = (
        -1.5
        + 0.35 * chamados_suporte
        - 0.45 * score_satisfacao
        - 0.02 * tempo_contrato_meses
        + 0.01 * mensalidade
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame(
        {
            "churn": churn,
            "idade": idade,
            "mensalidade_r$": mensalidade.round(2),
            "tempo_contrato_meses": tempo_contrato_meses,
            "chamados_suporte": chamados_suporte,
            "score_satisfacao": score_satisfacao.round(1),
        }
    )


def gerar_painel_empresas(seed: int = 4, n_empresas: int = 25, n_anos: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    linhas = []
    anos = pd.date_range("2014-01-01", periods=n_anos, freq="YS")
    for i in range(n_empresas):
        empresa = f"Empresa_{i+1:02d}"
        efeito_fixo = rng.normal(0, 4)
        tamanho_log = rng.normal(9, 1.2)
        for ano in anos:
            investimento_pct_receita = rng.normal(8, 3, 1)[0].clip(min=0)
            alavancagem = rng.normal(0.4, 0.15, 1)[0].clip(0, 1.2)
            roa = (
                efeito_fixo
                + 0.55 * investimento_pct_receita
                - 3.2 * alavancagem
                + 0.3 * tamanho_log
                + rng.normal(0, 1.5)
            )
            linhas.append(
                {
                    "empresa": empresa,
                    "ano": ano,
                    "roa_%": round(roa, 2),
                    "investimento_pct_receita": round(investimento_pct_receita, 2),
                    "alavancagem": round(alavancagem, 3),
                    "tamanho_log_ativos": round(tamanho_log, 3),
                }
            )
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    gerar_vendas_mensais().to_csv(SAIDA / "vendas_mensais.csv", index=False)
    gerar_imoveis().to_csv(SAIDA / "imoveis.csv", index=False)
    gerar_clientes_churn().to_csv(SAIDA / "clientes_churn.csv", index=False)
    gerar_painel_empresas().to_csv(SAIDA / "painel_empresas.csv", index=False)
    print("Exemplos gerados em", SAIDA)
