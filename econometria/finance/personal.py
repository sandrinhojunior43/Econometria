"""Financas pessoais: juros compostos, financiamentos (Price/SAC), aposentadoria e orcamento."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils.formatting import fmt_currency, fmt_percent


def valor_futuro(
    valor_presente: float,
    taxa_periodo: float,
    n_periodos: int,
    aporte_periodico: float = 0.0,
    aporte_no_inicio: bool = False,
) -> float:
    """Valor futuro de um montante inicial + aportes periodicos, a juros compostos.

    Parameters
    ----------
    taxa_periodo: taxa de juros por periodo, em decimal (ex.: 0,01 para 1% ao mes).
    n_periodos: numero de periodos (mesma unidade da taxa).
    aporte_periodico: valor investido a cada periodo (0 se nao houver aportes).
    aporte_no_inicio: True para aportes no inicio do periodo (antecipados), False para o fim (postecipados).
    """
    montante_inicial = valor_presente * (1 + taxa_periodo) ** n_periodos
    if aporte_periodico == 0 or n_periodos == 0:
        return float(montante_inicial)
    if taxa_periodo == 0:
        montante_aportes = aporte_periodico * n_periodos
    else:
        fator = ((1 + taxa_periodo) ** n_periodos - 1) / taxa_periodo
        montante_aportes = aporte_periodico * fator * ((1 + taxa_periodo) if aporte_no_inicio else 1)
    return float(montante_inicial + montante_aportes)


def valor_presente(valor_futuro: float, taxa_periodo: float, n_periodos: int) -> float:
    """Valor presente de um montante futuro, descontado a uma taxa periodica."""
    return float(valor_futuro / (1 + taxa_periodo) ** n_periodos)


def taxa_equivalente(taxa: float, freq_atual: float, freq_nova: float) -> float:
    """Converte uma taxa de juros compostos entre periodicidades.

    `freq_atual` e `freq_nova` sao o numero de periodos por ano de cada taxa
    (1 = anual, 12 = mensal, 252 = diaria util, etc.).

    Exemplo: `taxa_equivalente(0.08, 1, 12)` converte 8% a.a. para a taxa mensal
    equivalente (~0,643% a.m.). `taxa_equivalente(0.00643, 12, 1)` faz o caminho inverso.
    """
    return float((1 + taxa) ** (freq_atual / freq_nova) - 1)


def tabela_price(valor_financiado: float, taxa_periodo: float, n_parcelas: int) -> pd.DataFrame:
    """Sistema Price (Tabela Price): parcelas fixas, com juros decrescentes e amortizacao crescente."""
    if taxa_periodo == 0:
        parcela = valor_financiado / n_parcelas
    else:
        parcela = valor_financiado * (taxa_periodo * (1 + taxa_periodo) ** n_parcelas) / ((1 + taxa_periodo) ** n_parcelas - 1)

    saldo = valor_financiado
    linhas = []
    for periodo in range(1, n_parcelas + 1):
        juros = saldo * taxa_periodo
        amortizacao = parcela - juros
        saldo = max(0.0, saldo - amortizacao)
        linhas.append(
            {"parcela_num": periodo, "prestacao": parcela, "juros": juros, "amortizacao": amortizacao, "saldo_devedor": saldo}
        )
    return pd.DataFrame(linhas).set_index("parcela_num")


def tabela_sac(valor_financiado: float, taxa_periodo: float, n_parcelas: int) -> pd.DataFrame:
    """Sistema de Amortizacao Constante (SAC): amortizacao fixa, parcelas e juros decrescentes."""
    amortizacao = valor_financiado / n_parcelas
    saldo = valor_financiado
    linhas = []
    for periodo in range(1, n_parcelas + 1):
        juros = saldo * taxa_periodo
        prestacao = amortizacao + juros
        saldo = max(0.0, saldo - amortizacao)
        linhas.append(
            {"parcela_num": periodo, "prestacao": prestacao, "juros": juros, "amortizacao": amortizacao, "saldo_devedor": saldo}
        )
    return pd.DataFrame(linhas).set_index("parcela_num")


@dataclass
class SimulacaoAposentadoria:
    patrimonio_na_aposentadoria: float
    renda_mensal_sustentavel: float
    total_aportado: float
    total_juros_ganhos: float
    anos_ate_aposentadoria: float
    evolucao: pd.DataFrame

    def resumo_texto(self) -> str:
        return (
            f"Em {self.anos_ate_aposentadoria:.0f} anos, com os aportes informados, o patrimonio "
            f"projetado na aposentadoria e de **{fmt_currency(self.patrimonio_na_aposentadoria)}**. "
            f"Isso pode sustentar uma renda mensal aproximada de **{fmt_currency(self.renda_mensal_sustentavel)}** "
            f"(mantendo o principal, via taxa de retirada informada). "
            f"Do total acumulado, {fmt_currency(self.total_aportado)} vieram de aportes e "
            f"{fmt_currency(self.total_juros_ganhos)} de juros compostos."
        )


def simular_aposentadoria(
    idade_atual: int,
    idade_aposentadoria: int,
    patrimonio_atual: float,
    aporte_mensal: float,
    taxa_retorno_anual: float,
    taxa_retirada_anual: float = 0.04,
) -> SimulacaoAposentadoria:
    """Projeta a fase de acumulacao (ate a aposentadoria) e estima a renda mensal sustentavel
    apos aposentar, usando a regra de taxa de retirada segura (`taxa_retirada_anual`, ex.: 4% a.a.).
    """
    if idade_aposentadoria <= idade_atual:
        raise ValueError("A idade de aposentadoria deve ser maior que a idade atual.")

    n_meses = (idade_aposentadoria - idade_atual) * 12
    taxa_mensal = taxa_equivalente(taxa_retorno_anual, 1, 12)

    saldo = patrimonio_atual
    linhas = [{"mes": 0, "idade": idade_atual, "patrimonio": saldo}]
    for mes in range(1, n_meses + 1):
        saldo = saldo * (1 + taxa_mensal) + aporte_mensal
        idade_fracionaria = idade_atual + mes / 12
        linhas.append({"mes": mes, "idade": idade_fracionaria, "patrimonio": saldo})

    evolucao = pd.DataFrame(linhas)
    total_aportado = patrimonio_atual + aporte_mensal * n_meses
    renda_mensal = saldo * taxa_retirada_anual / 12

    return SimulacaoAposentadoria(
        patrimonio_na_aposentadoria=float(saldo),
        renda_mensal_sustentavel=float(renda_mensal),
        total_aportado=float(total_aportado),
        total_juros_ganhos=float(saldo - total_aportado),
        anos_ate_aposentadoria=idade_aposentadoria - idade_atual,
        evolucao=evolucao,
    )


@dataclass
class AnaliseOrcamento:
    receita_total: float
    despesa_total: float
    saldo: float
    percentual_comprometido: float
    despesas_por_categoria: pd.Series
    alertas: list[str] = field(default_factory=list)

    def resumo_texto(self) -> str:
        linhas = [
            f"Receita total: {fmt_currency(self.receita_total)} | Despesas totais: {fmt_currency(self.despesa_total)} "
            f"| Saldo: {fmt_currency(self.saldo)}.",
            f"Percentual da receita comprometido com despesas: {fmt_percent(self.percentual_comprometido / 100)}.",
        ]
        if self.alertas:
            linhas.append("**Alertas:** " + " ".join(self.alertas))
        return "\n\n".join(linhas)


def analisar_orcamento(receita_mensal: float, despesas: dict[str, float]) -> AnaliseOrcamento:
    """Analisa um orcamento pessoal/familiar simples: saldo, comprometimento da renda e
    maiores categorias de gasto, com alertas praticos (regra geral: nao comprometer >70% da renda,
    manter poupanca/investimento >=10-20%).
    """
    despesa_total = sum(despesas.values())
    saldo = receita_mensal - despesa_total
    percentual = (despesa_total / receita_mensal * 100) if receita_mensal else float("inf")
    serie_despesas = pd.Series(despesas).sort_values(ascending=False)

    alertas = []
    if saldo < 0:
        alertas.append("Despesas superam a receita - orcamento no vermelho, requer acao imediata.")
    elif percentual > 90:
        alertas.append("Mais de 90% da renda esta comprometida - pouca margem para imprevistos.")
    elif percentual > 70:
        alertas.append("Mais de 70% da renda esta comprometida - considere reduzir gastos discricionarios.")

    if not serie_despesas.empty:
        maior_categoria = serie_despesas.index[0]
        peso_maior = serie_despesas.iloc[0] / despesa_total * 100 if despesa_total else 0
        if peso_maior > 40:
            alertas.append(f"'{maior_categoria}' concentra {peso_maior:.0f}% das despesas - avalie renegociar ou reduzir.")

    return AnaliseOrcamento(
        receita_total=float(receita_mensal),
        despesa_total=float(despesa_total),
        saldo=float(saldo),
        percentual_comprometido=float(percentual),
        despesas_por_categoria=serie_despesas,
        alertas=alertas,
    )
