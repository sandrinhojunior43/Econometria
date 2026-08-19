"""Financas empresariais: VPL, TIR, payback, ponto de equilibrio e indicadores financeiros."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy_financial as npf
import pandas as pd

from ..utils.formatting import fmt_currency, fmt_percent


def valor_presente_liquido(fluxos_caixa: list[float], taxa_desconto: float) -> float:
    """Valor Presente Liquido (VPL/NPV). `fluxos_caixa[0]` deve ser o investimento inicial (negativo)."""
    return float(npf.npv(taxa_desconto, fluxos_caixa))


def taxa_interna_retorno(fluxos_caixa: list[float]) -> float:
    """Taxa Interna de Retorno (TIR/IRR). Requer ao menos uma mudanca de sinal nos fluxos."""
    tir = npf.irr(fluxos_caixa)
    if tir is None or np.isnan(tir):
        raise ValueError(
            "Nao foi possivel calcular a TIR para esses fluxos de caixa. "
            "Verifique se ha ao menos uma saida (negativa) e uma entrada (positiva) de caixa."
        )
    return float(tir)


def payback_simples(fluxos_caixa: list[float]) -> float | None:
    """Periodo de payback simples (em numero de periodos, com fracao do ultimo periodo).

    `fluxos_caixa[0]` e o investimento inicial (negativo); os demais sao as entradas liquidas.
    Retorna None se o investimento nunca e recuperado dentro do horizonte informado.
    """
    acumulado = fluxos_caixa[0]
    for periodo in range(1, len(fluxos_caixa)):
        acumulado_anterior = acumulado
        acumulado += fluxos_caixa[periodo]
        if acumulado >= 0:
            fracao = -acumulado_anterior / fluxos_caixa[periodo] if fluxos_caixa[periodo] != 0 else 0
            return (periodo - 1) + fracao
    return None


def payback_descontado(fluxos_caixa: list[float], taxa_desconto: float) -> float | None:
    """Payback descontado: mesma logica do payback simples, mas sobre fluxos trazidos a valor presente."""
    fluxos_descontados = [fc / (1 + taxa_desconto) ** i for i, fc in enumerate(fluxos_caixa)]
    return payback_simples(fluxos_descontados)


@dataclass
class PontoEquilibrio:
    quantidade_equilibrio: float
    receita_equilibrio: float
    margem_contribuicao_unitaria: float
    margem_contribuicao_percentual: float

    def resumo_texto(self) -> str:
        return (
            f"Ponto de equilibrio: **{self.quantidade_equilibrio:.1f} unidades** "
            f"(receita de {fmt_currency(self.receita_equilibrio)}). "
            f"Cada unidade vendida contribui com {fmt_currency(self.margem_contribuicao_unitaria)} "
            f"({fmt_percent(self.margem_contribuicao_percentual / 100)} do preco) para cobrir custos fixos e gerar lucro."
        )


def ponto_equilibrio(custos_fixos: float, preco_venda_unitario: float, custo_variavel_unitario: float) -> PontoEquilibrio:
    """Ponto de equilibrio (break-even) em unidades e em receita."""
    margem_unitaria = preco_venda_unitario - custo_variavel_unitario
    if margem_unitaria <= 0:
        raise ValueError(
            "O preco de venda precisa ser maior que o custo variavel unitario para existir ponto de equilibrio."
        )
    qtd_equilibrio = custos_fixos / margem_unitaria
    receita_equilibrio = qtd_equilibrio * preco_venda_unitario
    margem_pct = margem_unitaria / preco_venda_unitario * 100
    return PontoEquilibrio(
        quantidade_equilibrio=float(qtd_equilibrio),
        receita_equilibrio=float(receita_equilibrio),
        margem_contribuicao_unitaria=float(margem_unitaria),
        margem_contribuicao_percentual=float(margem_pct),
    )


def indicadores_financeiros(
    receita_liquida: float,
    lucro_bruto: float,
    lucro_liquido: float,
    ativo_total: float,
    patrimonio_liquido: float,
    ativo_circulante: float,
    passivo_circulante: float,
    passivo_total: float,
) -> pd.DataFrame:
    """Bateria de indicadores financeiros classicos: margens, rentabilidade, liquidez e endividamento."""
    indicadores = {
        "Margem bruta": lucro_bruto / receita_liquida if receita_liquida else np.nan,
        "Margem liquida": lucro_liquido / receita_liquida if receita_liquida else np.nan,
        "ROA (retorno sobre ativos)": lucro_liquido / ativo_total if ativo_total else np.nan,
        "ROE (retorno sobre patrimonio)": lucro_liquido / patrimonio_liquido if patrimonio_liquido else np.nan,
        "Liquidez corrente": ativo_circulante / passivo_circulante if passivo_circulante else np.nan,
        "Endividamento geral": passivo_total / ativo_total if ativo_total else np.nan,
        "Giro do ativo": receita_liquida / ativo_total if ativo_total else np.nan,
    }
    tabela = pd.DataFrame.from_dict(indicadores, orient="index", columns=["valor"])
    return tabela


@dataclass
class AnaliseViabilidade:
    vpl: float
    tir: float | None
    payback: float | None
    taxa_minima_atratividade: float
    viavel: bool
    justificativa: str

    def resumo_texto(self) -> str:
        veredito = "VIAVEL" if self.viavel else "NAO VIAVEL"
        tir_txt = fmt_percent(self.tir) if self.tir is not None else "nao calculavel"
        payback_txt = f"{self.payback:.1f} periodos" if self.payback is not None else "nao recupera o investimento no horizonte"
        return (
            f"**Veredito: {veredito}** ao custo de capital de {fmt_percent(self.taxa_minima_atratividade)}.\n\n"
            f"VPL = {fmt_currency(self.vpl)} | TIR = {tir_txt} | Payback = {payback_txt}.\n\n"
            f"{self.justificativa}"
        )


def analisar_viabilidade(fluxos_caixa: list[float], taxa_minima_atratividade: float) -> AnaliseViabilidade:
    """Combina VPL, TIR e Payback em um veredito unico de viabilidade de um projeto/investimento."""
    vpl = valor_presente_liquido(fluxos_caixa, taxa_minima_atratividade)
    try:
        tir = taxa_interna_retorno(fluxos_caixa)
    except ValueError:
        tir = None
    payback = payback_simples(fluxos_caixa)

    viavel = vpl > 0
    motivos = []
    if vpl > 0:
        motivos.append(f"o VPL e positivo ({fmt_currency(vpl)}), ou seja, o projeto gera valor acima da TMA.")
    else:
        motivos.append(f"o VPL e negativo ou nulo ({fmt_currency(vpl)}), o projeto destroi valor no custo de capital informado.")
    if tir is not None:
        if tir > taxa_minima_atratividade:
            motivos.append(f"a TIR ({fmt_percent(tir)}) supera a TMA ({fmt_percent(taxa_minima_atratividade)}), reforcando a atratividade.")
        else:
            motivos.append(f"a TIR ({fmt_percent(tir)}) e inferior a TMA ({fmt_percent(taxa_minima_atratividade)}).")
    if payback is not None:
        motivos.append(f"o investimento e recuperado em {payback:.1f} periodos.")
    else:
        motivos.append("o investimento nao se recupera dentro do horizonte de fluxos informado.")

    return AnaliseViabilidade(
        vpl=vpl,
        tir=tir,
        payback=payback,
        taxa_minima_atratividade=taxa_minima_atratividade,
        viavel=viavel,
        justificativa=" ".join(m[0].upper() + m[1:] for m in motivos),
    )
