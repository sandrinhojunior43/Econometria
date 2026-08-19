import plotly.graph_objects as go
import streamlit as st

import _bootstrap  # noqa: F401
from components import botoes_download_relatorio, estilizar, kpis
from state import sidebar_dados
from theme import aplicar_tema, cabecalho, caixa, rodape

from econometria.finance import (
    analisar_viabilidade,
    indicadores_financeiros,
    payback_descontado,
    payback_simples,
    ponto_equilibrio,
    taxa_interna_retorno,
    valor_presente_liquido,
)
from econometria.reports import ReportBuilder
from econometria.utils import fmt_currency, fmt_percent

st.set_page_config(page_title="Financas Empresariais - Econometria", page_icon="🏢", layout="wide")
aplicar_tema()
sidebar_dados()
cabecalho(
    "🏢 Financas Empresariais",
    "Avalie a viabilidade de projetos (VPL, TIR, payback), calcule o ponto de equilibrio e "
    "acompanhe os principais indicadores financeiros da empresa.",
)

aba_viab, aba_equilibrio, aba_indic = st.tabs(["📊 Viabilidade de Projetos", "⚖️ Ponto de Equilibrio", "🧮 Indicadores Financeiros"])

# -------------------------------------------------------------- viabilidade
with aba_viab:
    st.caption("Informe os fluxos de caixa do projeto, um por periodo. O primeiro fluxo e normalmente o investimento inicial (negativo).")
    n_periodos = st.number_input("Numero de periodos (incluindo o periodo 0)", min_value=2, max_value=30, value=6)
    tma = st.number_input("Taxa Minima de Atratividade - TMA (% ao periodo)", min_value=0.0, value=10.0, step=0.5)

    cols = st.columns(min(6, int(n_periodos)))
    fluxos = []
    for i in range(int(n_periodos)):
        valor_padrao = -10000.0 if i == 0 else 3000.0
        fluxos.append(cols[i % len(cols)].number_input(f"Fluxo t={i}", value=valor_padrao, step=500.0, key=f"fluxo_{i}"))

    if st.button("Analisar viabilidade", type="primary"):
        try:
            analise = analisar_viabilidade(fluxos, tma / 100)
            payback_desc = payback_descontado(fluxos, tma / 100)
            kpis(
                [
                    ("VPL", fmt_currency(analise.vpl), ""),
                    ("TIR", fmt_percent(analise.tir) if analise.tir is not None else "n/d", ""),
                    ("Payback simples", f"{analise.payback:.1f} periodos" if analise.payback is not None else "n/d", ""),
                    ("Payback descontado", f"{payback_desc:.1f} periodos" if payback_desc is not None else "n/d", ""),
                ]
            )
            caixa(analise.resumo_texto().replace("\n\n", "<br><br>"), "sucesso" if analise.viavel else "alerta")

            acumulado = []
            saldo = 0
            for f in fluxos:
                saldo += f
                acumulado.append(saldo)
            fig = go.Figure(go.Bar(x=list(range(len(fluxos))), y=fluxos, name="Fluxo do periodo"))
            fig.add_trace(go.Scatter(x=list(range(len(fluxos))), y=acumulado, name="Fluxo acumulado", mode="lines+markers"))
            st.plotly_chart(estilizar(fig, "Fluxos de caixa e acumulado"), width="stretch")
            st.session_state["emp_viabilidade"] = (analise, fluxos, tma)
        except Exception as exc:
            st.error(str(exc))

# -------------------------------------------------------------- ponto de equilibrio
with aba_equilibrio:
    c1, c2, c3 = st.columns(3)
    custos_fixos = c1.number_input("Custos fixos mensais (R$)", min_value=0.0, value=10000.0, step=500.0)
    preco_venda = c2.number_input("Preco de venda unitario (R$)", min_value=0.01, value=50.0, step=1.0)
    custo_variavel = c3.number_input("Custo variavel unitario (R$)", min_value=0.0, value=30.0, step=1.0)

    if st.button("Calcular ponto de equilibrio"):
        try:
            pe = ponto_equilibrio(custos_fixos, preco_venda, custo_variavel)
            kpis(
                [
                    ("Quantidade de equilibrio", f"{pe.quantidade_equilibrio:.1f} un.", ""),
                    ("Receita de equilibrio", fmt_currency(pe.receita_equilibrio), ""),
                    ("Margem de contribuicao", f"{pe.margem_contribuicao_percentual:.1f}%", ""),
                ]
            )
            caixa(pe.resumo_texto(), "info")

            quantidades = list(range(0, int(pe.quantidade_equilibrio * 2) + 10))
            receita = [q * preco_venda for q in quantidades]
            custo_total = [custos_fixos + q * custo_variavel for q in quantidades]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=quantidades, y=receita, name="Receita"))
            fig.add_trace(go.Scatter(x=quantidades, y=custo_total, name="Custo total"))
            fig.add_vline(x=pe.quantidade_equilibrio, line_dash="dash", line_color="#d93025")
            st.plotly_chart(estilizar(fig, "Receita x Custo total"), width="stretch")
            st.session_state["emp_equilibrio"] = pe
        except Exception as exc:
            st.error(str(exc))

# -------------------------------------------------------------- indicadores
with aba_indic:
    st.caption("Informe os dados do balanco/DRE para calcular os indicadores classicos.")
    c1, c2, c3 = st.columns(3)
    receita_liq = c1.number_input("Receita liquida (R$)", min_value=0.0, value=1000000.0, step=10000.0)
    lucro_bruto = c2.number_input("Lucro bruto (R$)", min_value=0.0, value=400000.0, step=10000.0)
    lucro_liq = c3.number_input("Lucro liquido (R$)", value=120000.0, step=10000.0)
    c4, c5 = st.columns(2)
    ativo_total = c4.number_input("Ativo total (R$)", min_value=0.0, value=800000.0, step=10000.0)
    patrimonio_liq = c5.number_input("Patrimonio liquido (R$)", min_value=0.0, value=350000.0, step=10000.0)
    c6, c7, c8 = st.columns(3)
    ativo_circ = c6.number_input("Ativo circulante (R$)", min_value=0.0, value=300000.0, step=10000.0)
    passivo_circ = c7.number_input("Passivo circulante (R$)", min_value=0.0, value=200000.0, step=10000.0)
    passivo_total = c8.number_input("Passivo total (R$)", min_value=0.0, value=450000.0, step=10000.0)

    if st.button("Calcular indicadores"):
        tabela = indicadores_financeiros(
            receita_liq, lucro_bruto, lucro_liq, ativo_total, patrimonio_liq, ativo_circ, passivo_circ, passivo_total
        )
        st.dataframe(tabela.style.format("{:.2%}"), width="stretch")
        fig = go.Figure(go.Bar(x=tabela.index, y=tabela["valor"]))
        st.plotly_chart(estilizar(fig, "Indicadores financeiros"), width="stretch")
        st.session_state["emp_indicadores"] = tabela

st.divider()
if st.button("📄 Gerar relatorio consolidado de financas empresariais"):
    builder = ReportBuilder("Relatorio de Financas Empresariais", "Resumo das analises realizadas nesta sessao")
    algo = False
    if "emp_viabilidade" in st.session_state:
        analise, fluxos, tma = st.session_state["emp_viabilidade"]
        builder.add_card("VPL", fmt_currency(analise.vpl))
        builder.add_card("TIR", fmt_percent(analise.tir) if analise.tir is not None else "n/d")
        builder.add_section("Analise de viabilidade", analise.resumo_texto())
        algo = True
    if "emp_equilibrio" in st.session_state:
        pe = st.session_state["emp_equilibrio"]
        builder.add_section("Ponto de equilibrio", pe.resumo_texto())
        algo = True
    if "emp_indicadores" in st.session_state:
        builder.add_section("Indicadores financeiros", "", tabelas=[("Indicadores", st.session_state["emp_indicadores"])])
        algo = True
    if not algo:
        st.warning("Rode ao menos uma analise acima antes de gerar o relatorio.")
    else:
        botoes_download_relatorio(builder, "financas_empresariais")

rodape()
