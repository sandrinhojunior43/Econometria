import plotly.graph_objects as go
import streamlit as st

import _bootstrap  # noqa: F401
from components import botoes_download_relatorio, estilizar, kpis
from state import sidebar_dados
from theme import aplicar_tema, cabecalho, caixa, rodape

from econometria.finance import (
    analisar_orcamento,
    simular_aposentadoria,
    tabela_price,
    tabela_sac,
    taxa_equivalente,
    valor_futuro,
)
from econometria.reports import ReportBuilder
from econometria.utils import fmt_currency

st.set_page_config(page_title="Financas Pessoais - Econometria", page_icon="💰", layout="wide")
aplicar_tema()
sidebar_dados()
cabecalho(
    "💰 Financas Pessoais",
    "Simule juros compostos, compare sistemas de financiamento (Price x SAC), projete sua "
    "aposentadoria e analise seu orcamento mensal.",
)

aba_juros, aba_financ, aba_aposent, aba_orc = st.tabs(
    ["📈 Juros Compostos", "🏠 Financiamento (Price x SAC)", "🏖️ Aposentadoria", "🧾 Orcamento"]
)

# -------------------------------------------------------------- juros compostos
with aba_juros:
    c1, c2, c3 = st.columns(3)
    valor_inicial = c1.number_input("Valor inicial (R$)", min_value=0.0, value=1000.0, step=100.0)
    taxa_pct = c2.number_input("Taxa de juros por periodo (%)", min_value=0.0, value=1.0, step=0.1)
    n_periodos = c3.number_input("Numero de periodos", min_value=1, value=24, step=1)
    aporte = st.number_input("Aporte periodico (R$, opcional)", min_value=0.0, value=0.0, step=50.0)

    if st.button("Calcular", key="btn_juros"):
        vf = valor_futuro(valor_inicial, taxa_pct / 100, int(n_periodos), aporte_periodico=aporte)
        total_aportado = valor_inicial + aporte * n_periodos
        kpis(
            [
                ("Valor futuro", fmt_currency(vf), ""),
                ("Total aportado", fmt_currency(total_aportado), ""),
                ("Juros ganhos", fmt_currency(vf - total_aportado), ""),
            ]
        )
        evolucao = [valor_futuro(valor_inicial, taxa_pct / 100, t, aporte_periodico=aporte) for t in range(int(n_periodos) + 1)]
        fig = go.Figure(go.Scatter(y=evolucao, mode="lines"))
        st.plotly_chart(estilizar(fig, "Evolucao do patrimonio"), width="stretch")

    with st.expander("🔁 Converter taxa de juros entre periodicidades"):
        cc1, cc2, cc3 = st.columns(3)
        taxa_base = cc1.number_input("Taxa (%)", value=12.0, key="taxa_conv")
        freq_atual = cc2.selectbox("Periodicidade da taxa informada", ["Anual", "Mensal", "Diaria (252 dias uteis)"])
        freq_nova = cc3.selectbox("Converter para", ["Anual", "Mensal", "Diaria (252 dias uteis)"], index=1)
        mapa_freq = {"Anual": 1, "Mensal": 12, "Diaria (252 dias uteis)": 252}
        if st.button("Converter"):
            resultado_taxa = taxa_equivalente(taxa_base / 100, mapa_freq[freq_atual], mapa_freq[freq_nova])
            st.success(f"Taxa equivalente: **{resultado_taxa*100:.4f}%** ({freq_nova.lower()})")

# -------------------------------------------------------------- financiamento
with aba_financ:
    c1, c2, c3 = st.columns(3)
    valor_financiado = c1.number_input("Valor financiado (R$)", min_value=0.0, value=200000.0, step=1000.0)
    taxa_financ = c2.number_input("Taxa de juros mensal (%)", min_value=0.0, value=0.8, step=0.05)
    n_parcelas = c3.number_input("Numero de parcelas", min_value=1, value=360, step=1)

    if st.button("Simular financiamento"):
        price = tabela_price(valor_financiado, taxa_financ / 100, int(n_parcelas))
        sac = tabela_sac(valor_financiado, taxa_financ / 100, int(n_parcelas))
        kpis(
            [
                ("1ª parcela - Price", fmt_currency(price['prestacao'].iloc[0]), ""),
                ("1ª parcela - SAC", fmt_currency(sac['prestacao'].iloc[0]), ""),
                ("Total pago - Price", fmt_currency(price['prestacao'].sum()), ""),
                ("Total pago - SAC", fmt_currency(sac['prestacao'].sum()), ""),
            ]
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=price["prestacao"], name="Price (parcela fixa)"))
        fig.add_trace(go.Scatter(y=sac["prestacao"], name="SAC (parcela decrescente)"))
        st.plotly_chart(estilizar(fig, "Evolucao das prestacoes"), width="stretch")
        caixa(
            "No <b>Price</b>, a parcela e fixa mas voce paga mais juros no total. No <b>SAC</b>, a parcela comeca "
            "maior e cai ao longo do tempo, com menos juros pagos no total. Se o orcamento mensal for apertado "
            "no inicio, o Price facilita o fluxo de caixa; se o objetivo e pagar menos juros, o SAC costuma ser melhor.",
            "info",
        )
        st.session_state["fin_price"] = price
        st.session_state["fin_sac"] = sac
        with st.expander("Ver tabelas completas"):
            t1, t2 = st.tabs(["Price", "SAC"])
            t1.dataframe(price.style.format("{:.2f}"), width="stretch")
            t2.dataframe(sac.style.format("{:.2f}"), width="stretch")

# -------------------------------------------------------------- aposentadoria
with aba_aposent:
    c1, c2 = st.columns(2)
    idade_atual = c1.number_input("Idade atual", min_value=16, max_value=90, value=30)
    idade_aposentadoria = c2.number_input("Idade desejada de aposentadoria", min_value=17, max_value=100, value=65)
    c3, c4 = st.columns(2)
    patrimonio_atual = c3.number_input("Patrimonio atual (R$)", min_value=0.0, value=10000.0, step=1000.0)
    aporte_mensal = c4.number_input("Aporte mensal (R$)", min_value=0.0, value=800.0, step=50.0)
    c5, c6 = st.columns(2)
    taxa_retorno = c5.number_input("Retorno real esperado (% a.a.)", min_value=0.0, value=6.0, step=0.5)
    taxa_retirada = c6.number_input("Taxa de retirada segura (% a.a.)", min_value=1.0, value=4.0, step=0.5)

    if st.button("Simular aposentadoria"):
        try:
            sim = simular_aposentadoria(int(idade_atual), int(idade_aposentadoria), patrimonio_atual, aporte_mensal, taxa_retorno / 100, taxa_retirada / 100)
            kpis(
                [
                    ("Patrimonio na aposentadoria", fmt_currency(sim.patrimonio_na_aposentadoria), ""),
                    ("Renda mensal sustentavel", fmt_currency(sim.renda_mensal_sustentavel), ""),
                    ("Total de juros ganhos", fmt_currency(sim.total_juros_ganhos), ""),
                ]
            )
            fig = go.Figure(go.Scatter(x=sim.evolucao["idade"], y=sim.evolucao["patrimonio"], mode="lines"))
            st.plotly_chart(estilizar(fig, "Evolucao do patrimonio ate a aposentadoria"), width="stretch")
            caixa(sim.resumo_texto(), "sucesso")
            st.session_state["fin_aposentadoria"] = sim
        except Exception as exc:
            st.error(str(exc))

# -------------------------------------------------------------- orcamento
with aba_orc:
    receita = st.number_input("Receita mensal total (R$)", min_value=0.0, value=6000.0, step=100.0)
    st.caption("Informe suas despesas por categoria (deixe 0 para ignorar):")
    categorias_padrao = ["Moradia", "Alimentacao", "Transporte", "Saude", "Lazer", "Educacao", "Outros"]
    despesas = {}
    cols = st.columns(4)
    for i, cat in enumerate(categorias_padrao):
        despesas[cat] = cols[i % 4].number_input(cat, min_value=0.0, value=0.0, step=50.0, key=f"desp_{cat}")

    if st.button("Analisar orcamento"):
        despesas_validas = {k: v for k, v in despesas.items() if v > 0}
        if not despesas_validas:
            st.warning("Informe ao menos uma despesa maior que zero.")
        else:
            orc = analisar_orcamento(receita, despesas_validas)
            kpis(
                [
                    ("Saldo mensal", fmt_currency(orc.saldo), ""),
                    ("% da renda comprometida", f"{orc.percentual_comprometido:.1f}%", ""),
                ]
            )
            fig = go.Figure(go.Pie(labels=orc.despesas_por_categoria.index, values=orc.despesas_por_categoria.values, hole=0.45))
            st.plotly_chart(estilizar(fig, "Despesas por categoria"), width="stretch")
            caixa(orc.resumo_texto().replace("\n\n", "<br><br>"), "sucesso" if orc.saldo >= 0 else "alerta")
            st.session_state["fin_orcamento"] = orc

st.divider()
if st.button("📄 Gerar relatorio consolidado de financas pessoais"):
    builder = ReportBuilder("Relatorio de Financas Pessoais", "Resumo das simulacoes realizadas nesta sessao")
    algo_adicionado = False
    if "fin_price" in st.session_state:
        builder.add_section(
            "Financiamento - Price x SAC",
            "Comparativo entre o Sistema Price (parcelas fixas) e o SAC (amortizacao constante).",
            tabelas=[("Price", st.session_state["fin_price"]), ("SAC", st.session_state["fin_sac"])],
        )
        algo_adicionado = True
    if "fin_aposentadoria" in st.session_state:
        sim = st.session_state["fin_aposentadoria"]
        builder.add_card("Patrimonio projetado", fmt_currency(sim.patrimonio_na_aposentadoria))
        builder.add_section("Simulacao de aposentadoria", sim.resumo_texto())
        algo_adicionado = True
    if "fin_orcamento" in st.session_state:
        orc = st.session_state["fin_orcamento"]
        builder.add_section(
            "Analise de orcamento", orc.resumo_texto(), tabelas=[("Despesas por categoria", orc.despesas_por_categoria.to_frame("valor"))]
        )
        algo_adicionado = True
    if not algo_adicionado:
        st.warning("Rode ao menos uma simulacao acima antes de gerar o relatorio.")
    else:
        botoes_download_relatorio(builder, "financas_pessoais")

rodape()
