import pytest

from econometria.finance.personal import (
    analisar_orcamento,
    simular_aposentadoria,
    tabela_price,
    tabela_sac,
    taxa_equivalente,
    valor_futuro,
    valor_presente,
)


def test_valor_futuro_sem_aportes():
    # 1000 a 1% ao mes por 12 meses = 1000 * 1.01**12
    resultado = valor_futuro(1000, 0.01, 12)
    assert resultado == pytest.approx(1000 * 1.01**12, rel=1e-9)


def test_valor_futuro_com_aportes_postecipados():
    # Formula fechada da anuidade postecipada
    aporte, taxa, n = 100, 0.01, 12
    esperado = aporte * ((1.01**12 - 1) / 0.01)
    resultado = valor_futuro(0, taxa, n, aporte_periodico=aporte)
    assert resultado == pytest.approx(esperado, rel=1e-9)


def test_valor_presente_e_valor_futuro_sao_inversos():
    vf = valor_futuro(500, 0.02, 10)
    vp = valor_presente(vf, 0.02, 10)
    assert vp == pytest.approx(500, rel=1e-9)


def test_taxa_equivalente_anual_para_mensal():
    # 8% a.a. -> mensal deve ser bem menor que 8/12
    mensal = taxa_equivalente(0.08, 1, 12)
    assert 0 < mensal < 0.08 / 12 * 1.1
    # ida e volta deve devolver a taxa original
    anual_de_volta = taxa_equivalente(mensal, 12, 1)
    assert anual_de_volta == pytest.approx(0.08, rel=1e-9)


def test_tabela_price_parcela_constante():
    tabela = tabela_price(10000, 0.02, 12)
    assert tabela["prestacao"].nunique() == 1  # parcela fixa e caracteristica do Price
    assert tabela["saldo_devedor"].iloc[-1] == pytest.approx(0, abs=1e-6)


def test_tabela_sac_amortizacao_constante():
    tabela = tabela_sac(10000, 0.02, 12)
    assert tabela["amortizacao"].nunique() == 1
    assert tabela["prestacao"].is_monotonic_decreasing
    assert tabela["saldo_devedor"].iloc[-1] == pytest.approx(0, abs=1e-6)


def test_sac_paga_menos_juros_totais_que_price():
    price = tabela_price(10000, 0.02, 12)
    sac = tabela_sac(10000, 0.02, 12)
    assert sac["juros"].sum() < price["juros"].sum()


def test_simular_aposentadoria_patrimonio_positivo_e_razoavel():
    sim = simular_aposentadoria(30, 60, 5000, 500, 0.06)
    assert sim.patrimonio_na_aposentadoria > sim.total_aportado
    assert sim.patrimonio_na_aposentadoria < 50_000_000  # sanity: nada de estouro numerico
    assert sim.anos_ate_aposentadoria == 30


def test_simular_aposentadoria_idade_invalida():
    with pytest.raises(ValueError):
        simular_aposentadoria(60, 40, 1000, 100, 0.05)


def test_analisar_orcamento_saldo_negativo_gera_alerta():
    orc = analisar_orcamento(3000, {"aluguel": 2000, "contas": 1500})
    assert orc.saldo < 0
    assert any("vermelho" in a for a in orc.alertas)


def test_analisar_orcamento_saldo_positivo():
    orc = analisar_orcamento(8000, {"aluguel": 2000, "alimentacao": 1000})
    assert orc.saldo == pytest.approx(5000)
    assert orc.percentual_comprometido == pytest.approx(37.5)
