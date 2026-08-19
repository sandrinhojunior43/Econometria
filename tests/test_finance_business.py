import pytest

from econometria.finance.business import (
    analisar_viabilidade,
    indicadores_financeiros,
    payback_simples,
    ponto_equilibrio,
    taxa_interna_retorno,
    valor_presente_liquido,
)


def test_vpl_taxa_zero_igual_soma_fluxos():
    fluxos = [-1000, 300, 300, 300, 300]
    assert valor_presente_liquido(fluxos, 0.0) == pytest.approx(sum(fluxos))


def test_vpl_positivo_projeto_atrativo():
    fluxos = [-1000, 400, 400, 400, 400]
    vpl_baixa_taxa = valor_presente_liquido(fluxos, 0.05)
    vpl_alta_taxa = valor_presente_liquido(fluxos, 0.30)
    assert vpl_baixa_taxa > vpl_alta_taxa  # VPL cai conforme a taxa de desconto sobe


def test_tir_e_a_taxa_que_zera_o_vpl():
    fluxos = [-1000, 300, 400, 500, 300]
    tir = taxa_interna_retorno(fluxos)
    assert valor_presente_liquido(fluxos, tir) == pytest.approx(0, abs=1e-6)


def test_tir_sem_mudanca_de_sinal_gera_erro():
    with pytest.raises(ValueError):
        taxa_interna_retorno([100, 100, 100])


def test_payback_simples():
    # -1000, +400, +400, +400 -> recupera entre o 2o e o 3o periodo
    payback = payback_simples([-1000, 400, 400, 400])
    assert 2 < payback < 3


def test_payback_simples_nao_recupera():
    assert payback_simples([-1000, 100, 100]) is None


def test_ponto_equilibrio_valores_esperados():
    pe = ponto_equilibrio(custos_fixos=10000, preco_venda_unitario=50, custo_variavel_unitario=30)
    assert pe.quantidade_equilibrio == pytest.approx(500)
    assert pe.receita_equilibrio == pytest.approx(25000)
    assert pe.margem_contribuicao_percentual == pytest.approx(40)


def test_ponto_equilibrio_preco_menor_que_custo_gera_erro():
    with pytest.raises(ValueError):
        ponto_equilibrio(10000, 20, 30)


def test_indicadores_financeiros_calculo_basico():
    tabela = indicadores_financeiros(
        receita_liquida=1000,
        lucro_bruto=400,
        lucro_liquido=100,
        ativo_total=800,
        patrimonio_liquido=300,
        ativo_circulante=200,
        passivo_circulante=100,
        passivo_total=500,
    )
    assert tabela.loc["Margem bruta", "valor"] == pytest.approx(0.4)
    assert tabela.loc["Margem liquida", "valor"] == pytest.approx(0.1)
    assert tabela.loc["ROE (retorno sobre patrimonio)", "valor"] == pytest.approx(100 / 300)
    assert tabela.loc["Liquidez corrente", "valor"] == pytest.approx(2.0)


def test_analisar_viabilidade_projeto_viavel():
    analise = analisar_viabilidade([-10000, 4000, 4000, 4000, 4000], 0.1)
    assert analise.viavel is True
    assert analise.vpl > 0
    assert analise.tir > 0.1


def test_analisar_viabilidade_projeto_inviavel():
    analise = analisar_viabilidade([-10000, 1000, 1000, 1000], 0.1)
    assert analise.viavel is False
    assert analise.vpl < 0
