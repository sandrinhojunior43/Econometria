"""
econometria
===========

Motor de calculos econometricos para uso pessoal e empresarial.

Submodulos
----------
- ``econometria.data``        : carga, deteccao de tipos e limpeza de dados.
- ``econometria.stats``       : estatistica descritiva e testes de hipotese.
- ``econometria.regression``  : regressao linear, diagnosticos e escolha discreta.
- ``econometria.timeseries``  : estacionariedade, decomposicao, ARIMA/SARIMA e GARCH.
- ``econometria.panel``       : dados em painel (Pooled OLS, Efeitos Fixos/Aleatorios, Hausman).
- ``econometria.forecasting`` : interface unificada de previsao e backtesting.
- ``econometria.finance``     : financas pessoais e empresariais (VP/VF, VPL, TIR, financiamentos...).
- ``econometria.reports``     : geracao automatica de relatorios (HTML, DOCX, XLSX).

Cada modulo funciona de forma independente do resto do sistema: pode ser usado
via a API Python diretamente, ou atraves da aplicacao Streamlit em ``app/``.
"""

__version__ = "0.1.0"
