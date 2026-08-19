# 📐 Econometria

Um sistema de econometria completo — para calculos **pessoais** e **empresariais** — pensado para
ser intuitivo, visualmente agradavel e, acima de tudo, **pratico**: voce carrega os dados (ou usa um
exemplo pronto), escolhe a analise, e o sistema entrega resultados interpretados e um **relatorio
automatico** (HTML, Word ou Excel) pronto para compartilhar.

O projeto tem duas camadas independentes:

- **`econometria/`** — o motor de calculo em Python puro (sem nenhuma dependencia de UI). Pode ser
  usado direto em notebooks/scripts, ou por qualquer outra interface que se queira construir no futuro.
- **`app/`** — uma aplicacao web (Streamlit) que expoe todo o motor atraves de uma interface visual,
  com upload de dados, graficos interativos e exportacao de relatorios.

## 🧭 Frentes cobertas

| Area | O que tem |
|---|---|
| **Estatistica descritiva** | Medidas de posicao/dispersao/forma, correlacao (Pearson/Spearman/Kendall), testes de normalidade (Jarque-Bera, Shapiro-Wilk) |
| **Testes de hipotese** | Teste t (uma amostra, duas amostras, pareado), ANOVA one-way, qui-quadrado de independencia, teste de proporcao |
| **Regressao linear** | OLS e WLS multiplos, erros padrao robustos (HC0-HC3), intervalos de confianca |
| **Diagnosticos de regressao** | VIF (multicolinearidade), Breusch-Pagan e White (heterocedasticidade), Durbin-Watson e Breusch-Godfrey (autocorrelacao), Jarque-Bera (normalidade dos residuos), RESET de Ramsey (forma funcional) |
| **Escolha discreta** | Logit e Probit, efeitos marginais, razao de chances, matriz de confusao |
| **Series temporais** | ADF e KPSS (estacionariedade), decomposicao STL/classica, ARIMA/SARIMA com selecao automatica de ordem (AIC), GARCH (volatilidade) |
| **Dados em painel** | Pooled OLS, Efeitos Fixos, Efeitos Aleatorios, teste de Hausman |
| **Previsao** | Ingenuo, media movel, Holt-Winters, ARIMA — comparados por backtest (validacao fora da amostra) com MAE/RMSE/MAPE |
| **Financas pessoais** | Juros compostos, conversao de taxas, financiamento (Price x SAC), simulacao de aposentadoria, analise de orcamento |
| **Financas empresariais** | VPL, TIR, payback simples/descontado, ponto de equilibrio, indicadores financeiros (margens, ROA, ROE, liquidez, endividamento), veredito de viabilidade |
| **Relatorios automaticos** | Todo resultado pode virar um relatorio HTML (visual, printavel), Word (.docx) ou Excel (.xlsx), com texto interpretativo pronto |

Cada resultado no sistema vem com um `resumo_texto()` (ou equivalente) que traduz numeros em
interpretacao pratica em portugues — o objetivo e que o usuario nao precise ser um econometrista
para entender o que o numero significa.

## 🗂️ Estrutura do projeto

```
econometria/              # motor de calculo (independente de UI)
├── data/                 # ingestao, limpeza e perfilamento automatico de dados
├── stats/                # estatistica descritiva e testes de hipotese
├── regression/            # OLS/WLS, diagnosticos, Logit/Probit
├── timeseries/            # estacionariedade, decomposicao, ARIMA, GARCH
├── panel/                 # Pooled OLS, Efeitos Fixos/Aleatorios, Hausman
├── forecasting/           # interface unificada de previsao + backtest
├── finance/                # financas pessoais e empresariais
├── reports/                # geracao de relatorios HTML/DOCX/XLSX
└── utils/                  # validacao e formatacao compartilhadas

app/                       # aplicacao Streamlit (interface visual)
├── Home.py                 # pagina inicial (upload de dados, navegacao)
├── pages/                  # uma pagina por frente de analise
├── theme.py, state.py, components.py   # infraestrutura visual compartilhada

examples/                  # datasets de exemplo (gerados por scripts/gerar_exemplos.py)
tests/                      # suite pytest do motor de calculo
```

## 🚀 Como rodar

Requer Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/Home.py
```

A aplicacao abre em `http://localhost:8501`. Na barra lateral, envie seus proprios dados (CSV, Excel,
JSON ou Parquet) ou clique em um dos **datasets de exemplo** para explorar qualquer pagina na hora.

> Alternativa: `pip install -e .` instala o pacote `econometria` no ambiente, permitindo usar o motor
> de calculo diretamente em scripts/notebooks (`from econometria.regression import ols_regression`).

## ☁️ Deploy (uso online)

Este e um app **Streamlit**: um servidor Python com estado (WebSocket + sessao em memoria), nao um
site estatico nem uma API serverless. Por isso ele **nao roda em plataformas serverless como Vercel**
— precisa de um host que mantenha um processo Python vivo. A opcao gratuita e oficial para isso e o
**Streamlit Community Cloud**, e o repositorio ja esta pronto para ele (`requirements.txt` na raiz,
`.streamlit/config.toml` com o tema visual do sistema, `.python-version` fixando Python 3.11):

1. Acesse **[share.streamlit.io](https://share.streamlit.io)** e entre com sua conta GitHub.
2. Clique em **"New app"** e selecione este repositorio e a branch desejada.
3. Em **"Main file path"**, informe `app/Home.py`.
4. Clique em **"Deploy"**. Em poucos minutos o app fica disponivel em uma URL publica
   (formato `https://<nome-do-app>.streamlit.app`), pronta para acessar do navegador ou compartilhar.

Cada novo `git push` na branch conectada atualiza o app automaticamente.

**Alternativas** (quando se precisa de mais controle, dominio proprio ou infraestrutura paga): qualquer
plataforma que rode containers/servidores persistentes funciona — Render, Railway, Fly.io, Hugging Face
Spaces ou um servidor proprio com `streamlit run app/Home.py` atras de um proxy reverso.

## 🧪 Testes

```bash
pip install -r requirements.txt   # ja inclui pytest
python -m pytest
```

A suite cobre o motor de calculo (`econometria/`): formulas financeiras conferidas contra calculo
fechado, recuperacao de coeficientes conhecidos em regressao/painel, metricas de erro de previsao,
geracao de relatorios, e o pipeline de ingestao/limpeza de dados.

## 📊 Datasets de exemplo

Gerados de forma sintetica e determinística (`scripts/gerar_exemplos.py`), para o app funcionar sem
depender de nenhum download externo:

- **`vendas_mensais.csv`** — serie temporal (vendas, marketing, temperatura) para as paginas de
  Series Temporais e Previsao.
- **`imoveis.csv`** — corte transversal (preco, area, quartos...) para Regressao Linear.
- **`clientes_churn.csv`** — variavel binaria de churn para Escolha Discreta.
- **`painel_empresas.csv`** — 25 empresas x 10 anos para Dados em Painel.

## 🧩 Design do sistema

- **Motor de calculo desacoplado da UI.** Toda funcao em `econometria/` recebe dados (`DataFrame`/`Series`)
  e devolve um objeto de resultado tipado (`@dataclass`) com os numeros **e** um metodo de interpretacao
  em texto (`resumo_texto()`/`conclusao()`/`interpretacao()`). Isso permite reusar o mesmo motor em
  qualquer interface futura (CLI, API, notebook) sem reescrever logica.
- **Validacao com mensagens em portugues.** Erros comuns (colunas faltando, dados nao numericos,
  poucas observacoes) levantam `ValidationError` com uma mensagem pronta para mostrar ao usuario final.
- **Relatorios como cidadaos de primeira classe.** `econometria.reports.ReportBuilder` monta um
  relatorio a partir de cartoes de indicadores, secoes de texto, tabelas e graficos — e exporta para
  HTML, DOCX ou XLSX com a mesma API, independente da analise que o originou.
- **Deteccao automatica de dados.** `econometria.data.profile_dataset` faz um raio-x do arquivo
  carregado (tipos de coluna, dados faltantes, se parece serie temporal ou painel) para orientar o
  usuario sobre qual analise faz sentido.

## 🛣️ Proximos passos sugeridos

O sistema foi construido para crescer sem redesenho: cada frente e um modulo independente. Ideias de
extensao natural: modelos VAR/VECM para multiplas series, cointegracao (Engle-Granger/Johansen),
variaveis instrumentais (2SLS), regressao quantílica, series temporais em painel (dados em painel
dinamico), autenticacao multiusuario e persistencia de historico de analises.

---

_Sistema desenvolvido para cobrir, de forma pratica e extensivel, as principais frentes de aplicacao
da econometria em contextos pessoais e empresariais._
