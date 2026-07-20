# Para quem a política funciona? Efeitos heterogêneos com Double/Debiased ML

*Causal ML · DML / Causal Forests · Fronteira Chernozhukov–Athey*

> ⚪ **Status: Planejado.** Projeto 3 do `data-science-lab` — abre o repositório com a
> skill mais quente da interseção ML × causalidade.

---

## Pergunta

Um tratamento (desconto, programa público, campanha de crédito) tem efeito médio
conhecido — mas para **quem** funciona e para quem é desperdício? Quanto do orçamento é
mal alocado por ignorar heterogeneidade?

## Dados

- Base pública com tratamento real: Lalonde/PSID (clássico revisitado) ou microdados de
  programa brasileiro (Bolsa Família via CECAD) ou experimento de marketing (Criteo Uplift)
- Covariáveis ricas de pré-tratamento para modelar heterogeneidade

## Método

**Double/Debiased ML** (Chernozhukov et al. 2018) para o efeito médio com nuisance
functions em gradient boosting; **Causal Forests** (Athey–Wager) para o CATE. Validação
com overlap, calibração de CATE e política ótima de targeting avaliada out-of-sample.

## Resultado-alvo

Curva de uplift + mapa de heterogeneidade: "tratando só os 30% com maior CATE, captura-se
Y% do efeito total com Z% do custo". Comparação honesta DML vs. regressão tradicional.

## Versão Mínima Viável (V1)

Lalonde com DML básico via EconML/DoubleML, um gráfico de CATE bem feito e README
explicando as hipóteses de identificação. Causal Forest e política de targeting entram na V2.

## Stack

`Python` · `EconML` / `DoubleML` · `scikit-learn` · `LightGBM` · `matplotlib`
