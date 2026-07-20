# Forecasting hierárquico com reconciliação ótima

*Séries hierárquicas · Reconciliação MinT · Benchmark honesto*

> ⚪ **Status: Planejado.** Projeto 4 do `data-science-lab`.

---

## Pergunta

Em uma hierarquia real (SKU → loja → região → total), como garantir previsões consistentes
entre níveis? E a pergunta que quase ninguém responde com honestidade: quando deep learning
vale a pena — e quando ETS/ARIMA vencem?

## Dados

- Dataset **M5** (Walmart, ~30 mil séries hierárquicas) — o benchmark canônico da área
- Splits temporais rigorosos, sem look-ahead bias

## Método

Baselines fortes (ETS, ARIMA, naive sazonal) por série; `statsforecast` em escala; um
modelo neural (TFT ou N-BEATS) como desafiante; reconciliação bottom-up, top-down e
**MinT** (Wickramasuriya et al.). Backtest expanding-window em múltiplos horizontes.

## Resultado-alvo

Tabela definitiva método × horizonte × nível hierárquico, com a conclusão honesta que a
literatura M5 confirma: o clássico bem-feito frequentemente vence.

## Versão Mínima Viável (V1)

Um subconjunto do M5 (uma região), três baselines clássicos + reconciliação bottom-up vs.
MinT, sem rede neural. A comparação com deep learning entra na V2.

## Stack

`Python` · `statsforecast` / `hierarchicalforecast` (Nixtla) · `darts` · `pandas`
