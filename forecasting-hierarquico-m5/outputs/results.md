# Resultados — Forecasting hierárquico M5-CA (V1)

Backtest expanding-window: 4 janelas × 28 dias, MASE escalada pelo naive sazonal
in-sample. Hierarquia: estado → loja → loja×categoria → loja×depto (45 séries).

## MASE médio (todas as janelas e níveis)

| modelo        |   base |   BottomUp |   TopDown |   MinTrace |
|:--------------|-------:|-----------:|----------:|-----------:|
| AutoARIMA     |  0.867 |      0.886 |     0.864 |      0.841 |
| AutoETS       |  0.817 |      0.81  |     0.818 |      0.817 |
| SeasonalNaive |  0.991 |      0.991 |     0.991 |      0.991 |

## MASE por nível (AutoETS)

| nivel                            |   base |   BottomUp |   TopDown |   MinTrace |
|:---------------------------------|-------:|-----------:|----------:|-----------:|
| state_id                         |  0.69  |      0.689 |     0.69  |      0.697 |
| state_id/store_id                |  0.82  |      0.793 |     0.81  |      0.805 |
| state_id/store_id/cat_id         |  0.804 |      0.805 |     0.813 |      0.812 |
| state_id/store_id/cat_id/dept_id |  0.952 |      0.952 |     0.958 |      0.955 |

## Leitura honesta

- **Melhor combinação global: AutoETS + bottom-up** (MASE 0.810).
  E o **AutoETS sozinho, sem reconciliação, já bate tudo** (MASE 0.817
  vs. 0.867 do ARIMA e 0.991 do
  naive) — a lição da competição M5 inteira: clássico bem ajustado é um
  adversário duríssimo, e a maioria dos métodos "modernos" não o supera de forma consistente.
- **O ganho da reconciliação é real, porém pequeno e DEPENDE do modelo base.** Para o
  AutoARIMA, MinT-shrink corta o MASE de 0.867 para
  0.841 (a covariância dos erros ajuda um modelo mais
  ruidoso). Para o AutoETS — já bem calibrado — a reconciliação é praticamente neutra
  (0.817 → 0.817); o único ganho
  visível é do bottom-up no nível de lojas.
- **TopDown nunca é o melhor** e degrada levemente as folhas — proporções históricas
  apagam a dinâmica local.
- **Conclusão que a literatura M5 confirma:** a reconciliação vale mais pela *coerência*
  garantida entre níveis (as previsões somam — requisito operacional em varejo/supply
  chain) do que por um salto de acurácia. Reportar isso, em vez de vender MinT como bala
  de prata, é a maturidade que o benchmark honesto exige.

V2: desafiante neural (TFT / N-BEATS) sob o MESMO protocolo, WRMSSE oficial do M5 e
hierarquia completa até o item (~30 mil séries).
