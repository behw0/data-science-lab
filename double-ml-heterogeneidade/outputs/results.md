# Resultados — Double ML no NSW/Lalonde (V1)

| Método | n | ATE (US$) | EP | IC 90% |
|---|---|---|---|---|
| Experimento NSW (benchmark) | 445 | +1 794 | 671 | [+691  +2 898] |
| Diferença de médias (ingênua) | 2 675 | -15 205 | 656 | [-16 284  -14 126] |
| OLS com controles | 2 675 | +752 | 783 | [-537  +2 040] |
| DML partialling-out (LightGBM) | 2 675 | -1 679 | 855 | [-3 086  -272] |
| AIPW/IRM cross-fitted | 2 675 | -11 380 | 4 020 | [-17 992  -4 767] |
| AIPW/IRM na região de overlap | 387 | -2 592 | 1 591 | [-5 209  +25] |

## O achado (honesto e contraintuitivo)

**Nenhum método observacional recupera o experimento — nem o Double ML.** Mas a
comparação é informativa em três camadas:

1. **A ingenuidade custa US$ 17 mil**: a diferença de médias contra o PSID erra por
   -16 999 e inverte o sinal do programa. Seleção brutal: quem entra num
   programa para desempregados é muito mais pobre que a média populacional.
2. **O ML flexível remove ~80% do viés** (erro de -16 999 →
   -3 473 no DML partialling-out)  automatizando o que exigiria engenharia
   manual de especificação. É muito — e não é suficiente.
3. **O resíduo não é problema de algoritmo  é de identificação**: 74%
   dos controles PSID têm propensity < 0 05 (ver `overlap.png`) e a seleção remanescente
   é em *não-observáveis* (o "Ashenfelter dip" — a queda de renda pré-programa dos
   participantes). Nem AIPW duplamente robusto com trimming escapa: sem suporte comum e
   sem desenho  não há estimador que salve.

**A lição de LaLonde (1986) sobrevive ao ML de 2018** — conclusão que a literatura
moderna confirma (Imbens & Xu  2024). Para o praticante de decision science: DML é a
ferramenta certa *quando a identificação já está de pé* (experimentos com covariáveis 
descontinuidades  painéis ricos); não é um substituto para o desenho.

## CATE — para quem funciona? (na amostra experimental)

No experimento (identificação limpa)  o CATE linear em (idade  escolaridade) sugere
efeito **crescente com a idade e com a escolaridade**: para quem tem ensino médio  o
efeito estimado vai de ~US$ 1.800 (aos 18) a ~US$ 3.500 (aos 50); sem diploma  a curva
começa perto de zero. Leitura: o treinamento rende mais para quem já tem capital humano
básico para convertê-lo em emprego. Com n = 445  os ICs são largos — a linha "sem
diploma" não exclui zero em boa parte do suporte: heterogeneidade é *sugerida*  não
provada. V2: Causal Forests (Athey–Wager)  calibração de CATE e política de targeting
avaliada out-of-sample.
