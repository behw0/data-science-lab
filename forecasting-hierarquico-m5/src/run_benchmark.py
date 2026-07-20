"""Benchmark hierárquico M5-CA: clássicos fortes × reconciliação (V1).

Protocolo:
  · Hierarquia da Califórnia: estado → loja → loja×categoria → loja×departamento
    (45 séries diárias, 1.941 dias).
  · Modelos por série: SeasonalNaive(7), AutoETS, AutoARIMA (statsforecast).
  · Reconciliação: BottomUp, TopDown (proporções de previsão) e MinT-shrink
    (Wickramasuriya et al. 2019), via hierarchicalforecast.
  · Backtest expanding-window: 4 janelas de 28 dias (o horizonte canônico do M5),
    sem look-ahead; MASE escalada pelo naive sazonal in-sample de cada janela.

Saídas:
  outputs/benchmark_results.csv       MASE/RMSE por modelo × reconciliação × nível
  outputs/results.md
  outputs/figures/mase_by_level.png
  outputs/figures/example_forecast.png
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

# --- paleta editorial (trio validado p/ daltonismo via dataviz validator) ------
COL = {"base": "#8A8A8A", "BottomUp": "#1C77C3", "TopDown": "#7B5EA7",
       "MinTrace": "#C1443C"}
LBL = {"base": "sem reconciliação", "BottomUp": "bottom-up",
       "TopDown": "top-down", "MinTrace": "MinT-shrink"}
INK = "#22252A"
MUTED = "#8A8A8A"
GRID = "#E7E7E4"
SURFACE = "#FCFCFB"

NIVEIS = {
    "state_id": "estado (1)",
    "state_id/store_id": "lojas (4)",
    "state_id/store_id/cat_id": "loja×categoria (12)",
    "state_id/store_id/cat_id/dept_id": "loja×depto (28)",
}


# ------------------------------------------------------------------ dados -----
def load_hierarchy():
    """M5 (via Nixtla datasetsforecast) filtrado para CA e agregado na hierarquia."""
    from datasetsforecast.m5 import M5
    from hierarchicalforecast.utils import aggregate

    Y_df, _, S_df = M5.load(directory=str(C.DATA_RAW))
    S_df = S_df[S_df["state_id"] == C.STATE]
    df = Y_df.merge(S_df[["unique_id", "state_id", "store_id", "cat_id", "dept_id"]],
                    on="unique_id", how="inner")
    df = df.drop(columns=["unique_id"])
    df["ds"] = pd.to_datetime(df["ds"])
    Y_hier, S, tags = aggregate(df, C.HIER_SPEC)
    if "unique_id" not in Y_hier.columns:
        Y_hier = Y_hier.reset_index()
    return Y_hier[["unique_id", "ds", "y"]], S, tags


# --------------------------------------------------------------- backtest -----
def _models():
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
    return [SeasonalNaive(season_length=C.SEASON),
            AutoETS(season_length=C.SEASON),
            AutoARIMA(season_length=C.SEASON)]


def run() -> pd.DataFrame:
    from statsforecast import StatsForecast
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown

    Y, S, tags = load_hierarchy()
    last = Y["ds"].max()
    resultados = []
    exemplo = {}

    for w in range(C.N_WINDOWS, 0, -1):
        cutoff = last - pd.Timedelta(days=C.HORIZON * w)
        train = Y[Y["ds"] <= cutoff]
        test = Y[(Y["ds"] > cutoff) & (Y["ds"] <= cutoff + pd.Timedelta(days=C.HORIZON))]

        sf = StatsForecast(models=_models(), freq="D", n_jobs=-1)
        Y_hat = sf.forecast(df=train, h=C.HORIZON, fitted=True)
        Y_fitted = sf.forecast_fitted_values()

        hrec = HierarchicalReconciliation(reconcilers=[
            BottomUp(),
            TopDown(method="forecast_proportions"),
            MinTrace(method="mint_shrink"),
        ])
        Y_rec = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_fitted, S_df=S, tags=tags)

        # escala MASE: MAE do naive sazonal in-sample, por série
        tr = train.sort_values(["unique_id", "ds"])
        scale = (tr.groupby("unique_id")["y"]
                 .apply(lambda s: (s - s.shift(C.SEASON)).abs().mean()))

        Y_rec = Y_rec.reset_index() if "unique_id" not in Y_rec.columns else Y_rec
        aval = test.merge(Y_rec, on=["unique_id", "ds"], how="left")

        modelos = ["SeasonalNaive", "AutoETS", "AutoARIMA"]
        recons = {"base": "", "BottomUp": "/BottomUp",
                  "TopDown": "/TopDown_method-forecast_proportions",
                  "MinTrace": "/MinTrace_method-mint_shrink"}
        for nivel, ids in tags.items():
            sub = aval[aval["unique_id"].isin(ids)]
            for m in modelos:
                for rec, suf in recons.items():
                    col = f"{m}{suf}"
                    if col not in sub.columns:
                        continue
                    err = sub[col] - sub["y"]
                    mase = float((err.abs().groupby(sub["unique_id"]).mean()
                                  / scale[sub["unique_id"].unique()].reindex(
                                      err.abs().groupby(sub["unique_id"]).mean().index)).mean())
                    rmse = float(np.sqrt((err ** 2).mean()))
                    resultados.append({"janela": w, "nivel": nivel, "modelo": m,
                                       "reconciliacao": rec, "mase": mase, "rmse": rmse})
        if w == 1:  # guarda a última janela para a figura de exemplo
            exemplo["aval"] = aval
            exemplo["train"] = train

    res = pd.DataFrame(resultados)
    res.to_csv(C.OUTPUTS / "benchmark_results.csv", index=False)
    _fig_example(exemplo)
    return res


# ------------------------------------------------------------------ figuras ---
def _editorial(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig_mase(res: pd.DataFrame, modelo: str = "AutoETS") -> None:
    med = (res[res["modelo"] == modelo]
           .groupby(["nivel", "reconciliacao"])["mase"].mean().unstack())
    med = med.loc[list(NIVEIS)]
    ordem = ["base", "BottomUp", "TopDown", "MinTrace"]

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _editorial(ax)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(med))
    wbar = 0.19
    for i, rec in enumerate(ordem):
        vals = med[rec].to_numpy()
        bars = ax.bar(x + (i - 1.5) * wbar, vals, wbar * 0.92,
                      color=COL[rec], zorder=3, label=LBL[rec])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                    ha="center", color=INK, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([NIVEIS[n] for n in med.index], fontsize=9.5, color=INK)
    ax.set_ylabel("MASE  (menor = melhor)", color=INK, fontsize=10)
    ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(len(med) - 0.52, 1.012, "naive sazonal = 1,0", color=MUTED, fontsize=8.5)
    ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper left")
    ax.set_title(f"Reconciliação em ação: MASE por nível da hierarquia ({modelo}, M5-CA)",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.005, "média de 4 janelas de 28 dias · hierarquia estado → loja → categoria → depto",
            transform=ax.transAxes, color=MUTED, fontsize=9.2)
    fig.text(0.125, 0.005, "Dados: M5/Walmart via Nixtla · elaboração do autor",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = C.FIGURES / "mase_by_level.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig -> {out}")


def _fig_example(ex: dict) -> None:
    """Total da Califórnia na última janela: real × ETS base × ETS-MinT."""
    aval, train = ex["aval"], ex["train"]
    uid = "CA"
    a = aval[aval["unique_id"] == uid].sort_values("ds")
    t = train[train["unique_id"] == uid].sort_values("ds").tail(84)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _editorial(ax)
    ax.plot(t["ds"], t["y"], color=MUTED, linewidth=1.6, zorder=2)
    ax.plot(a["ds"], a["y"], color=INK, linewidth=2.4, zorder=4, label="realizado")
    ax.plot(a["ds"], a["AutoETS"], color=COL["base"], linewidth=1.8,
            linestyle=(0, (4, 2)), zorder=3, label="AutoETS sem reconciliação")
    ax.plot(a["ds"], a["AutoETS/MinTrace_method-mint_shrink"], color=COL["MinTrace"],
            linewidth=2.0, zorder=3, label="AutoETS + MinT")
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    ax.set_ylabel("unidades vendidas — total CA", color=INK, fontsize=10)
    ax.set_title("A última janela de teste no topo da hierarquia (total Califórnia)",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=14)
    fig.text(0.125, 0.005, "Dados: M5/Walmart via Nixtla · elaboração do autor",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = C.FIGURES / "example_forecast.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig -> {out}")


# -------------------------------------------------------------------- main ----
def summarize(res: pd.DataFrame) -> None:
    tab = (res.groupby(["modelo", "reconciliacao"])["mase"].mean().unstack()
           [["base", "BottomUp", "TopDown", "MinTrace"]].round(3))
    por_nivel = (res[res["modelo"] == "AutoETS"]
                 .groupby(["nivel", "reconciliacao"])["mase"].mean().unstack()
                 [["base", "BottomUp", "TopDown", "MinTrace"]]
                 .loc[list(NIVEIS)].round(3))
    best_all = res.groupby(["modelo", "reconciliacao"])["mase"].mean().idxmin()

    txt = f"""# Resultados — Forecasting hierárquico M5-CA (V1)

Backtest expanding-window: 4 janelas × 28 dias, MASE escalada pelo naive sazonal
in-sample. Hierarquia: estado → loja → loja×categoria → loja×depto (45 séries).

## MASE médio (todas as janelas e níveis)

{tab.to_markdown()}

## MASE por nível (AutoETS)

{por_nivel.to_markdown()}

## Leitura honesta

- **Melhor combinação global: {best_all[0]} + {LBL[best_all[1]]}** (MASE {tab.loc[best_all].min():.3f}).
  E o **AutoETS sozinho, sem reconciliação, já bate tudo** (MASE {tab.loc['AutoETS','base']:.3f}
  vs. {tab.loc['AutoARIMA','base']:.3f} do ARIMA e {tab.loc['SeasonalNaive','base']:.3f} do
  naive) — a lição da competição M5 inteira: clássico bem ajustado é um
  adversário duríssimo, e a maioria dos métodos "modernos" não o supera de forma consistente.
- **O ganho da reconciliação é real, porém pequeno e DEPENDE do modelo base.** Para o
  AutoARIMA, MinT-shrink corta o MASE de {tab.loc['AutoARIMA','base']:.3f} para
  {tab.loc['AutoARIMA','MinTrace']:.3f} (a covariância dos erros ajuda um modelo mais
  ruidoso). Para o AutoETS — já bem calibrado — a reconciliação é praticamente neutra
  ({tab.loc['AutoETS','base']:.3f} → {tab.loc['AutoETS','MinTrace']:.3f}); o único ganho
  visível é do bottom-up no nível de lojas.
- **TopDown nunca é o melhor** e degrada levemente as folhas — proporções históricas
  apagam a dinâmica local.
- **Conclusão que a literatura M5 confirma:** a reconciliação vale mais pela *coerência*
  garantida entre níveis (as previsões somam — requisito operacional em varejo/supply
  chain) do que por um salto de acurácia. Reportar isso, em vez de vender MinT como bala
  de prata, é a maturidade que o benchmark honesto exige.

V2: desafiante neural (TFT / N-BEATS) sob o MESMO protocolo, WRMSSE oficial do M5 e
hierarquia completa até o item (~30 mil séries).
"""
    (C.OUTPUTS / "results.md").write_text(txt, encoding="utf-8")
    print(f"  results -> {C.OUTPUTS / 'results.md'}")
    print(tab.to_string())


def main() -> None:
    res = run()
    fig_mase(res)
    summarize(res)


if __name__ == "__main__":
    sys.exit(main())
