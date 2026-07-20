"""Configuração central do projeto de forecasting hierárquico M5 (V1)."""
from __future__ import annotations

import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
for _d in (DATA_RAW, DATA_PROC, OUTPUTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --- Desenho V1 ----------------------------------------------------------------
# Uma região do M5: Califórnia. Hierarquia estado -> loja -> (loja×categoria)
# -> (loja×departamento). 45 séries diárias, 1.941 dias.
STATE = "CA"
HIER_SPEC = [
    ["state_id"],
    ["state_id", "store_id"],
    ["state_id", "store_id", "cat_id"],
    ["state_id", "store_id", "cat_id", "dept_id"],  # dept aninha em cat no M5
]

HORIZON = 28        # horizonte canônico do M5 (28 dias)
N_WINDOWS = 4       # janelas de cross-validation rolling (4 × 28 = 112 dias de teste)
SEASON = 7          # sazonalidade semanal
SEED = 42
