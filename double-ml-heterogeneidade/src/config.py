"""Configuração central do projeto Double ML (V1)."""
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

# Dados Dehejia-Wahba (NSW re74 subsample) + controles PSID, hospedados no NBER.
NBER_BASE = "https://users.nber.org/~rdehejia/data/"
FILES = {
    "nsw_treated": "nswre74_treated.txt",     # 185 tratados do experimento NSW
    "nsw_control": "nswre74_control.txt",     # 260 controles experimentais
    "psid_controls": "psid_controls.txt",     # 2.490 controles observacionais (PSID)
}
COLS = ["treat", "age", "educ", "black", "hisp", "married",
        "nodegree", "re74", "re75", "re78"]

SEED = 42
N_FOLDS = 5  # cross-fitting do DML
