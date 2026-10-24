# CLAUDE.md section 8.1 mandated test: quantify, on the Madrid series, (a) the property
# refresh v1 -> v2, (b) the per site pressure (Madrid stations sit at 500 to 610 m, about
# 948 hPa), and (c) the vapour partial pressure entropy variant. Decision rule from the
# brief: if the variant moves WUE outside the +-1.5% propagated band, it is material and
# must be adopted and documented; report the result either way.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dcfootprint.core.properties import V1, V2, V2Properties
from dcfootprint.core.tower import model_vec
from dcfootprint.core.modes import select_mode

P_MADRID = 94850.0   # Pa, barometric at the 555 m mean elevation of the SIAR stations

cols = ["IdEstacion", "Fecha", "Anio", "Dia", "HoraMin", "T", "HR"]
df = pd.read_csv(ROOT / "Old_analysis" / "todosLosDatos.txt", sep="\t",
                 usecols=[1, 2, 3, 4, 5, 6, 7], names=cols, header=0)
df = df.dropna(subset=["T", "HR"]).reset_index(drop=True)
T = df["T"].values; HR = df["HR"].values
n = len(df)
T1, T4, ev, ch, md = select_mode(T, HR, T4base=26.0)
HR5 = np.full(n, 100.0)

variants = [
    ("v1 fits, 1e5 Pa (baseline)",            V1, 1e5),
    ("v2 fits, 1e5 Pa (property refresh)",    V2, 1e5),
    ("v2 fits, 948.5 hPa (site pressure)",    V2, P_MADRID),
    ("v2 fits, 948.5 hPa, partial pressure",  V2Properties(vapour_partial_pressure=True), P_MADRID),
]

base = None
print(f"Madrid series, {n:,} records, loop 50/26\n")
print(f"{'variant':42s} {'PUE':>8s} {'WUE':>8s} {'T5 mean':>8s} {'infeas':>7s} "
      f"{'dPUE %':>7s} {'dWUE %':>7s}")
for name, props, p in variants:
    T5, WUE, PUEnoIT, Tdew2 = model_vec(T1, T, HR, T, T4, HR5, ev, ch, p=p, props=props)
    PUE = PUEnoIT + 1.0
    ok = ~np.isnan(PUE) & ~np.isnan(WUE)
    pm = float(PUE[ok].mean()); wm = float(WUE[ok].mean()); t5m = float(T5[ok].mean())
    ninf = int((~ok).sum())
    if base is None:
        base = (pm, wm)
        print(f"{name:42s} {pm:8.4f} {wm:8.4f} {t5m:8.3f} {ninf:7d} {'':>7s} {'':>7s}")
    else:
        dp = 100 * (pm - base[0]) / base[0]; dw = 100 * (wm - base[1]) / base[1]
        print(f"{name:42s} {pm:8.4f} {wm:8.4f} {t5m:8.3f} {ninf:7d} {dp:7.2f} {dw:7.2f}")
print("\npropagated WUE band from the GSA: +-1.5% -> WUE within [",
      f"{base[1]*0.985:.4f}, {base[1]*1.015:.4f}] is immaterial")
