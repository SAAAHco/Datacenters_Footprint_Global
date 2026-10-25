# Sensitivity of the chiller term to a downward condenser water reset. The production policy
# holds the condenser water at the design set point and resets it upward only; plant practice
# also relieves it downward toward the wet bulb plus approach. This script re-evaluates the
# chiller electricity of every chiller hour with the set point floated to
#     T4 = min(T4 of the run, max(T4_min, ceil(Twb + A_WET)))
# for two chiller minimum entering temperatures, and reports the PUE saving (station means,
# world and per class). The tower side is left unchanged (the outlet state and the evaporation
# move little with T4; see Section 2.4 of the main text), so the saving is the chiller term only.
# Output: results_world/reset_down_sensitivity.csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcfootprint.core.tower import cop            # noqa: E402
from dcfootprint.core.modes import A_WET          # noqa: E402

OUT = ROOT / "results_world"
res = pd.read_csv(OUT / "station_results.csv"); res = res[res["screen_pass"] == True]    # noqa: E712
t4b = dict(zip(res["STATION"], res["T4base"]))
dc = pd.read_csv(OUT / "dc_classes.csv")[["STATION", "dc_class"]]
FLOORS = (18.0, 20.0, 22.0)
pf = pq.ParquetFile(OUT / "world_hourly.parquet"); rows = []
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "Twb", "mode", "reset"]).to_pandas()
    st = d["STATION"].iloc[0]
    if st not in t4b:
        continue
    ch = d["mode"].to_numpy() == 2
    T4 = t4b[st] + d["reset"].to_numpy(float)
    r = dict(STATION=st, n=len(d), chill_share=float(ch.mean()))
    for tmin in FLOORS:
        T4f = np.minimum(T4, np.maximum(tmin, np.ceil(d["Twb"].to_numpy(float) + A_WET)))
        dP = np.where(ch, 1.0 / cop(T4) - 1.0 / cop(T4f), 0.0)
        r[f"dPUE_floor{int(tmin)}"] = float(dP.mean())
        r[f"dT4_floor{int(tmin)}"] = float((T4 - T4f)[ch].mean()) if ch.any() else 0.0
    rows.append(r)
S = pd.DataFrame(rows).merge(dc, on="STATION", how="left")
S.to_csv(OUT / "reset_down_stations.csv", index=False)
summ = []
for scope, g in [("world", S)] + [(f"class {int(c)}", S[S["dc_class"] == c]) for c in sorted(S["dc_class"].dropna().unique())]:
    row = dict(scope=scope, n=len(g))
    for tmin in FLOORS:
        row[f"dPUE_floor{int(tmin)}"] = g[f"dPUE_floor{int(tmin)}"].mean()
        row[f"dT4_floor{int(tmin)}"] = g[f"dT4_floor{int(tmin)}"].mean()
    summ.append(row)
su = pd.DataFrame(summ); su.to_csv(OUT / "reset_down_sensitivity.csv", index=False)
print(su.round(4).to_string(index=False))
