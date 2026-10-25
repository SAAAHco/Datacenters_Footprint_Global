# Where the closure sits, and two convention sensitivities, on the temperature-humidity grid.
#
# 1. For every (design set point, pressure) group of the panel, the production policy is run on
#    the 0.5 C by 1% grid and each station's hour histogram is integrated against it, giving
#    the share of hours in which the outlet air sits at the warm-water cap (T5 = T1) in each
#    mode, and the median depth of the zero-entropy-production root below the warm water in
#    the chiller hours where it lies inside the bracket. Reported world-wide (hour weighted)
#    and per design set point.
# 2. Chiller loop split: the chiller-mode plateau (hour-weighted over the chiller hours of the
#    panel, at each station's own design and pressure) with the inherited 24 K split and with
#    a conventional 5.5 K condenser range (T1 = T4 + 5.5), same set points.
# Outputs: results_world/closure_bound_stats.csv, results_world/closure_split_sensitivity.csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dcfootprint.core.modes as M                               # noqa: E402
from dcfootprint.core.modes import run_policy, LOOP_SPLIT      # noqa: E402

OUT = ROOT / "results_world"
t0 = time.time()
T_E = np.arange(-50.0, 50.5, 0.5); RH_E = np.arange(0.5, 101.5, 1.0)
T_C = (T_E[:-1] + T_E[1:]) / 2; RH_C = (RH_E[:-1] + RH_E[1:]) / 2
TT, RR = np.meshgrid(T_C, RH_C, indexing="ij")
GT, GR = TT.ravel(), RR.ravel()
P_STEP = 5000.0

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].set_index("STATION")      # noqa: E712
hist = {}; pgrp = {}
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "RH", "p_pa"]).to_pandas()
    st = d["STATION"].iloc[0]
    if st not in res.index:
        continue
    h, _, _ = np.histogram2d(d["T"], d["RH"].clip(0.5, 100.5), bins=[T_E, RH_E])
    hist[st] = h.ravel(); pgrp[st] = int(round(float(d["p_pa"].median()) / P_STEP))
print(f"histograms for {len(hist)} stations ({time.time()-t0:.0f}s)")

groups = {}
for st in hist:
    groups.setdefault((float(res.loc[st, "T4base"]), pgrp[st]), []).append(st)
print(f"{len(groups)} (design, pressure) groups")

rows = []; split_rows = []
for (t4b, pl), sts in groups.items():
    p = pl * P_STEP
    r = run_policy(GT, GR, T4base=t4b, p=p)
    mode = r["mode"]; T5 = r["T5"]; T1 = r["T1"]; ok = np.isfinite(T5)
    at_cap = ok & (np.abs(T5 - T1) < 1e-3)
    depth = np.where(ok & ~at_cap, T1 - T5, np.nan)
    # 5.5 K condenser range at the same set points (chiller cells only)
    M.LOOP_SPLIT = 5.5
    r55 = run_policy(GT, GR, T4base=t4b, p=p)
    M.LOOP_SPLIT = LOOP_SPLIT
    for st in sts:
        w = hist[st]; n = w.sum()
        row = dict(STATION=st, T4base=t4b, n_hours=n)
        for m, name in ((0, "dry"), (1, "evap"), (2, "chill")):
            sel = (mode == m) & ok
            wm = w[sel].sum()
            row[f"hours_{name}"] = wm
            row[f"cap_{name}"] = w[sel & at_cap].sum()
        sel = (mode == 2) & ok & ~at_cap
        row["chill_depth_wsum"] = float(np.nansum(w[sel] * depth[sel])); row["chill_inside_hours"] = w[sel].sum()
        # weighted median depth per station is not needed; keep the weighted histogram of depth in 0.5 K bins
        if w[sel].sum() > 0:
            dh, _ = np.histogram(depth[sel], bins=np.arange(0, 30.5, 0.5), weights=w[sel])
            row["depth_hist"] = ";".join(f"{x:.0f}" for x in dh)
        rows.append(row)
        if r55 is not None:
            selc = (mode == 2) & ok & np.isfinite(r55["T5"]) & (r55["mode"] == 2)
            ww = w[selc]
            if ww.sum() > 0:
                split_rows.append(dict(STATION=st, hours=ww.sum(),
                                       pue24=float((ww * r["PUE"][selc]).sum() / ww.sum()), wue24=float((ww * r["WUE"][selc]).sum() / ww.sum()),
                                       pue55=float((ww * r55["PUE"][selc]).sum() / ww.sum()), wue55=float((ww * r55["WUE"][selc]).sum() / ww.sum())))
S = pd.DataFrame(rows)
S.to_csv(OUT / "closure_bound_stations.csv", index=False)
tot = S[["hours_dry", "hours_evap", "hours_chill", "cap_dry", "cap_evap", "cap_chill", "chill_inside_hours"]].sum()
dh = np.zeros(60)
for s in S["depth_hist"].dropna():
    dh += np.array([float(x) for x in s.split(";")])
cum = np.cumsum(dh) / dh.sum(); med_depth = 0.5 * (np.searchsorted(cum, 0.5) + 0.5)
summ = [dict(scope="world", hours_chill=tot["hours_chill"], cap_chill_pct=100 * tot["cap_chill"] / tot["hours_chill"],
             cap_evap_pct=100 * tot["cap_evap"] / max(tot["hours_evap"], 1), cap_dry_pct=100 * tot["cap_dry"] / max(tot["hours_dry"], 1),
             depth_median_K=med_depth)]
for t4b, g in S.groupby("T4base"):
    hc = g["hours_chill"].sum()
    if hc > 0:
        summ.append(dict(scope=f"design {t4b:.0f}", hours_chill=hc, cap_chill_pct=100 * g["cap_chill"].sum() / hc,
                         cap_evap_pct=100 * g["cap_evap"].sum() / max(g["hours_evap"].sum(), 1),
                         cap_dry_pct=100 * g["cap_dry"].sum() / max(g["hours_dry"].sum(), 1), depth_median_K=np.nan))
su = pd.DataFrame(summ); su.to_csv(OUT / "closure_bound_stats.csv", index=False)
print(su.round(2).to_string(index=False))
if split_rows:
    sp = pd.DataFrame(split_rows)
    w = sp["hours"]
    out = dict(pue24=(w * sp["pue24"]).sum() / w.sum(), pue55=(w * sp["pue55"]).sum() / w.sum(),
               wue24=(w * sp["wue24"]).sum() / w.sum(), wue55=(w * sp["wue55"]).sum() / w.sum(), split24=LOOP_SPLIT)
    pd.DataFrame([out]).to_csv(OUT / "closure_split_sensitivity.csv", index=False)
    print("chiller plateau, 24 K vs 5.5 K split:", {k: round(float(v), 4) for k, v in out.items()})
else:
    print("run_policy has no split argument; split sensitivity skipped")
print(f"done ({time.time()-t0:.0f}s)")
