# Closure robustness: the reversible (zero entropy production) closure of the production
# model against the Merkel finite transfer closure of the fill literature, at every station
# and in every sprayed mode, for a sweep of tower sizes (Merkel numbers).
#
# Both closures run the same operating policy (economizer sequence with approach
# temperatures, dcfootprint.core.modes) on the (T, RH) grid of each (design, pressure)
# group; the Merkel maps replace the sprayed cells by the finite tower solution of
# dcfootprint.core.closure_merkel and climb the same ladder where the finite tower cannot
# hold the loop. Station means follow by integrating each station's own climate histogram
# against the maps, so the comparison is closure against closure with the convolution bias
# cancelled. Nothing is calibrated on the reversible model: Me is prescribed.
# Outputs: results_world/closure_merkel.csv (per station per Me), closure_merkel_summary.csv
# (world and class means, per mode) and the printed summary.
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_world"

from dcfootprint.core.properties import V2_PARTIALP                       # noqa: E402
from dcfootprint.core.modes import (run_policy, select_mode_economizer,    # noqa: E402
                                    VCAP, A_WET, LOOP_SPLIT)
from dcfootprint.core.closure_merkel import merkel_sprayed, pue_from       # noqa: E402

t0 = time.time()
MES = [0.75, 1.0, 1.5, 2.0, 3.0]
T_E = np.arange(-40.0, 50.5, 0.5); RH_E = np.arange(0.5, 101.5, 1.0)
T_C = (T_E[:-1] + T_E[1:]) / 2; RH_C = (RH_E[:-1] + RH_E[1:]) / 2
P_STEP = 2500.0
TT, RR = np.meshgrid(T_C, RH_C, indexing="ij")
GT, GR = TT.ravel(), RR.ravel()
NCELL = GT.size
MAX_RUNGS = 12

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].set_index("STATION")     # noqa: E712
dc = pd.read_csv(OUT / "dc_classes.csv").set_index("STATION")
K = int(dc["dc_class"].max())

# ---------------------------------------------------- histograms and pressure per station
hist = {}; p_med = {}
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "RH", "p_pa"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        h, _, _ = np.histogram2d(g["T"], g["RH"], bins=[T_E, RH_E])
        hist[stn] = hist.get(stn, 0) + h.astype(np.float64)
        p_med.setdefault(stn, []).append(float(np.median(g["p_pa"])))
p_med = {k: float(np.median(v)) for k, v in p_med.items()}
print(f"histograms ({time.time()-t0:.0f}s)")

groups = {}
for stn in res.index:
    if stn in hist:
        groups.setdefault((float(res.loc[stn, "T4base"]),
                           int(round(p_med[stn] / P_STEP))), []).append(stn)


def merkel_map(t4b, p, Me, rev):
    """Production policy on the grid, closed by the Merkel tower of size Me. Dry cells keep
    the reversible solution; sprayed cells that the finite tower cannot serve climb the
    ladder (evaporative -> chiller at the reset set point, chiller -> +1 C)."""
    T1, T4, evap, chill, mode, twb = select_mode_economizer(GT, GR, p, T4base=t4b)
    T1 = T1.copy(); T4 = T4.copy(); chill = chill.copy(); mode = mode.copy()
    PUE = rev["PUE"].copy(); WUE = rev["WUE"].copy()
    state = rev["state"].copy()
    spr = evap == 1
    m = merkel_sprayed(T1[spr], GT[spr], GR[spr], T4[spr], p, Me)
    ok = m["feasible"] & (m["VAmix5"] <= VCAP)
    PUE[spr] = np.where(ok, pue_from(m["fan"], m["pump"], chill[spr], T4[spr]), np.nan)
    WUE[spr] = np.where(ok, m["WUE"], np.nan)
    T4c = np.maximum(t4b, np.ceil(twb + A_WET))
    for rung in range(MAX_RUNGS):
        need = spr & ~np.isfinite(PUE)
        if not need.any():
            break
        idx = np.where(need)[0]
        was_evap = mode[idx] == 1
        T4[idx] = np.where(was_evap, T4c[idx], T4[idx] + 1.0)
        T1[idx] = T4[idx] + LOOP_SPLIT
        chill[idx] = 1.0; mode[idx] = 2
        m = merkel_sprayed(T1[idx], GT[idx], GR[idx], T4[idx], p, Me)
        ok = m["feasible"] & (m["VAmix5"] <= VCAP)
        PUE[idx] = np.where(ok, pue_from(m["fan"], m["pump"], 1.0, T4[idx]), np.nan)
        WUE[idx] = np.where(ok, m["WUE"], np.nan)
    state = np.where(mode == 2, np.where(T4 > t4b, 3, 2), mode)
    return dict(PUE=PUE, WUE=WUE, state=state)


maps = {}
for gi, ((t4b, plev), stns) in enumerate(groups.items()):
    p = plev * P_STEP
    r = run_policy(GT, GR, T4base=t4b, p=p, props=V2_PARTIALP, max_rungs=MAX_RUNGS)
    rev = dict(PUE=r["PUE"], WUE=r["WUE"],
               state=np.where((r["mode"] == 2) & (r["reset"] > 0), 3, r["mode"]))
    maps[(t4b, plev)] = {"rev": rev}
    for Me in MES:
        maps[(t4b, plev)][Me] = merkel_map(t4b, p, Me, rev)
    if (gi + 1) % 10 == 0:
        print(f"  {gi+1}/{len(groups)} map groups ({time.time()-t0:.0f}s)", flush=True)
print(f"{len(maps)} map groups built ({time.time()-t0:.0f}s)")


def station_means(f, mp):
    ok = np.isfinite(mp["PUE"])
    w = np.where(ok, f, 0.0)
    lost = 1.0 - w.sum()
    w = w / w.sum()
    out = dict(PUE=float(np.nansum(w * mp["PUE"])), WUE=float(np.nansum(w * mp["WUE"])), lost=lost)
    for k, name in ((1, "evap"), (2, "chill"), (3, "reset")):
        sel = ok & (mp["state"] == k)
        ws = w[sel].sum()
        out[f"PUE_{name}"] = float((w[sel] * mp["PUE"][sel]).sum() / ws) if ws > 0 else np.nan
        out[f"WUE_{name}"] = float((w[sel] * mp["WUE"][sel]).sum() / ws) if ws > 0 else np.nan
        out[f"occ_{name}"] = float(ws)
    out["occ_dry"] = float(w[ok & (mp["state"] == 0)].sum())
    return out


rows = []
for (t4b, plev), stns in groups.items():
    for stn in stns:
        f = hist[stn].ravel(); f = f / f.sum()
        base = station_means(f, maps[(t4b, plev)]["rev"])
        for Me in MES:
            mm = station_means(f, maps[(t4b, plev)][Me])
            rows.append(dict(STATION=stn, dc_class=int(dc.loc[stn, "dc_class"]) if stn in dc.index else -1,
                             Me=Me, PUE_rev=base["PUE"], WUE_rev=base["WUE"],
                             PUE_me=mm["PUE"], WUE_me=mm["WUE"], lost_me=mm["lost"],
                             PUE_evap_rev=base["PUE_evap"], PUE_evap_me=mm["PUE_evap"],
                             WUE_evap_rev=base["WUE_evap"], WUE_evap_me=mm["WUE_evap"],
                             PUE_chill_rev=base["PUE_chill"], PUE_chill_me=mm["PUE_chill"],
                             WUE_chill_rev=base["WUE_chill"], WUE_chill_me=mm["WUE_chill"],
                             occ_reset_rev=base["occ_reset"], occ_reset_me=mm["occ_reset"],
                             occ_evap_rev=base["occ_evap"], occ_evap_me=mm["occ_evap"]))
cc = pd.DataFrame(rows)
cc["dPUE"] = cc["PUE_me"] - cc["PUE_rev"]
cc["dWUE"] = cc["WUE_me"] - cc["WUE_rev"]
cc["dPUE_pct"] = 100 * cc["dPUE"] / cc["PUE_rev"]
cc["dWUE_pct"] = 100 * cc["dWUE"] / cc["WUE_rev"]
cc.to_csv(OUT / "closure_merkel.csv", index=False)

summ = []
for Me, g in cc.groupby("Me"):
    for scope, gg in [("world", g)] + [(f"class {c}", g[g["dc_class"] == c]) for c in range(1, K + 1)]:
        summ.append(dict(Me=Me, scope=scope, n=len(gg),
                         PUE_rev=gg["PUE_rev"].mean(), PUE_me=gg["PUE_me"].mean(),
                         WUE_rev=gg["WUE_rev"].mean(), WUE_me=gg["WUE_me"].mean(),
                         dPUE_mean=gg["dPUE"].mean(), dPUE_pct_median=gg["dPUE_pct"].median(),
                         dPUE_pct_p95abs=gg["dPUE_pct"].abs().quantile(0.95),
                         dWUE_mean=gg["dWUE"].mean(), dWUE_pct_median=gg["dWUE_pct"].median(),
                         dWUE_pct_p95abs=gg["dWUE_pct"].abs().quantile(0.95),
                         PUE_evap_rev=gg["PUE_evap_rev"].mean(), PUE_evap_me=gg["PUE_evap_me"].mean(),
                         WUE_evap_rev=gg["WUE_evap_rev"].mean(), WUE_evap_me=gg["WUE_evap_me"].mean(),
                         PUE_chill_rev=gg["PUE_chill_rev"].mean(), PUE_chill_me=gg["PUE_chill_me"].mean(),
                         WUE_chill_rev=gg["WUE_chill_rev"].mean(), WUE_chill_me=gg["WUE_chill_me"].mean(),
                         occ_reset_rev=gg["occ_reset_rev"].mean(), occ_reset_me=gg["occ_reset_me"].mean(),
                         occ_evap_rev=gg["occ_evap_rev"].mean(), occ_evap_me=gg["occ_evap_me"].mean(),
                         lost_me_max=gg["lost_me"].max()))
su = pd.DataFrame(summ)
su.to_csv(OUT / "closure_merkel_summary.csv", index=False)
pd.set_option("display.width", 250)
print("\nMerkel closure against the reversible closure (station means; world and classes):")
print(su[["Me", "scope", "PUE_rev", "PUE_me", "WUE_rev", "WUE_me", "dPUE_pct_median",
          "dPUE_pct_p95abs", "dWUE_pct_median", "dWUE_pct_p95abs", "occ_reset_rev", "occ_reset_me"]]
      .round(4).to_string(index=False))
print("\nper mode plateaus (world means): evaporative and chiller at design set point")
print(su[su["scope"] == "world"][["Me", "PUE_evap_rev", "PUE_evap_me", "WUE_evap_rev", "WUE_evap_me",
                                  "PUE_chill_rev", "PUE_chill_me", "WUE_chill_rev", "WUE_chill_me"]]
      .round(4).to_string(index=False))
print(f"[done] {time.time()-t0:.0f}s")
