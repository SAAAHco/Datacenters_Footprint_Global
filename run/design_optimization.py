# Design and operation: operating policies, the water energy frontier, the chiller set point
# sweep and the direct liquid cooling archetype, all under the production policy rules
# (economizer sequence with approach temperatures, dcfootprint.core.modes).
#
# 1. Candidate mode maps per (design, pressure) group on the (T, RH) grid: dry free cooling
#    on 27/18 where T2 <= 18 - A_DRY, evaporative on 27/18 where Twb <= 18 - A_WET, and the
#    chiller at the lowest set point the approach allows, T4 = max(T4base, ceil(Twb + A_WET)),
#    climbing +1 C where the reversible solution is infeasible or above VCAP. The production
#    map itself is run_policy (dry preferred, then evaporative, then chiller).
# 2. Policies per station: production; exact energy optimum (per cell argmin PUE among the
#    feasible candidates); exact water optimum; and the frontier argmin(PUE + lambda WUE).
#    The water price lambda* at which the frontier returns to the production water use is
#    solved by bisection on the world mean (and per class), not read off a grid.
# 3. Chiller set point sweep: production policy with T4base from 26 to 34 C per class.
# 4. Direct liquid cooling archetype: 39/30 C loop (9 K split); dry where T2 <= 30 - A_DRY,
#    sprayed where Twb <= 30 - A_WET, warm loop climbing to T4 = 35 C, and the conventional
#    chiller (at its own reset set point) only as residual. Same tower physics and rules.
# Outputs: design_policies.csv, design_policy_summary.csv, design_frontier_stations.csv,
# design_frontier.csv, design_lambda_star.csv, design_t4_sweep.csv, dlc_archetype.csv.
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IN = ROOT / "results_world"
# DCF_OUT redirects the outputs (sensitivity runs, e.g. with DCF_DRY_PUMP_MPA set); inputs stay in results_world
OUT = Path(os.environ["DCF_OUT"]) if os.environ.get("DCF_OUT") else IN
OUT.mkdir(parents=True, exist_ok=True)

from dcfootprint.core.properties import V2_PARTIALP, twb_c                 # noqa: E402
from dcfootprint.core.tower import model_vec                               # noqa: E402
from dcfootprint.core.modes import (run_policy, VCAP, A_WET, A_DRY,        # noqa: E402
                                    T_SUPPLY, T_RETURN, LOOP_SPLIT)

t0 = time.time()
T_E = np.arange(-40.0, 50.5, 0.5); RH_E = np.arange(0.5, 101.5, 1.0)
T_C = (T_E[:-1] + T_E[1:]) / 2; RH_C = (RH_E[:-1] + RH_E[1:]) / 2
NT, NR = len(T_C), len(RH_C)
P_STEP = 2500.0
TT, RR = np.meshgrid(T_C, RH_C, indexing="ij")
GT, GR = TT.ravel(), RR.ravel()
NCELL = GT.size
H100 = np.full(NCELL, 100.0)
LAMBDAS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1e9]
MAX_RUNGS = 12

res = pd.read_csv(IN / "station_results.csv")
res = res[res["screen_pass"] == True].set_index("STATION")     # noqa: E712
dc = pd.read_csv(IN / "dc_classes.csv").set_index("STATION")
K = int(dc["dc_class"].max())

# ---------------------------------------------------- histograms and pressure per station
hist = {}; p_med = {}
pf = pq.ParquetFile(IN / "world_hourly.parquet")
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "RH", "p_pa"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        h, _, _ = np.histogram2d(g["T"], g["RH"], bins=[T_E, RH_E])
        hist[stn] = hist.get(stn, 0) + h.astype(np.float64)
        p_med.setdefault(stn, []).append(float(np.median(g["p_pa"])))
p_med = {k: float(np.median(v)) for k, v in p_med.items()}
print(f"histograms ({time.time()-t0:.0f}s)")


def mode_map(T1, T4, evap, chill, p):
    """Reversible solution of one loop on the grid; NaN where infeasible or above VCAP.
    T1 and T4 may be arrays over cells."""
    T5, WUE, PUEnoIT, Tdew, ex = model_vec(
        np.broadcast_to(np.asarray(T1, float), (NCELL,)), GT, GR, GT,
        np.broadcast_to(np.asarray(T4, float), (NCELL,)), H100,
        np.full(NCELL, evap), np.full(NCELL, chill), p=np.full(NCELL, p),
        props=V2_PARTIALP, extras=True)
    bad = ~np.isfinite(PUEnoIT) | (ex["VAmix5"] > VCAP)
    return np.where(bad, np.nan, PUEnoIT), np.where(bad, np.nan, WUE)


def chiller_candidate(t4b, p, twb, split=LOOP_SPLIT, t4_cap=None):
    """Chiller at the lowest set point the approach allows, climbing where infeasible."""
    T4 = np.maximum(float(t4b), np.ceil(twb + A_WET))
    P = np.full(NCELL, np.nan); W = np.full(NCELL, np.nan); T4used = np.full(NCELL, np.nan)
    for rung in range(MAX_RUNGS + 1):
        need = np.isnan(P)
        if t4_cap is not None:
            need &= T4 <= t4_cap
        if not need.any():
            break
        pm, wm = mode_map(T4 + split, T4, 1.0, 1.0, p)
        fill = need & np.isfinite(pm)
        P[fill] = pm[fill]; W[fill] = wm[fill]; T4used[fill] = T4[fill]
        T4 = np.where(need & ~fill, T4 + 1.0, T4)
    return P, W, T4used


def candidates(t4b, p):
    twb = twb_c(GT, GR, p)
    dP, dW = mode_map(T_RETURN, T_SUPPLY, 0.0, 0.0, p)
    dry_ok = GT <= T_SUPPLY - A_DRY
    dP = np.where(dry_ok, dP, np.nan); dW = np.where(dry_ok & np.isfinite(dP), 0.0, np.nan)
    eP, eW = mode_map(T_RETURN, T_SUPPLY, 1.0, 0.0, p)
    wet_ok = twb <= T_SUPPLY - A_WET
    eP = np.where(wet_ok, eP, np.nan); eW = np.where(wet_ok, eW, np.nan)
    cP, cW, cT4 = chiller_candidate(t4b, p, twb)
    r = run_policy(GT, GR, T4base=t4b, p=p, props=V2_PARTIALP, max_rungs=MAX_RUNGS)
    return dict(P3=np.stack([dP, eP, cP]), W3=np.stack([dW, eW, cW]),
                prodP=r["PUE"] - 1.0, prodW=r["WUE"], twb=twb)


groups = {}
for stn in res.index:
    if stn in hist:
        groups.setdefault((float(res.loc[stn, "T4base"]),
                           int(round(p_med[stn] / P_STEP))), []).append(stn)
maps = {key: candidates(key[0], key[1] * P_STEP) for key in groups}
print(f"{len(maps)} map groups built ({time.time()-t0:.0f}s)")


def policy_eval(f, m, which, lam=None):
    """Per cell mode choice among the feasible candidates, then climate weighted means.
    which: production | pueopt | wueopt | lam. Returns (PUE, WUE, occupancy of the three)."""
    P3, W3 = m["P3"], m["W3"]
    if which == "production":
        P, W = m["prodP"], m["prodW"]
        choice = None
    else:
        if which == "pueopt":
            score = P3
        elif which == "wueopt":
            score = W3 + 1e-9 * P3
        else:
            score = P3 + lam * W3
        score = np.where(np.isnan(score), np.inf, score)
        choice = np.argmin(score, axis=0)
        P = P3[choice, np.arange(NCELL)]; W = W3[choice, np.arange(NCELL)]
    ok = np.isfinite(P)
    w = np.where(ok, f, 0.0)
    tot = w.sum()
    if tot <= 0:
        return np.nan, np.nan, np.full(3, np.nan)
    w = w / tot
    occ = (np.array([float(w[(choice == k) & ok].sum()) for k in range(3)])
           if choice is not None else np.full(3, np.nan))
    return 1.0 + float(np.nansum(w * P)), float(np.nansum(w * W)), occ


# ------------------------------------------------------ per station policies + frontier
stn_f = {stn: (hist[stn] / hist[stn].sum()).ravel() for stn in hist}
stn_key = {stn: (float(res.loc[stn, "T4base"]), int(round(p_med[stn] / P_STEP))) for stn in hist}
rows = []; frows = []
for stn, f in stn_f.items():
    m = maps[stn_key[stn]]
    cls = int(dc.loc[stn, "dc_class"]) if stn in dc.index else -1
    pb, wb, _ = policy_eval(f, m, "production")
    pv, wv, occv = policy_eval(f, m, "pueopt")
    pw, ww, occw = policy_eval(f, m, "wueopt")
    rows.append(dict(STATION=stn, dc_class=cls,
                     PUE_record=float(res.loc[stn, "PUE"]), WUE_record=float(res.loc[stn, "WUE"]),
                     PUE_base=pb, WUE_base=wb,
                     PUE_pueopt=pv, WUE_pueopt=wv, occ_dry_pueopt=occv[0],
                     PUE_wueopt=pw, WUE_wueopt=ww, occ_dry_wueopt=occw[0]))
    for lam in LAMBDAS:
        pl, wl, _ = policy_eval(f, m, "lam", lam)
        frows.append(dict(STATION=stn, dc_class=cls, lam=lam, PUE=pl, WUE=wl))
pol = pd.DataFrame(rows); pol.to_csv(OUT / "design_policies.csv", index=False)
fr = pd.DataFrame(frows); fr.to_csv(OUT / "design_frontier_stations.csv", index=False)
print(f"policies evaluated ({time.time()-t0:.0f}s)")

print("\npolicy comparison (means over stations):")
summ = []
for scope, g in [("world", pol)] + [(f"class {c}", pol[pol["dc_class"] == c]) for c in range(1, K + 1)]:
    summ.append(dict(scope=scope, PUE_record=g["PUE_record"].mean(), WUE_record=g["WUE_record"].mean(),
                     PUE_base=g["PUE_base"].mean(), WUE_base=g["WUE_base"].mean(),
                     PUE_pueopt=g["PUE_pueopt"].mean(), WUE_pueopt=g["WUE_pueopt"].mean(),
                     occ_dry_pueopt=g["occ_dry_pueopt"].mean(),
                     PUE_wueopt=g["PUE_wueopt"].mean(), WUE_wueopt=g["WUE_wueopt"].mean()))
su = pd.DataFrame(summ); su.to_csv(OUT / "design_policy_summary.csv", index=False)
print(su.round(4).to_string(index=False))

frontier = fr.groupby(["dc_class", "lam"])[["PUE", "WUE"]].mean().reset_index()
frontier.to_csv(OUT / "design_frontier.csv", index=False)
print("\nworld frontier (lambda sweep, means):")
print(fr.groupby("lam")[["PUE", "WUE"]].mean().round(4).to_string())


# ------------------------------------------------------ lambda*: solved, not read off a grid
def world_frontier(lam, stns):
    P = []; W = []
    for stn in stns:
        pl, wl, _ = policy_eval(stn_f[stn], maps[stn_key[stn]], "lam", lam)
        P.append(pl); W.append(wl)
    return float(np.nanmean(P)), float(np.nanmean(W))


lam_rows = []
for scope, stns in [("world", list(stn_f))] + [
        (f"class {c}", [s for s in stn_f if s in dc.index and int(dc.loc[s, "dc_class"]) == c])
        for c in range(1, K + 1)]:
    target = float(np.nanmean([pol.set_index("STATION").loc[s, "WUE_base"] for s in stns]))
    lo, hi = 0.0, 1.0
    p_hi, w_hi = world_frontier(hi, stns)
    if w_hi > target:        # even a very high price does not recover the production water
        lam_rows.append(dict(scope=scope, lambda_star=np.nan, PUE_at=p_hi, WUE_at=w_hi, WUE_target=target))
        continue
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        _, w_mid = world_frontier(mid, stns)
        if w_mid > target:
            lo = mid
        else:
            hi = mid
    p_at, w_at = world_frontier(hi, stns)
    p0, w0 = world_frontier(0.0, stns)
    lam_rows.append(dict(scope=scope, lambda_star=hi, PUE_at=p_at, WUE_at=w_at, WUE_target=target,
                         PUE_energy_only=p0, WUE_energy_only=w0,
                         energy_gain_recovered_pct=100 * (p0 - p_at) / (p0 - su.set_index("scope").loc[scope, "PUE_base"])
                         if (p0 - su.set_index("scope").loc[scope, "PUE_base"]) != 0 else np.nan))
ls = pd.DataFrame(lam_rows); ls.to_csv(OUT / "design_lambda_star.csv", index=False)
print("\nwater price at which the frontier returns to the production water use:")
print(ls.round(5).to_string(index=False))

# ------------------------------------------------------ T4 sweep per class
cls_hist = {c: 0 for c in range(1, K + 1)}
cls_p = {c: [] for c in range(1, K + 1)}
for stn, h in hist.items():
    if stn in dc.index:
        c = int(dc.loc[stn, "dc_class"])
        cls_hist[c] = cls_hist[c] + h
        cls_p[c].append(p_med[stn])
sweep = []
for c in range(1, K + 1):
    f = (cls_hist[c] / cls_hist[c].sum()).ravel()
    p = float(np.median(cls_p[c]))
    for t4 in range(26, 35):
        r = run_policy(GT, GR, T4base=float(t4), p=p, props=V2_PARTIALP, max_rungs=MAX_RUNGS)
        ok = np.isfinite(r["PUE"]); w = np.where(ok, f, 0.0); w = w / w.sum()
        sweep.append(dict(dc_class=c, T4=t4, PUE=float(np.nansum(w * r["PUE"])),
                          WUE=float(np.nansum(w * r["WUE"])),
                          reset_share=float(w[(r["mode"] == 2) & (r["reset"] > 0)].sum())))
sw = pd.DataFrame(sweep); sw.to_csv(OUT / "design_t4_sweep.csv", index=False)
print("\nchiller set point sweep (PUE at T4base = 26 / 30 / 34 per class):")
for c in range(1, K + 1):
    g = sw[sw["dc_class"] == c].set_index("T4")
    print(f"  class {c}: PUE {g.loc[26,'PUE']:.4f} / {g.loc[30,'PUE']:.4f} / {g.loc[34,'PUE']:.4f}   "
          f"WUE {g.loc[26,'WUE']:.4f} / {g.loc[30,'WUE']:.4f} / {g.loc[34,'WUE']:.4f}")

# ------------------------------------------------------ direct liquid cooling archetype
DLC_T4, DLC_SPLIT, DLC_T4_MAX = 30.0, 9.0, 35.0
dlc_maps = {}
for plev in sorted(set(k[1] for k in groups)):
    p = plev * P_STEP
    twb = twb_c(GT, GR, p)
    dfP, dfW = mode_map(DLC_T4 + DLC_SPLIT, DLC_T4, 0.0, 0.0, p)
    dry_ok = GT <= DLC_T4 - A_DRY
    dfP = np.where(dry_ok, dfP, np.nan); dfW = np.where(dry_ok & np.isfinite(dfP), 0.0, np.nan)
    # warm sprayed loop, set point from the approach, climbing to DLC_T4_MAX
    T4 = np.maximum(DLC_T4, np.ceil(twb + A_WET))
    P = np.full(NCELL, np.nan); W = np.full(NCELL, np.nan)
    for rung in range(MAX_RUNGS):
        need = np.isnan(P) & (T4 <= DLC_T4_MAX)
        if not need.any():
            break
        pm, wm = mode_map(T4 + DLC_SPLIT, T4, 1.0, 0.0, p)
        fill = need & np.isfinite(pm)
        P[fill] = pm[fill]; W[fill] = wm[fill]
        T4 = np.where(need & ~fill, T4 + 1.0, T4)
    cP, cW, _ = chiller_candidate(26.0, p, twb)      # residual conventional chiller
    wet_P = np.where(np.isnan(P), cP, P); wet_W = np.where(np.isnan(P), cW, W)
    useP = np.where(np.isfinite(dfP), dfP, wet_P)
    useW = np.where(np.isfinite(dfP), dfW, wet_W)
    chill_used = np.isnan(dfP) & np.isnan(P) & np.isfinite(cP)
    dlc_maps[plev] = (useP, useW, np.isfinite(dfP), chill_used)
dlc_rows = []
for stn, f in stn_f.items():
    plev = stn_key[stn][1]
    useP, useW, dry, chill_used = dlc_maps[plev]
    ok = np.isfinite(useP); w = np.where(ok, f, 0.0)
    if w.sum() <= 0:
        continue
    w = w / w.sum()
    dlc_rows.append(dict(STATION=stn,
                         dc_class=int(dc.loc[stn, "dc_class"]) if stn in dc.index else -1,
                         PUE_dlc=1 + float(np.nansum(w * useP)),
                         WUE_dlc=float(np.nansum(w * useW)),
                         dry_share=float(w[dry].sum()), chiller_share=float(w[chill_used].sum()),
                         PUE_base=float(res.loc[stn, "PUE"]),
                         WUE_base=float(res.loc[stn, "WUE"])))
dl = pd.DataFrame(dlc_rows); dl.to_csv(OUT / "dlc_archetype.csv", index=False)
print(f"\ndirect liquid cooling archetype (loop 39/30, dry below {DLC_T4 - A_DRY:.0f} C, "
      f"sprayed up to Twb {DLC_T4 - A_WET:.0f} C):")
for scope, g in [("world", dl)] + [(f"class {c}", dl[dl["dc_class"] == c]) for c in range(1, K + 1)]:
    print(f"  {scope}: PUE {g['PUE_dlc'].mean():.4f} (base {g['PUE_base'].mean():.4f})  "
          f"WUE {g['WUE_dlc'].mean():.4f} (base {g['WUE_base'].mean():.4f})  "
          f"dry share {100 * g['dry_share'].mean():.0f}%  chiller share {100 * g['chiller_share'].mean():.1f}%")
print(f"[done] {time.time()-t0:.0f}s")
