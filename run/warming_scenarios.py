# Tier 2 package C: warming scenarios by density shift.
# Each station's cleaned (T, RH) climate density is shifted by +1, +2, +3 C under two
# humidity conventions: constant relative humidity (the density moves along T) and constant
# dew point (RH falls as pvsat(T) rises; each cell is remapped). The shifted densities are
# integrated against footprint and mode maps precomputed with the production model and the
# station's OWN CURRENT design set point, so the results describe existing designs under a
# warmer climate. Interpretable outputs: mean footprint and mode occupancy per scenario,
# design margin erosion (stations whose p99.5 dew point would exceed the design rule), and
# class migration under the planner rule (rule versus rule, so rule error cancels).
# Outputs: results_world/warming_scenarios.csv, warming_summary.csv, printed summary.
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_world"

from dcfootprint.core.properties import V2_PARTIALP, pvsat_pa, tdew_c, twb_c   # noqa: E402
from dcfootprint.core.modes import run_policy, A_WET                          # noqa: E402

t0 = time.time()
T_E = np.arange(-40.0, 54.0, 0.5)          # extended to 53.5 so +3 C stays on grid
RH_E = np.arange(0.5, 101.5, 1.0)
T_C = (T_E[:-1] + T_E[1:]) / 2; RH_C = (RH_E[:-1] + RH_E[1:]) / 2
NT, NR = len(T_C), len(RH_C)
P_STEP = 2500.0
DELTAS = [1.0, 2.0, 3.0]

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].set_index("STATION")   # noqa: E712
dc = pd.read_csv(OUT / "dc_classes.csv").set_index("STATION")
K = int(dc["dc_class"].max())
mon = pd.read_csv(OUT / "station_monthly.csv")
t_range = mon.groupby("STATION")["T"].agg(lambda x: x.max() - x.min())

# ---------------------------------------------------- per station histograms + pressure
hist = {}; p_med = {}
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "RH", "p_pa"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        h, _, _ = np.histogram2d(g["T"], g["RH"], bins=[T_E, RH_E])
        hist[stn] = hist.get(stn, 0) + h.astype(np.float64)
        p_med.setdefault(stn, []).append(float(np.median(g["p_pa"])))
p_med = {k: float(np.median(v)) for k, v in p_med.items()}
print(f"histograms for {len(hist)} stations ({time.time()-t0:.0f}s)")

# ---------------------------------------------------- footprint and mode maps per group
TT, RR = np.meshgrid(T_C, RH_C, indexing="ij")
groups = {}
for stn in res.index:
    if stn in hist:
        groups.setdefault((float(res.loc[stn, "T4base"]),
                           int(round(p_med[stn] / P_STEP))), []).append(stn)
gmaps = {}
twb_by_plev = {}
for key in groups:
    t4b, plev = key
    r = run_policy(TT.ravel(), RR.ravel(), T4base=t4b, p=plev * P_STEP,
                   props=V2_PARTIALP, max_rungs=12)
    ok = np.isfinite(r["PUE"])
    # four operating states: dry, evaporative, chiller at design, chiller with reset
    state = np.where((r["mode"] == 2) & (r["reset"] > 0), 3, r["mode"])
    if plev not in twb_by_plev:
        twb_by_plev[plev] = twb_c(TT.ravel(), RR.ravel(), plev * P_STEP)
    gmaps[key] = dict(PUE=r["PUE"], WUE=r["WUE"], mode=state, ok=ok, twb=twb_by_plev[plev])
print(f"{len(gmaps)} footprint maps ({time.time()-t0:.0f}s)")

# dew point of each grid cell (planner rule normals) and remap indices per scenario; the
# design rule itself uses the wet bulb of the cell at the group's pressure
TD_CELL = np.where(RR.ravel() >= 100.0, TT.ravel(),
                   tdew_c(RR.ravel() / 100.0 * pvsat_pa(TT.ravel())))

remaps = {}
for dT in DELTAS:
    s = int(round(dT / 0.5))
    # constant RH: pure shift along T
    remaps[("constRH", dT)] = ("shift", s)
    # constant dew point: cell (i, j) -> (i + s, j') with RH' = RH pvsat(T)/pvsat(T + dT)
    src_i, src_j = np.meshgrid(np.arange(NT), np.arange(NR), indexing="ij")
    Ti = T_C[src_i]; RHj = RH_C[src_j]
    RHn = RHj * pvsat_pa(Ti) / pvsat_pa(Ti + dT)
    jn = np.clip(np.round(RHn).astype(int) - 1, 0, NR - 1)
    inn = np.clip(src_i + s, 0, NT - 1)
    remaps[("constTd", dT)] = ("map", inn.ravel(), jn.ravel())


def shifted(f, key):
    kind = remaps[key][0]
    if kind == "shift":
        s = remaps[key][1]
        out = np.zeros_like(f)
        out[s:, :] = f[:-s, :]
        out[-1, :] += f[-s:, :].sum(0)   # clip guard; empty in practice
        return out
    _, inn, jn = remaps[key]
    out = np.zeros_like(f)
    np.add.at(out, (inn, jn), f.ravel())
    return out


def integrate(fflat, gm):
    w = fflat * gm["ok"]
    tot = w.sum()
    if tot <= 0:
        return dict(PUE=np.nan, WUE=np.nan, occ=np.full(4, np.nan), lost=1.0)
    w = w / tot
    occ = np.array([float(w[gm["mode"] == m].sum()) for m in range(4)])
    return dict(PUE=float(np.nansum(w * gm["PUE"])), WUE=float(np.nansum(w * gm["WUE"])),
                occ=occ, lost=float(1.0 - tot / fflat.sum()))


def wq(fflat, cell, q=0.995):
    """Weighted quantile of a per cell quantity under the density fflat."""
    order = np.argsort(cell)
    cw = np.cumsum(fflat[order]); cw = cw / cw[-1]
    return float(cell[order][min(np.searchsorted(cw, q), cell.size - 1)])


SCEN = [("base", 0.0)] + [(v, d) for v in ("constRH", "constTd") for d in DELTAS]
rows = []
for stn, f in hist.items():
    key = (float(res.loc[stn, "T4base"]), int(round(p_med[stn] / P_STEP)))
    gm = gmaps[key]
    fbase = f / f.sum()
    tmean0 = float((fbase.sum(1) * T_C).sum())
    for name, dT in SCEN:
        ff = fbase if name == "base" else shifted(fbase, (name, dT))
        fflat = ff.ravel()
        out = integrate(fflat, gm)
        twbp = wq(fflat, gm["twb"])
        td_mean = float((fflat * TD_CELL).sum() / fflat.sum())
        rows.append(dict(STATION=stn, scenario=name, dT=dT,
                         PUE=out["PUE"], WUE=out["WUE"],
                         occ_free=out["occ"][0], occ_evap=out["occ"][1],
                         occ_chill=out["occ"][2], occ_esc=out["occ"][3],
                         lost_mass=out["lost"], Twb_p995=twbp,
                         T4_needed=max(26.0, np.floor(twbp + A_WET) + 1.0),
                         T_mean=tmean0 + dT, Td_mean=td_mean))
wm = pd.DataFrame(rows)
wm["dc_class"] = wm["STATION"].map(dc["dc_class"])
wm["T4base"] = wm["STATION"].map(res["T4base"])
wm["T_range"] = wm["STATION"].map(t_range)
wm.to_csv(OUT / "warming_scenarios.csv", index=False)
print(f"scenario table written ({time.time()-t0:.0f}s)")

# --------------------------------------------------------- validation of the machinery
base = wm[wm["scenario"] == "base"].set_index("STATION")
joined = base.join(res[["PUE", "WUE"]], rsuffix="_prod")
dP = 100 * (joined["PUE"] - joined["PUE_prod"]).abs() / joined["PUE_prod"]
dW = 100 * (joined["WUE"] - joined["WUE_prod"]).abs() / joined["WUE_prod"]
print(f"base scenario vs production record level: median |dPUE| {dP.median():.2f}%  "
      f"|dWUE| {dW.median():.2f}%  (convolution bias, expected small)")

# --------------------------------------------------------- summaries
summ = []
for (name, dT), g in wm.groupby(["scenario", "dT"]):
    summ.append(dict(scenario=name, dT=dT, PUE=g["PUE"].mean(), WUE=g["WUE"].mean(),
                     occ_free=100 * g["occ_free"].mean(), occ_evap=100 * g["occ_evap"].mean(),
                     occ_chill=100 * g["occ_chill"].mean(), occ_esc=100 * g["occ_esc"].mean(),
                     design_eroded=int((g["T4_needed"] > g["T4base"]).sum()),
                     lost_mass_max=float(g["lost_mass"].max())))
su = pd.DataFrame(summ).sort_values(["scenario", "dT"])
su.to_csv(OUT / "warming_summary.csv", index=False)
print("\nworld means by scenario (2,081 stations, current designs):")
print(su.round(4).to_string(index=False))

for cls in range(1, K + 1):
    g = wm[wm["dc_class"] == cls]
    row = []
    for (name, dT) in [("base", 0.0), ("constRH", 2.0), ("constTd", 2.0)]:
        gg = g[(g["scenario"] == name) & (g["dT"] == dT)]
        row.append(f"{name}+{dT:.0f}: PUE {gg['PUE'].mean():.4f} WUE {gg['WUE'].mean():.4f}")
    print(f"class {cls}: " + "  |  ".join(row))

# --------------------------------------------------------- class migration, planner rule
cent = wm[wm["scenario"] == "base"].groupby("dc_class")[["T_mean", "Td_mean", "T_range"]].mean()
sd = wm[wm["scenario"] == "base"][["T_mean", "Td_mean", "T_range"]].std()


def rule_assign(g):
    Zs = (g[["T_mean", "Td_mean", "T_range"]] / sd).to_numpy()
    Zc = (cent / sd).to_numpy()
    return cent.index.to_numpy()[np.argmin(((Zs[:, None, :] - Zc[None, :, :]) ** 2).sum(2), 1)]


base_lab = pd.Series(rule_assign(wm[wm["scenario"] == "base"]),
                     index=wm[wm["scenario"] == "base"]["STATION"])
print("\nclass migration under the planner rule (rule vs rule), share of stations moving:")
for name in ("constRH", "constTd"):
    for dT in DELTAS:
        g = wm[(wm["scenario"] == name) & (wm["dT"] == dT)]
        lab = rule_assign(g)
        moved = (lab != base_lab.loc[g["STATION"]].to_numpy())
        up = (lab > base_lab.loc[g["STATION"]].to_numpy())
        print(f"  {name} +{dT:.0f}C: moved {100 * moved.mean():.1f}%  "
              f"(toward hotter classes {100 * up.mean():.1f}%)")
        if dT == 2.0:
            mig = pd.crosstab(base_lab.loc[g["STATION"]].to_numpy(), lab,
                              rownames=["base"], colnames=[f"{name}+2C"])
            print(mig.to_string())
print(f"[done] {time.time()-t0:.0f}s")
