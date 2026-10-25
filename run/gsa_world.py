# Tier 1 point 2: global sensitivity analysis re-run under the production configuration
# (v2 Table 6 fits with the partial pressure entropy treatment, corrected auxiliary term),
# evaluated against the world climate, with uncertainty propagation per DC class.
#
# Follows Old_analysis/cooling_tower_param.py and gsa_sensitivity.py: nine parameters
# (COP c0/c1/c2 at +-10/20/20 percent, design dp air +-50 percent and dp loop +-30 percent,
# and reference consistent property scalings of cp water, latent heat, cp air and ds vapour
# at +-0.5 percent, pinned at 20 C so the enthalpy datum is unchanged). Sobol indices by
# Saltelli sampling (base 1024, 11,264 evaluations, first and total order), and uncertainty
# by 4,096 Latin hypercube draws. Each evaluation is the climate weighted mean footprint
# over the pooled (T, RH) histograms of the five DC classes (1 C by 2 percent cells, 99.9
# percent of climatic mass, class modal design set point, class mean station pressure);
# the world value is the station count weighted mean of the class values. The escalation
# ladder is not applied (as in the original GSA); baseline offsets versus the record level
# production run are reported so the intervals can be applied around the production means.
#
# Outputs: results_world/gsa_world_sobol.csv, gsa_world_ci.csv, and the printed summary.
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_world"

from dcfootprint.core.properties import RA, RV, K0, EPS_W, pvsat_pa, twb_c   # noqa: E402
from dcfootprint.core.modes import select_mode_economizer, A_WET, A_DRY      # noqa: E402
from dcfootprint.core.tower import LEGACY_PHYSICS                            # noqa: E402

t0 = time.time()

# ------------------------------------------------------- parametrized v2pp tower model
# base V2 fits (Celsius forms)
def hL0(T): return 4.1887 * T
def hV0(T): return 2500.4620 + 1.8784 * T + 1.2279e-3 * T**2
def hA0(T): return 399.2857 + 1.0074 * T
def sL0(T): return 1.5153e-2 * T - 2.1274e-5 * T**2
def sV0(T): return 9.1540 - 2.6506e-2 * T + 1.1160e-4 * T**2 - 2.6642e-7 * T**3

def sA0(T, pa):
    pM = pa / 1e6
    return (4.3983 + 3.6069e-3 * T - 4.7568e-6 * T**2
            - 1.0982e1 * pM + 6.7642e1 * pM**2 - 1.7960e2 * pM**3)

# eleven parameters: the nine of the regional study plus the two approach temperatures of
# the production policy (wet bulb approach of the sprayed modes, dry bulb approach of the
# dry mode), which set the mode occupancy and the condenser water reset
P0 = np.array([6.88, 0.18, 0.12, 250.0, 0.21e6, 1.0, 1.0, 1.0, 1.0, A_WET, A_DRY])
NAMES = ["COP c0", "COP c1 (Tchw)", "COP c2 (T4)", "dp air", "dp loop",
         "cp water", "latent heat", "cp air", "ds vapour", "approach wet", "approach dry"]
BOUNDS = [[6.19, 7.57], [0.144, 0.216], [0.096, 0.144],
          [125.0, 375.0], [0.147e6, 0.273e6],
          [0.995, 1.005], [0.995, 1.005], [0.995, 1.005], [0.995, 1.005],
          [A_WET - 1.0, A_WET + 1.0], [A_DRY - 2.0, A_DRY + 2.0]]


def model_param(par, T1, T2, HR2, T4, evap, chill, p, n_iter=40):
    """v2pp model with the perturbable physical parameters (first nine entries of par).
    Returns (WUE, PUEnoIT) per cell; NaN where infeasible (condensation or unsatisfiable
    entropy balance)."""
    cop0, cop1, cop2, dpA, dpL, cpw, Lvap, cpa, dsv = par[:9]
    Tref = 20.0
    aL1 = 4.1887 * cpw
    hL = lambda T: hL0(Tref) + aL1 * (T - Tref)
    aA1 = 1.0074 * cpa
    hA = lambda T: hA0(Tref) + aA1 * (T - Tref)
    hV = lambda T: hL(T) + Lvap * (hV0(T) - hL0(T))
    sVsat = lambda T: sL0(T) + dsv * (sV0(T) - sL0(T))

    def sV(T, pV):
        with np.errstate(divide="ignore", invalid="ignore"):
            return sVsat(T) - (RV / 1000.0) * np.log(pV / pvsat_pa(T))

    def sA(T, p_, pV):
        return sA0(T, p_ - pV)

    T3 = T2
    hL1 = hL(T1); sL1 = sL0(T1)
    hV2 = hV(T2); hA2 = hA(T2)
    pV2 = HR2 / 100 * pvsat_pa(T2)
    w2 = EPS_W * pV2 / (p - pV2)
    sV2 = sV(T2, pV2); sA2 = sA(T2, p, pV2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Tdew2 = np.where(HR2 == 100, T2, 237.3 / (17.27 / np.log(pV2 / 610.78) - 1))
    hL3 = hL(T3); sL3 = sL0(T3)
    hL4 = hL(T4); sL4 = sL0(T4)
    copv = cop0 + cop1 * 18 - cop2 * T4
    if LEGACY_PHYSICS:
        Qt = np.ones_like(T2)
        T5max = np.full_like(T2, 50.0); T5min = T2.copy()
    else:   # production physics, as in tower.model_vec: condenser load, bounded outlet
        Qt = 1.0 + np.asarray(chill, float) / copv
        T5max = np.broadcast_to(np.asarray(T1, float), T2.shape).copy()
        T5min = np.minimum(T2, np.broadcast_to(np.asarray(T4, float), T2.shape))
    mL1 = Qt / (hL1 - hL4); mL4 = mL1

    def ev(T5):
        hV5 = hV(T5); hA5 = hA(T5)
        pV5e = pvsat_pa(T5)
        w5e = EPS_W * pV5e / (p - pV5e)
        w5 = np.where(evap == 0, w2, w5e)
        pV5 = np.where(evap == 0, p * w2 / (EPS_W + w2), pV5e)
        sV5 = sV(T5, pV5); sA5 = sA(T5, p, pV5)
        mA = mL1 * (hL1 - hL4) / ((hA5 + w5 * hV5) - (hA2 + w2 * hV2) - (w5 - w2) * hL3)
        VAmix5 = mA * RA * (K0 + T5) / (p - pV5)
        mV5 = pV5 / RV * VAmix5 / (K0 + T5)
        VAmix2 = mA * RA * (K0 + T2) / (p - pV2)
        mV2 = pV2 / RV * VAmix2 / (K0 + T2)
        mL3 = (w5 - w2) * mA
        WUE = mL3 * 3600
        fan = VAmix5 * dpA / 1000.0
        pump = (mL1 / 1000.0) * dpL / 1000.0
        PUEnoIT = fan + pump + chill / copv
        varE = (sL4 * mL4 + (sA5 * mA + sV5 * mV5) - sL1 * mL1
                - (sA2 * mA + sV2 * mV2) - sL3 * mL3)
        return WUE, PUEnoIT, varE, fan

    lo = T5min.copy(); hi = T5max.copy()
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        _, _, vE, _ = ev(mid)
        hot = vE < 0
        hi = np.where(hot, mid, hi)
        lo = np.where(hot, lo, mid)
    T5 = (lo + hi) / 2
    WUE, PUEnoIT, varE, fan = ev(T5)
    bad = (T4 < Tdew2) | (varE < -1e-6)
    return np.where(bad, np.nan, WUE), np.where(bad, np.nan, PUEnoIT), fan


# --------------------------------------------- class climates from the cleaned record
T_E = np.arange(-40.0, 51.0, 1.0); RH_E = np.arange(0.0, 102.0, 2.0)
T_C = (T_E[:-1] + T_E[1:]) / 2; RH_C = (RH_E[:-1] + RH_E[1:]) / 2

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].set_index("STATION")    # noqa: E712
dc = pd.read_csv(OUT / "dc_classes.csv").set_index("STATION")
K = int(dc["dc_class"].max())

hists = {c: 0 for c in range(1, K + 1)}
psum = {c: 0.0 for c in range(1, K + 1)}; pcnt = {c: 0 for c in range(1, K + 1)}
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "RH", "p_pa"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        if stn not in dc.index:
            continue
        c = int(dc.loc[stn, "dc_class"])
        h, _, _ = np.histogram2d(g["T"], g["RH"], bins=[T_E, RH_E])
        hists[c] = hists[c] + h
        psum[c] += float(g["p_pa"].sum()); pcnt[c] += len(g)
print(f"class climates built ({time.time()-t0:.0f}s)")

n_by_class = dc[dc.index.isin(res.index)]["dc_class"].value_counts().to_dict()
t4_modal = {c: float(res.loc[dc[dc["dc_class"] == c].index.intersection(res.index),
                             "T4base"].mode().iloc[0]) for c in range(1, K + 1)}
p_mean = {c: psum[c] / pcnt[c] for c in range(1, K + 1)}
print("class design/pressure:", {c: (t4_modal[c], round(p_mean[c])) for c in range(1, K + 1)})

cells = []
for c in range(1, K + 1):
    H = hists[c]
    TT, RRH = np.meshgrid(T_C, RH_C, indexing="ij")
    w = (H / H.sum()).ravel()
    T2 = TT.ravel(); RH2 = RRH.ravel()
    order = np.argsort(w)[::-1]
    cum = np.cumsum(w[order]); ncell = int(np.searchsorted(cum, 0.999)) + 1
    keep = np.zeros(w.size, bool); keep[order[:ncell]] = True
    keep &= w > 0
    cells.append(dict(cls=c, T2=T2[keep], RH2=RH2[keep], w=w[keep],
                      p=np.full(int(keep.sum()), p_mean[c])))
CT2 = np.concatenate([x["T2"] for x in cells]); CRH = np.concatenate([x["RH2"] for x in cells])
CP = np.concatenate([x["p"] for x in cells])
CCLS = np.concatenate([np.full(x["T2"].size, x["cls"]) for x in cells])
CW = np.concatenate([x["w"] for x in cells])
CT4B = np.concatenate([np.full(x["T2"].size, t4_modal[x["cls"]]) for x in cells])
CTWB = twb_c(CT2, CRH, CP)
print(f"{CT2.size} climate cells over 5 classes")
ntot = sum(n_by_class.values())
cls_share = {c: n_by_class[c] / ntot for c in range(1, K + 1)}


def modes_for(a_wet, a_dry):
    """Production mode selection on the cells for a pair of approach temperatures; the
    per class design set point follows the same approach (one degree above the 99.5th
    percentile wet bulb plus a_wet, floored at 26 C) so the two move together."""
    out = {}
    for c in range(1, K + 1):
        sel = CCLS == c
        order = np.argsort(CTWB[sel]); cw = np.cumsum(CW[sel][order]); cw = cw / cw[-1]
        tw995 = CTWB[sel][order][min(np.searchsorted(cw, 0.995), order.size - 1)]
        out[c] = max(26.0, float(np.floor(tw995 + a_wet) + 1.0))
    T4B = np.array([out[int(c)] for c in CCLS])
    T1 = np.empty_like(CT2); T4 = np.empty_like(CT2); EV = np.empty_like(CT2); CH = np.empty_like(CT2)
    for t4b in np.unique(T4B):
        sel = T4B == t4b
        t1, t4, ev, ch, _, _ = select_mode_economizer(CT2[sel], CRH[sel], CP[sel], T4base=float(t4b),
                                                     a_wet=a_wet, a_dry=a_dry, twb=CTWB[sel])
        T1[sel] = t1; T4[sel] = t4; EV[sel] = ev; CH[sel] = ch
    return T1, T4, EV, CH


def evaluate(par):
    T1, T4, EV, CH = modes_for(float(par[9]), float(par[10]))
    W, P, fan = model_param(par, T1, CT2, CRH, T4, EV, CH, CP)
    feas = np.isfinite(W) & np.isfinite(P)
    CWf = np.where(feas, CW, 0.0)
    out = {}
    for c in range(1, K + 1):
        w = np.where(CCLS == c, CWf, 0.0); w = w / w.sum()
        out[("PUE", c)] = 1.0 + float(np.nansum(w * P))
        out[("WUE", c)] = float(np.nansum(w * W))
    out[("PUE", 0)] = sum(cls_share[c] * out[("PUE", c)] for c in range(1, K + 1))
    out[("WUE", 0)] = sum(cls_share[c] * out[("WUE", c)] for c in range(1, K + 1))
    out["fan_mean"] = float(np.nansum(CWf / CWf.sum() * fan))
    return out

base = evaluate(P0)
_tab = pd.read_csv(OUT / "dc_class_table.csv").set_index("dc_class")
_shares = {c: n_by_class[c] / ntot for c in range(1, K + 1)}
prod_world = dict(PUE=float(sum(_shares[c] * _tab.loc[c, "PUE"] for c in range(1, K + 1))),
                  WUE=float(sum(_shares[c] * _tab.loc[c, "WUE"] for c in range(1, K + 1))))
print(f"baseline (grid, no residual ladder): world PUE {base[('PUE',0)]:.4f} WUE {base[('WUE',0)]:.4f}  "
      f"(production record level, class weighted: {prod_world['PUE']:.4f} / {prod_world['WUE']:.4f})")

# --------------------------------------------------------------- Sobol (Saltelli)
try:
    from SALib.sample import sobol as sobol_sample
    sampler = lambda prob, n: sobol_sample.sample(prob, n, calc_second_order=False)
except ImportError:
    from SALib.sample import saltelli as sobol_sample
    sampler = lambda prob, n: sobol_sample.sample(prob, n, calc_second_order=False)
from SALib.analyze import sobol as sobol_an

problem = dict(num_vars=len(NAMES), names=NAMES, bounds=BOUNDS)
X = sampler(problem, 1024)
print(f"Saltelli matrix: {X.shape} ({time.time()-t0:.0f}s)")
Y = {k: np.empty(X.shape[0]) for k in [("PUE", 0), ("WUE", 0)]}
drop_fan = 0
for i in range(X.shape[0]):
    o = evaluate(X[i])
    if o["fan_mean"] > 0.5:
        drop_fan += 1
    Y[("PUE", 0)][i] = o[("PUE", 0)]
    Y[("WUE", 0)][i] = o[("WUE", 0)]
    if (i + 1) % 2000 == 0:
        print(f"  sobol {i+1}/{X.shape[0]}, {time.time()-t0:.0f}s", flush=True)
print(f"evaluations with mean fan adder > 0.5: {drop_fan}")

rows = []
for met in ("PUE", "WUE"):
    Si = sobol_an.analyze(problem, Y[(met, 0)], calc_second_order=False, print_to_console=False)
    for j, nm in enumerate(NAMES):
        rows.append(dict(metric=met, param=nm, S1=float(Si["S1"][j]),
                         S1_conf=float(Si["S1_conf"][j]), ST=float(Si["ST"][j]),
                         ST_conf=float(Si["ST_conf"][j])))
sob = pd.DataFrame(rows)
sob.to_csv(OUT / "gsa_world_sobol.csv", index=False)
print(sob.round(4).to_string(index=False))

# ------------------------------------------------------- LHS uncertainty propagation
from scipy.stats import qmc
lhs = qmc.LatinHypercube(d=len(NAMES), seed=42).random(4096)
lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])
XL = lo + lhs * (hi - lo)
draws = {k: np.empty(4096) for k in [(m, c) for m in ("PUE", "WUE") for c in range(0, K + 1)]}
for i in range(4096):
    o = evaluate(XL[i])
    for k in draws:
        draws[k][i] = o[k]
    if (i + 1) % 1000 == 0:
        print(f"  lhs {i+1}/4096, {time.time()-t0:.0f}s", flush=True)

ci_rows = []
for (met, c), v in draws.items():
    b = base[(met, c)]
    scope = "world" if c == 0 else f"class {c}"
    prod = prod_world[met] if c == 0 else float(
        pd.read_csv(OUT / "dc_class_table.csv").set_index("dc_class").loc[c, met])
    rel_sd = float(v.std() / b)
    ci_rows.append(dict(scope=scope, metric=met, grid_baseline=b, mean=float(v.mean()),
                        sd=float(v.std()), rel_sd_pct=100 * rel_sd,
                        ci95_lo_rel_pct=100 * (np.percentile(v, 2.5) - b) / b,
                        ci95_hi_rel_pct=100 * (np.percentile(v, 97.5) - b) / b,
                        production_mean=prod,
                        production_ci_lo=prod * (1 + (np.percentile(v, 2.5) - b) / b),
                        production_ci_hi=prod * (1 + (np.percentile(v, 97.5) - b) / b)))
ci = pd.DataFrame(ci_rows)
ci.to_csv(OUT / "gsa_world_ci.csv", index=False)
print(ci.round(4).to_string(index=False))
print(f"[done] {time.time()-t0:.0f}s")
