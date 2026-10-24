# Koppen correspondence and transferability at world scale (the generalization of
# CLAUDE.md section 5.4, answering the question posed in prelim_ideas.docx: does the
# Koppen class predict the DC footprint, or is a DC specific classification needed).
#
# Inputs: results_world/station_results.csv, station_koppen.csv, world_hourly.parquet.
# Outputs (results_world/):
#   koppen_class_table.csv   per class: stations, footprint means and spreads, occupancy
#   koppen_correspondence.csv  variance of station means explained by class (eta2), also
#                            by country for comparison
#   koppen_loo.csv           leave one out within class: predict each station from the
#                            pooled climate of its class mates (station excluded), using
#                            the station's own design map; the world scale test of the
#                            paper's "one representative record per climatic class" claim
#   koppen_hellinger.csv     within class mean Hellinger distances, climate vs footprint
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_world"

from dcfootprint.core.properties import V2_PARTIALP                # noqa: E402
from dcfootprint.core.modes import run_policy                      # noqa: E402

t0 = time.time()

# (T, RH) grid, same resolution as the Madrid work: 0.5 C by 1 % RH
T_EDGES = np.arange(-40.0, 50.5, 0.5)
RH_EDGES = np.arange(0.5, 101.5, 1.0)
T_MID = (T_EDGES[:-1] + T_EDGES[1:]) / 2
RH_MID = (RH_EDGES[:-1] + RH_EDGES[1:]) / 2
NT, NR = len(T_MID), len(RH_MID)
P_STEP = 2500.0            # Pa: pressure bin width for the precomputed g maps
MIN_CLASS_N = 5
MAX_ASSIGN_DIST = 2.0      # deg: exclude nearest-polygon assignments farther than this

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].copy()          # noqa: E712
kop = pd.read_csv(OUT / "station_koppen.csv")
res = res.merge(kop[["STATION", "koppen", "dist_deg"]], on="STATION", how="left")
res = res[res["dist_deg"] <= MAX_ASSIGN_DIST].copy()
res["country"] = res["STATION"].str[:2]
print(f"{len(res)} stations in the class analysis, {res['koppen'].nunique()} classes")

# ------------------------------------------------------------------ per class table
cls = res.groupby("koppen").agg(
    n_stations=("STATION", "size"),
    PUE=("PUE", "mean"), PUE_sd=("PUE", "std"), PUE_min=("PUE", "min"), PUE_max=("PUE", "max"),
    WUE=("WUE", "mean"), WUE_sd=("WUE", "std"), WUE_min=("WUE", "min"), WUE_max=("WUE", "max"),
    occ_free=("occ_free", "mean"), occ_evap=("occ_evap", "mean"),
    occ_chill=("occ_chill", "mean"), occ_chill_esc=("occ_chill_esc", "mean"),
    T4base_max=("T4base", "max"),
).sort_values("n_stations", ascending=False)
cls.to_csv(OUT / "koppen_class_table.csv")

# --------------------------------------------------- correspondence: eta2 by partition
def eta2(df, group_col, y):
    m0 = df[y].mean()
    between = df.groupby(group_col)[y].agg(["size", "mean"])
    ssb = float((between["size"] * (between["mean"] - m0) ** 2).sum())
    sst = float(((df[y] - m0) ** 2).sum())
    return ssb / sst if sst > 0 else np.nan

rows = []
for y in ("PUE", "WUE"):
    rows.append(dict(metric=y,
                     eta2_koppen=eta2(res, "koppen", y),
                     eta2_koppen_group=eta2(res.assign(g=res["koppen"].str[0]), "g", y),
                     eta2_country=eta2(res, "country", y)))
corr = pd.DataFrame(rows)
corr.to_csv(OUT / "koppen_correspondence.csv", index=False)
print(corr.to_string(index=False))

# ------------------------------------------------- climate histograms per station
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
hist = {}
p_med = {}
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "RH", "p_pa"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        h, _, _ = np.histogram2d(g["T"].to_numpy(np.float64),
                                 g["RH"].to_numpy(np.float64),
                                 bins=[T_EDGES, RH_EDGES])
        hist[stn] = hist.get(stn, 0) + h.astype(np.float32)
        p_med.setdefault(stn, []).append(float(np.median(g["p_pa"])))
    if (i + 1) % 50 == 0:
        print(f"[hist] row group {i+1}/{pf.num_row_groups}, {time.time()-t0:.0f}s", flush=True)
p_med = {k: float(np.median(v)) for k, v in p_med.items()}

# common Scott bandwidth over stations (matches the Madrid methodology)
ns = np.array([h.sum() for h in hist.values()])

def _wsd(w, x):
    tot = w.sum()
    if tot <= 0:
        return 0.0
    m = (w @ x) / tot
    return float(np.sqrt(max((w @ (x ** 2)) / tot - m ** 2, 1e-9)))

sd_T = np.array([_wsd(h.sum(1).astype(np.float64), T_MID) for h in hist.values()])
sd_R = np.array([_wsd(h.sum(0).astype(np.float64), RH_MID) for h in hist.values()])
scott = float(np.mean(ns) ** (-1.0 / 6.0))
sig_T = float(np.mean(sd_T)) * scott / 0.5    # in T cells
sig_R = float(np.mean(sd_R)) * scott / 1.0    # in RH cells
print(f"Scott bandwidth: {sig_T*0.5:.2f} C x {sig_R:.2f} %RH  "
      f"({time.time()-t0:.0f}s)")

smooth = {}
for stn, h in hist.items():
    f = gaussian_filter(h, sigma=(sig_T, sig_R), mode="constant")
    s = f.sum()
    smooth[stn] = (f / s).astype(np.float32) if s > 0 else f.astype(np.float32)

# ------------------------------------------------- g maps per (design, pressure level)
res_idx = res.set_index("STATION")
designs = {}
for stn in res_idx.index:
    if stn not in smooth or stn not in p_med:
        continue
    key = (float(res_idx.loc[stn, "T4base"]), int(round(p_med[stn] / P_STEP)))
    designs.setdefault(key, []).append(stn)
print(f"{len(designs)} (design, pressure) g maps to precompute")

TT, RR = np.meshgrid(T_MID, RH_MID, indexing="ij")
gmaps = {}
for (t4b, plev) in designs:
    r = run_policy(TT.ravel(), RR.ravel(), T4base=t4b,
                   p=plev * P_STEP, props=V2_PARTIALP, max_rungs=12)
    gmaps[(t4b, plev)] = (r["PUE"].reshape(NT, NR), r["WUE"].reshape(NT, NR))
print(f"g maps done ({time.time()-t0:.0f}s)")

# ------------------------------------------------- Hellinger + leave one out per class
def hellinger_matrix(F):
    """F: stations x cells, rows sum to 1. Returns pairwise Hellinger distances."""
    S = np.sqrt(F)
    bc = np.clip(S @ S.T, 0, 1)
    return np.sqrt(np.clip(1 - bc, 0, 1))

loo_rows = []
hel_rows = []
for kclass, grp in res.groupby("koppen"):
    stns = [s for s in grp["STATION"] if s in smooth]
    if len(stns) < MIN_CLASS_N:
        continue
    F = np.stack([smooth[s].ravel() for s in stns])
    counts = np.array([hist[s].sum() for s in stns])

    Hc = hellinger_matrix(F.astype(np.float64))
    iu = np.triu_indices(len(stns), 1)
    hel_rows.append(dict(koppen=kclass, n=len(stns),
                         H_climate_mean=float(Hc[iu].mean()),
                         H_climate_max=float(Hc[iu].max())))

    raw = np.stack([hist[s].ravel() for s in stns])
    for j, stn in enumerate(stns):
        t4b = float(res_idx.loc[stn, "T4base"]); plev = int(round(p_med[stn] / P_STEP))
        gP, gW = gmaps[(t4b, plev)]
        pool = raw.sum(0) - raw[j]
        f = gaussian_filter(pool.reshape(NT, NR), sigma=(sig_T, sig_R), mode="constant")
        f = f / f.sum()
        ok = ~np.isnan(gP)
        w = f * ok
        w = w / w.sum()
        pred_P = float(np.nansum(w * gP)); pred_W = float(np.nansum(w * gW))
        own = smooth[stn]
        wo = own * ok; wo = wo / wo.sum()
        self_P = float(np.nansum(wo * gP)); self_W = float(np.nansum(wo * gW))
        loo_rows.append(dict(
            STATION=stn, koppen=kclass,
            PUE_actual=float(res_idx.loc[stn, "PUE"]), PUE_loo=pred_P, PUE_self=self_P,
            WUE_actual=float(res_idx.loc[stn, "WUE"]), WUE_loo=pred_W, WUE_self=self_W))

loo = pd.DataFrame(loo_rows)
loo["dPUE_pct"] = 100 * (loo["PUE_loo"] - loo["PUE_actual"]) / loo["PUE_actual"]
loo["dWUE_pct"] = 100 * (loo["WUE_loo"] - loo["WUE_actual"]) / loo["WUE_actual"]
loo["dPUE_self_pct"] = 100 * (loo["PUE_self"] - loo["PUE_actual"]) / loo["PUE_actual"]
loo["dWUE_self_pct"] = 100 * (loo["WUE_self"] - loo["WUE_actual"]) / loo["WUE_actual"]
loo.to_csv(OUT / "koppen_loo.csv", index=False)
pd.DataFrame(hel_rows).to_csv(OUT / "koppen_hellinger.csv", index=False)

summ = loo.groupby("koppen").agg(
    n=("STATION", "size"),
    PUE_mae=("dPUE_pct", lambda x: float(np.abs(x).median())),
    PUE_p95=("dPUE_pct", lambda x: float(np.abs(x).quantile(0.95))),
    WUE_mae=("dWUE_pct", lambda x: float(np.abs(x).median())),
    WUE_p95=("dWUE_pct", lambda x: float(np.abs(x).quantile(0.95))),
).sort_values("n", ascending=False)
print(summ.to_string())
print(f"\nself convolution check (smoothing bias): "
      f"PUE median {loo['dPUE_self_pct'].abs().median():.2f}%  "
      f"WUE median {loo['dWUE_self_pct'].abs().median():.2f}%")
print(f"[done] {time.time()-t0:.0f}s")
