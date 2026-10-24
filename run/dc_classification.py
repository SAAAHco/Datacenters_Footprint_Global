# DC specific climate classification (the prelim_ideas.docx proposal: if Koppen does not
# fully predict the footprint, derive a classification that does).
#
# Method
#   1. Feature vector per station: mode occupancy (free, evap, chiller incl escalated),
#      PUE, WUE, and the absolute seasonal amplitude of monthly WUE (max - min, L/kWh,
#      hemisphere free). Standardized, Ward hierarchical clustering.
#   2. k selection: for k = 2..12 report silhouette and the variance of station PUE and
#      WUE explained (eta2), against the 26 class Koppen benchmark (0.839 / 0.853).
#      The eta2 here is in sample by construction; the claim is parsimony (few classes),
#      verified independently by the leave one out transfer test in step 4.
#   3. Characterization: centroid table, cross tab and adjusted Rand index vs Koppen,
#      and a nearest centroid rule on two climate normals (annual mean T, annual mean
#      dew point) so any location can be classified without running the model.
#   4. Leave one out transfer INSIDE each DC class (same machinery as the Koppen test):
#      the operative proof that the new classes are at least as transferable.
#
# Reads results_world/ only. Outputs: dc_classes.csv, dc_class_table.csv,
# dc_k_selection.csv, dc_vs_koppen_crosstab.csv, dc_loo.csv, dc_classes_map.png.
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_world"

from dcfootprint.core.properties import V2_PARTIALP, pvsat_pa, tdew_c   # noqa: E402
from dcfootprint.core.modes import run_policy                           # noqa: E402

t0 = time.time()
# the 26 class Koppen benchmark of the current run (written by world_koppen_analysis.py)
_kc = pd.read_csv(OUT / "koppen_correspondence.csv").set_index("metric")
KOPPEN_ETA2 = {"PUE": float(_kc.loc["PUE", "eta2_koppen"]),
               "WUE": float(_kc.loc["WUE", "eta2_koppen"])}
print(f"Koppen benchmark eta2: {KOPPEN_ETA2}")

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].copy()          # noqa: E712
kop = pd.read_csv(OUT / "station_koppen.csv")
res = res.merge(kop[["STATION", "koppen", "dist_deg"]], on="STATION", how="left")
res = res[res["dist_deg"] <= 2.0].copy()

mon = pd.read_csv(OUT / "station_monthly.csv")
amp = mon.groupby("STATION")["WUE"].agg(lambda x: x.max() - x.min()).rename("WUE_amp")
res = res.merge(amp, on="STATION", how="inner")
res["occ_chill_all"] = res["occ_chill"] + res["occ_chill_esc"]
print(f"{len(res)} stations")

# ------------------------------------------------- climate normals (one parquet pass)
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
acc = {}
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "RH"]).to_pandas()
    T = d["T"].to_numpy(np.float64); RH = d["RH"].to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        Td = np.where(RH >= 100.0, T, tdew_c(RH / 100.0 * pvsat_pa(T)))
    d2 = pd.DataFrame({"s": d["STATION"], "T": T, "Td": Td})
    g = d2.groupby("s").agg(n=("T", "size"), sT=("T", "sum"), sTd=("Td", "sum"))
    for s, r in g.iterrows():
        a = acc.setdefault(s, [0.0, 0.0, 0.0])
        a[0] += r["n"]; a[1] += r["sT"]; a[2] += r["sTd"]
norm = pd.DataFrame({s: dict(T_mean=a[1] / a[0], Td_mean=a[2] / a[0])
                     for s, a in acc.items()}).T
norm.index.name = "STATION"
res = res.merge(norm, on="STATION", how="left")
print(f"climate normals done ({time.time()-t0:.0f}s)")

# ------------------------------------------------------------- clustering + k selection
FEATS = ["occ_free", "occ_evap", "occ_chill_all", "PUE", "WUE", "WUE_amp"]
X = res[FEATS].to_numpy(np.float64)
Xz = (X - X.mean(0)) / X.std(0)
Z = linkage(Xz, method="ward")
D = squareform(pdist(Xz))

def eta2(labels, y):
    m0 = y.mean(); sst = ((y - y.mean()) ** 2).sum()
    ssb = sum(len(y[labels == c]) * (y[labels == c].mean() - m0) ** 2
              for c in np.unique(labels))
    return float(ssb / sst)

def silhouette(labels):
    s_all = np.zeros(len(labels))
    for i in range(len(labels)):
        same = labels == labels[i]; same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, labels == c].mean() for c in np.unique(labels) if c != labels[i])
        s_all[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s_all.mean())

pue = res["PUE"].to_numpy(); wue = res["WUE"].to_numpy()
sel_rows = []
labels_by_k = {}
for k in range(2, 13):
    lab = fcluster(Z, k, criterion="maxclust")
    labels_by_k[k] = lab
    sel_rows.append(dict(k=k, silhouette=silhouette(lab),
                         eta2_PUE=eta2(lab, pue), eta2_WUE=eta2(lab, wue)))
sel = pd.DataFrame(sel_rows)
sel.to_csv(OUT / "dc_k_selection.csv", index=False)
print(sel.round(3).to_string(index=False))

ok_k = sel[(sel["eta2_PUE"] >= KOPPEN_ETA2["PUE"]) & (sel["eta2_WUE"] >= KOPPEN_ETA2["WUE"])]
K = int(ok_k["k"].min()) if len(ok_k) else int(sel.loc[sel["silhouette"].idxmax(), "k"])
lab = labels_by_k[K]
res["dc_class"] = lab
print(f"\nchosen k = {K} (smallest k matching the 26 class Koppen eta2)")

# order classes by mean PUE so class 1 is the coolest, and build the table
order = res.groupby("dc_class")["PUE"].mean().sort_values().index
remap = {c: i + 1 for i, c in enumerate(order)}
res["dc_class"] = res["dc_class"].map(remap)
tab = res.groupby("dc_class").agg(
    n=("STATION", "size"), PUE=("PUE", "mean"), PUE_sd=("PUE", "std"),
    WUE=("WUE", "mean"), WUE_sd=("WUE", "std"),
    occ_free=("occ_free", "mean"), occ_evap=("occ_evap", "mean"),
    occ_chill=("occ_chill_all", "mean"), WUE_amp=("WUE_amp", "mean"),
    T_mean=("T_mean", "mean"), Td_mean=("Td_mean", "mean"),
    T4base_max=("T4base", "max"), lat_absmean=("lat", lambda x: float(np.abs(x).mean())))
tab.to_csv(OUT / "dc_class_table.csv")
print(tab.round(3).to_string())

# --------------------------------------------------------- comparison against Koppen
ct = pd.crosstab(res["dc_class"], res["koppen"])
ct.to_csv(OUT / "dc_vs_koppen_crosstab.csv")

def ari(a, b):
    ctab = pd.crosstab(a, b).to_numpy(np.float64)
    n = ctab.sum()
    comb = lambda x: x * (x - 1) / 2.0
    sij = comb(ctab).sum(); ai = comb(ctab.sum(1)).sum(); bj = comb(ctab.sum(0)).sum()
    exp = ai * bj / comb(n); mx = (ai + bj) / 2.0
    return float((sij - exp) / (mx - exp))

print(f"\nadjusted Rand index DC classes vs Koppen (26): {ari(res['dc_class'], res['koppen']):.3f}")
print(f"adjusted Rand index DC classes vs Koppen groups (A..E): "
      f"{ari(res['dc_class'], res['koppen'].str[0]):.3f}")

# ------------------------------------- planner rule: nearest centroid on (T_mean, Td_mean)
cent = res.groupby("dc_class")[["T_mean", "Td_mean"]].mean()
sdv = res[["T_mean", "Td_mean"]].std()
zc = (cent / sdv).to_numpy(); zs = (res[["T_mean", "Td_mean"]] / sdv).to_numpy()
pred = cent.index.to_numpy()[np.argmin(((zs[:, None, :] - zc[None, :, :]) ** 2).sum(2), axis=1)]
acc_rule = float((pred == res["dc_class"].to_numpy()).mean())
print(f"nearest centroid rule on (annual mean T, annual mean Td): accuracy {acc_rule*100:.1f}%")
print(cent.round(2).to_string())

# ------------------------------------------------- leave one out inside the DC classes
T_EDGES = np.arange(-40.0, 50.5, 0.5); RH_EDGES = np.arange(0.5, 101.5, 1.0)
T_MID = (T_EDGES[:-1] + T_EDGES[1:]) / 2; RH_MID = (RH_EDGES[:-1] + RH_EDGES[1:]) / 2
NT, NR = len(T_MID), len(RH_MID); P_STEP = 2500.0

hist = {}; p_med = {}
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "RH", "p_pa"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        h, _, _ = np.histogram2d(g["T"].to_numpy(np.float64), g["RH"].to_numpy(np.float64),
                                 bins=[T_EDGES, RH_EDGES])
        hist[stn] = hist.get(stn, 0) + h.astype(np.float32)
        p_med.setdefault(stn, []).append(float(np.median(g["p_pa"])))
p_med = {k: float(np.median(v)) for k, v in p_med.items()}

ns = np.array([h.sum() for h in hist.values()])
def _wsd(w, x):
    tot = w.sum()
    if tot <= 0:
        return 0.0
    m = (w @ x) / tot
    return float(np.sqrt(max((w @ (x ** 2)) / tot - m ** 2, 1e-9)))
scott = float(np.mean(ns) ** (-1.0 / 6.0))
sig_T = np.mean([_wsd(h.sum(1).astype(np.float64), T_MID) for h in hist.values()]) * scott / 0.5
sig_R = np.mean([_wsd(h.sum(0).astype(np.float64), RH_MID) for h in hist.values()]) * scott / 1.0

res_idx = res.set_index("STATION")
keys = {}
for stn in res_idx.index:
    if stn in hist and stn in p_med:
        keys.setdefault((float(res_idx.loc[stn, "T4base"]),
                         int(round(p_med[stn] / P_STEP))), []).append(stn)
TT, RR = np.meshgrid(T_MID, RH_MID, indexing="ij")
gmaps = {}
for (t4b, plev) in keys:
    r = run_policy(TT.ravel(), RR.ravel(), T4base=t4b, p=plev * P_STEP,
                   props=V2_PARTIALP, max_rungs=12)
    gmaps[(t4b, plev)] = (r["PUE"].reshape(NT, NR), r["WUE"].reshape(NT, NR))
print(f"\n{len(gmaps)} g maps ({time.time()-t0:.0f}s)")

loo_rows = []
for dclass, grp in res.groupby("dc_class"):
    stns = [s for s in grp["STATION"] if s in hist]
    if len(stns) < 5:
        continue
    raw = np.stack([hist[s].ravel() for s in stns])
    for j, stn in enumerate(stns):
        gP, gW = gmaps[(float(res_idx.loc[stn, "T4base"]), int(round(p_med[stn] / P_STEP)))]
        pool = raw.sum(0) - raw[j]
        f = gaussian_filter(pool.reshape(NT, NR), sigma=(sig_T, sig_R), mode="constant")
        f = f / f.sum()
        okm = ~np.isnan(gP)
        w = f * okm; w = w / w.sum()
        loo_rows.append(dict(STATION=stn, dc_class=dclass,
                             PUE_actual=float(res_idx.loc[stn, "PUE"]),
                             PUE_loo=float(np.nansum(w * gP)),
                             WUE_actual=float(res_idx.loc[stn, "WUE"]),
                             WUE_loo=float(np.nansum(w * gW))))
loo = pd.DataFrame(loo_rows)
loo["dPUE_pct"] = 100 * (loo["PUE_loo"] - loo["PUE_actual"]) / loo["PUE_actual"]
loo["dWUE_abs"] = (loo["WUE_loo"] - loo["WUE_actual"]).abs()
loo["dWUE_pct"] = 100 * (loo["WUE_loo"] - loo["WUE_actual"]) / loo["WUE_actual"]
loo.to_csv(OUT / "dc_loo.csv", index=False)
summ = loo.groupby("dc_class").agg(
    n=("STATION", "size"),
    PUE_mae_pct=("dPUE_pct", lambda x: float(np.abs(x).median())),
    WUE_mae_pct=("dWUE_pct", lambda x: float(np.abs(x).median())),
    WUE_mae_abs=("dWUE_abs", "median"))
print(summ.round(3).to_string())
print(f"\nDC classes overall LOO median |dPUE| {loo['dPUE_pct'].abs().median():.2f}%  "
      f"|dWUE| {loo['dWUE_abs'].median():.3f} L/kWh")

res[["STATION", "lat", "lon", "koppen", "dc_class"] + FEATS + ["T_mean", "Td_mean"]].to_csv(
    OUT / "dc_classes.csv", index=False)

# ------------------------------------------------------------------------------- map
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapefile as shp
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 12),
                              gridspec_kw={"height_ratios": [1.6, 1.0]})
r = shp.Reader(str(ROOT / "koppen" / "c1976_2000"))
patches = []
for s in r.iterShapes():
    pts = np.array(s.points)
    parts = list(s.parts) + [len(pts)]
    for a, b in zip(parts[:-1], parts[1:]):
        if b - a > 2:
            patches.append(MplPoly(pts[a:b]))
ax.add_collection(PatchCollection(patches, facecolor="0.92", edgecolor="none", zorder=0))
cmap = plt.get_cmap("tab10")
for c, grp in res.groupby("dc_class"):
    ax.scatter(grp["lon"], grp["lat"], s=7, color=cmap((c - 1) % 10),
               label=f"DC class {c} (n={len(grp)})", zorder=2)
ax.set_xlim(-180, 180); ax.set_ylim(-62, 78)
ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")
ax.set_title(f"DC footprint classification, k = {K} classes "
             f"(eta2 PUE {sel.loc[sel['k'] == K, 'eta2_PUE'].iloc[0]:.2f}, "
             f"WUE {sel.loc[sel['k'] == K, 'eta2_WUE'].iloc[0]:.2f}; "
             f"Koppen 26 classes: 0.84 / 0.85)")
ax.legend(loc="lower left", fontsize=8, ncol=2)

for c, grp in res.groupby("dc_class"):
    ax2.scatter(grp["T_mean"], grp["Td_mean"], s=8, color=cmap((c - 1) % 10))
cnt = res.groupby("dc_class")[["T_mean", "Td_mean"]].mean()
for c, row in cnt.iterrows():
    ax2.annotate(str(c), (row["T_mean"], row["Td_mean"]), fontsize=13, weight="bold")
ax2.set_xlabel("Annual mean temperature (C)")
ax2.set_ylabel("Annual mean dew point (C)")
ax2.set_title(f"The same classes in the two climate normals a planner has "
              f"(nearest centroid rule: {acc_rule*100:.0f}% accurate)")
fig.tight_layout()
fig.savefig(OUT / "dc_classes_map.png", dpi=150)
print(f"[done] {time.time()-t0:.0f}s")
