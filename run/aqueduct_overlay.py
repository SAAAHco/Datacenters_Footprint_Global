# Aqueduct water stress overlay (CLAUDE.md 8.6 extension 1): where does the water a DC
# evaporates collide with basin scarcity. Joins every screened world station to its WRI
# Aqueduct 4.0 sub basin (baseline annual water stress, bws) and crosses the stress
# category with the DC classification and the 100 MW benchmark water demand.
#
# Outputs (results_world/):
#   station_water_stress.csv     per station: stress score/category + footprint + class
#   stress_summary.csv           stations, water demand and WUE by stress category
#   stress_dc_crosstab.csv       DC class x stress category (station counts)
#   country_stress.csv           per country: share of stations in scarce basins
#   fig_water_stress.png         map + composition panels
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyogrio.raw import read as ogr_read
import shapely
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_world"
GDB = ROOT / "aqueduct" / "Aqueduct40_waterrisk_download_Y2023M07D05" / "GDB" / "Aq40_Y2023D07M05.gdb"

t0 = time.time()
meta, _, geom_wkb, fields = ogr_read(
    str(GDB), layer="baseline_annual",
    columns=["pfaf_id", "name_0", "bws_raw", "bws_score", "bws_cat", "bws_label"])
geoms = shapely.from_wkb(geom_wkb)
cols = dict(zip(meta["fields"], fields))
print(f"{len(geoms)} basins loaded ({time.time()-t0:.0f}s)")
tree = STRtree(geoms)

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].copy()          # noqa: E712
dc = pd.read_csv(OUT / "dc_classes.csv")[["STATION", "dc_class", "koppen"]]
ann = pd.read_csv(OUT / "annual_benchmark.csv")[["STATION", "water_ML_yr", "overhead_GWh_yr"]]
res = res.merge(dc, on="STATION", how="left").merge(ann, on="STATION", how="left")
res["country"] = res["STATION"].str[:2]

pts = shapely.points(res["lon"].to_numpy(), res["lat"].to_numpy())
rows = []
for i, pt in enumerate(pts):
    idx = tree.query(pt, predicate="within")
    if len(idx):
        gi = int(idx[0]); dist = 0.0
    else:
        gi = int(tree.nearest(pt))
        dist = float(shapely.distance(geoms[gi], pt))
    rows.append((gi, dist))
gi = np.array([r[0] for r in rows]); dist = np.array([r[1] for r in rows])
for name in ("pfaf_id", "name_0", "bws_raw", "bws_score", "bws_cat", "bws_label"):
    res[name] = np.asarray(cols[name])[gi]
res["basin_dist_deg"] = dist
res.loc[res["basin_dist_deg"] > 1.0, ["bws_cat", "bws_label"]] = [-9999, "No Data"]
print(f"joined ({time.time()-t0:.0f}s); nearest fallback used for "
      f"{int((dist > 0).sum())} stations, {int((dist > 1.0).sum())} set to No Data")

res.to_csv(OUT / "station_water_stress.csv", index=False)

CATS = ["Low (<10%)", "Low - Medium (10-20%)", "Medium - High (20-40%)",
        "High (40-80%)", "Extremely High (>80%)", "Arid and Low Water Use", "No Data"]
res["bws_label"] = res["bws_label"].where(res["bws_label"].isin(CATS), "No Data")
SCARCE = ["High (40-80%)", "Extremely High (>80%)", "Arid and Low Water Use"]

summ = res.groupby("bws_label").agg(
    n_stations=("STATION", "size"),
    share_pct=("STATION", lambda x: 100 * len(x) / len(res)),
    WUE=("WUE", "mean"), PUE=("PUE", "mean"),
    water_ML_yr=("water_ML_yr", "mean"),
    total_water_GL_yr=("water_ML_yr", lambda x: x.sum() / 1000.0),
).reindex(CATS)
summ.to_csv(OUT / "stress_summary.csv")
print(summ.round(2).to_string())

ct = pd.crosstab(res["dc_class"], res["bws_label"]).reindex(columns=CATS, fill_value=0)
ct.to_csv(OUT / "stress_dc_crosstab.csv")
print("\n", ct.to_string())

res["scarce"] = res["bws_label"].isin(SCARCE)
red = res[res["scarce"] & res["dc_class"].isin([4, 5])]
green = res[(res["bws_label"].isin(CATS[:2])) & (res["dc_class"] == 1)]
print(f"\nscarce basins (High + Extremely High + Arid): {int(res['scarce'].sum())} stations "
      f"({100 * res['scarce'].mean():.1f}%)")
print(f"RED zone (water hungry class 4 or 5 in scarce basin): {len(red)} "
      f"({100 * len(red) / len(res):.1f}%)")
print(f"GREEN zone (class 1 free cooling in Low or Low-Medium): {len(green)} "
      f"({100 * len(green) / len(res):.1f}%)")
w_scarce = res.loc[res["scarce"], "water_ML_yr"].sum() / res["water_ML_yr"].sum()
print(f"share of benchmark water drawn in scarce basins: {100 * w_scarce:.1f}%")

cn = res.groupby("country").agg(
    n=("STATION", "size"), scarce_pct=("scarce", lambda x: 100 * x.mean()),
    WUE=("WUE", "mean"), water_ML_yr=("water_ML_yr", "mean"))
cn = cn[cn["n"] >= 10].sort_values("scarce_pct", ascending=False)
cn.to_csv(OUT / "country_stress.csv")
print("\ncountries (n>=10) by share of stations in scarce basins:")
print(cn.round(1).head(15).to_string())

# ---------------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapefile as shp
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

COL = {"Low (<10%)": "#ffff99", "Low - Medium (10-20%)": "#ffe600",
       "Medium - High (20-40%)": "#ff9900", "High (40-80%)": "#ff1900",
       "Extremely High (>80%)": "#990000", "Arid and Low Water Use": "#808080",
       "No Data": "#c8c8c8"}

fig = plt.figure(figsize=(13, 11))
ax = fig.add_axes([0.06, 0.42, 0.9, 0.55])
r = shp.Reader(str(ROOT / "koppen" / "c1976_2000"))
patches = []
for s in r.iterShapes():
    p = np.array(s.points)
    parts = list(s.parts) + [len(p)]
    for a, b in zip(parts[:-1], parts[1:]):
        if b - a > 2:
            patches.append(MplPoly(p[a:b]))
ax.add_collection(PatchCollection(patches, facecolor="0.94", edgecolor="none", zorder=0))
for cat in CATS:
    g = res[res["bws_label"] == cat]
    if len(g):
        ax.scatter(g["lon"], g["lat"], s=8, color=COL[cat], zorder=2,
                   label=f"{cat} (n={len(g)})",
                   edgecolors="0.4" if cat.startswith(("Low", "No")) else "none",
                   linewidths=0.2)
ax.set_xlim(-180, 180); ax.set_ylim(-62, 78)
ax.set_title("Aqueduct 4.0 baseline water stress at the 2,081 station panel")
ax.legend(loc="lower left", fontsize=7, ncol=2)
ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")

ax2 = fig.add_axes([0.06, 0.06, 0.40, 0.28])
bottom = np.zeros(len(CATS))
cmap = plt.get_cmap("tab10")
for c in sorted(res["dc_class"].dropna().unique()):
    vals = [int(ct.loc[c, cat]) if c in ct.index else 0 for cat in CATS]
    ax2.bar(range(len(CATS)), vals, bottom=bottom, color=cmap((int(c) - 1) % 10),
            label=f"Cooling class {int(c)}")
    bottom += np.array(vals, float)
ax2.set_xticks(range(len(CATS)))
ax2.set_xticklabels([c.split(" (")[0] for c in CATS], rotation=30, ha="right", fontsize=7)
ax2.set_ylabel("Stations"); ax2.set_title("Cooling class by stress category")
ax2.legend(fontsize=7)

ax3 = fig.add_axes([0.56, 0.06, 0.40, 0.28])
x = np.arange(len(CATS))
ax3.bar(x - 0.2, summ["WUE"].values, 0.4, color="tab:green", label="mean WUE (L/kWh)")
ax3b = ax3.twinx()
ax3b.bar(x + 0.2, summ["water_ML_yr"].values, 0.4, color="tab:blue",
         label="mean water (ML/yr, 100 MW)")
ax3.set_xticks(x)
ax3.set_xticklabels([c.split(" (")[0] for c in CATS], rotation=30, ha="right", fontsize=7)
ax3.set_ylabel("WUE (L/kWh)"); ax3b.set_ylabel("Water (ML/yr)")
ax3.set_title("Water intensity rises with basin stress")
h1, l1 = ax3.get_legend_handles_labels(); h2, l2 = ax3b.get_legend_handles_labels()
ax3.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
fig.savefig(OUT / "fig_water_stress.png", dpi=150, bbox_inches="tight")
print(f"\n[done] {time.time()-t0:.0f}s")
