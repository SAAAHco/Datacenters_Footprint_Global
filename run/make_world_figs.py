# Figures for the world results, mirroring the old project's figure suite.
# Reads results_world/ only, writes PNG figures alongside the CSVs.
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapefile as shp
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_world"
DPI = 150

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].copy()          # noqa: E712


def world_axes(ax):
    r = shp.Reader(str(ROOT / "koppen" / "c1976_2000"))
    patches = []
    for s in r.iterShapes():
        pts = np.array(s.points)
        parts = list(s.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            if b - a > 2:
                patches.append(MplPoly(pts[a:b]))
    ax.add_collection(PatchCollection(patches, facecolor="0.93", edgecolor="none", zorder=0))
    ax.set_xlim(-180, 180); ax.set_ylim(-62, 78)
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")


# fig 1 and 2: world maps of PUE and WUE
for var, cmap, fname, label in (("PUE", "viridis", "fig_world_map_pue.png", "PUE (kWh/kWh)"),
                                ("WUE", "plasma", "fig_world_map_wue.png", "WUE (L/kWh)")):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    world_axes(ax)
    sc = ax.scatter(res["lon"], res["lat"], c=res[var], s=8, cmap=cmap, zorder=2)
    fig.colorbar(sc, ax=ax, shrink=0.75, label=label)
    ax.set_title(f"Long term {var}, 2,081 stations, 2022 to 2025")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=DPI); plt.close(fig)

# fig 3: nested variance decomposition
vd = pd.read_csv(OUT / "variance_decomp_world.csv")
scales = ["Country", "Station", "Interannual", "Seasonal", "Diurnal", "Synoptic"]
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(scales)); w = 0.38
for i, m in enumerate(("PUE", "WUE")):
    row = vd[vd["metric"] == m].iloc[0]
    ax.bar(x + (i - 0.5) * w, [row[s] for s in scales], w, label=m)
ax.set_xticks(x); ax.set_xticklabels(scales, rotation=20)
ax.set_ylabel("Share of total variance (%)")
ax.set_title("Nested variance decomposition, world scale")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_variance_decomp.png", dpi=DPI)
plt.close(fig)

# fig 4: averaging horizon (median over stations) vs the i.i.d. reference
hor = pd.read_csv(OUT / "horizon_world.csv", index_col=0)
hours = {"1h": 1, "1D": 24, "7D": 168, "MS": 730, "YS": 8760}
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for m in ("PUE", "WUE"):
    hh = [hours[c] for c in hor.columns]
    ax.loglog(hh, hor.loc[m].values, "o-", label=m)
hh = np.array(sorted(hours.values()))
ax.loglog(hh, 1.0 / hh, "k--", lw=1, label="i.i.d. 1/n")
ax.set_xlabel("Averaging window (hours)")
ax.set_ylabel("Normalized variance of block means")
ax.set_title("Averaging horizon, median over 2,081 stations")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_horizon.png", dpi=DPI)
plt.close(fig)

# fig 5: mode occupancy and per mode footprint, world means
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
modes = ["free", "evap", "chill", "chillesc"]   # chillesc = chiller with set point reset
mlabels = ["Free", "Evap", "Chiller", "Chiller esc"]
occ = [res[f"occ_{m}" if m != "chillesc" else "occ_chill_esc"].mean() * 100 for m in
       ["free", "evap", "chill", "chillesc"]]
axes[0].bar(mlabels, occ, color="tab:blue")
axes[0].set_ylabel("Occupancy (%)"); axes[0].set_title("Mode occupancy, station mean")
axes[1].bar(mlabels, [res[f"PUE_{m}"].mean() for m in modes], color="tab:orange")
axes[1].set_ylabel("PUE (kWh/kWh)"); axes[1].set_title("Per mode PUE")
axes[1].set_ylim(1.0, None)
axes[2].bar(mlabels, [res[f"WUE_{m}"].mean() for m in modes], color="tab:green")
axes[2].set_ylabel("WUE (L/kWh)"); axes[2].set_title("Per mode WUE")
for a in axes:
    a.tick_params(axis="x", rotation=15)
fig.tight_layout(); fig.savefig(OUT / "fig_modes.png", dpi=DPI); plt.close(fig)

# fig 6: footprint by Koppen class (classes with 25+ stations)
k = pd.read_csv(OUT / "koppen_class_table.csv", index_col=0)
k = k[k["n_stations"] >= 25].sort_values("PUE")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
axes[0].barh(k.index, k["PUE"], xerr=k["PUE_sd"], color="tab:orange")
axes[0].set_xlabel("PUE (kWh/kWh)"); axes[0].set_xlim(1.0, None)
axes[1].barh(k.index, k["WUE"], xerr=k["WUE_sd"], color="tab:green")
axes[1].set_xlabel("WUE (L/kWh)")
fig.suptitle("Footprint by Koppen class (mean and station spread, n >= 25)")
fig.tight_layout(); fig.savefig(OUT / "fig_koppen_footprint.png", dpi=DPI); plt.close(fig)

# fig 7: water calendar of the 100 MW benchmark by DC class, plus ET0 correlation
dc = pd.read_csv(OUT / "dc_classes.csv")
mon = pd.read_csv(OUT / "station_monthly.csv").merge(dc[["STATION", "dc_class", "lat"]],
                                                     on="STATION")
mon["Month_loc"] = np.where(mon["lat"] >= 0, mon["Month"], ((mon["Month"] + 5) % 12) + 1)
DIM = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
cmap = plt.get_cmap("tab10")
for c, g in mon.groupby("dc_class"):
    prof = g.groupby("Month_loc")["WUE"].mean()
    water = prof * 100 * 1000 * pd.Series({m: d * 24 for m, d in DIM.items()}) / 1e6
    ax.plot(water.index, water.values, "o-", color=cmap((c - 1) % 10), label=f"DC class {c}")
ax.set_xlabel("Calendar month (southern hemisphere shifted by 6)")
ax.set_ylabel("Water (ML per month, 100 MW facility)")
ax.set_title("Water calendar by DC class"); ax.legend(fontsize=8)
ann = pd.read_csv(OUT / "annual_benchmark.csv")
ax2.hist(ann["r_et0"].dropna(), bins=40, color="tab:blue")
ax2.axvline(0, color="k", lw=1)
ax2.set_xlabel("Pearson r, monthly DC water vs ET0")
ax2.set_ylabel("Stations")
ax2.set_title(f"DC water vs irrigation timing (median r = {ann['r_et0'].median():.2f})")
fig.tight_layout(); fig.savefig(OUT / "fig_water_calendar.png", dpi=DPI); plt.close(fig)

# fig 8: k selection curve for the DC classification
sel = pd.read_csv(OUT / "dc_k_selection.csv")
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(sel["k"], sel["eta2_PUE"], "o-", label="eta2 PUE (DC classes)")
ax.plot(sel["k"], sel["eta2_WUE"], "s-", label="eta2 WUE (DC classes)")
ax.axhline(0.839, color="tab:blue", ls="--", lw=1, label="Koppen 26 classes, PUE")
ax.axhline(0.853, color="tab:orange", ls="--", lw=1, label="Koppen 26 classes, WUE")
ax.axvline(5, color="k", ls=":", lw=1)
ax.set_xlabel("Number of DC classes k"); ax.set_ylabel("Variance of station means explained")
ax.set_title("Five DC classes match 26 Koppen classes")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(OUT / "fig_k_selection.png", dpi=DPI); plt.close(fig)

print("figures written")
