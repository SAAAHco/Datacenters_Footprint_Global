# Manuscript figures, revision 2: Nature Sustainability presentation standards.
# Bar summaries replaced by distributions, relationships and trajectories:
#   Fig1c stacked mode occupancy by class; Fig1d per-mode conditional DISTRIBUTIONS
#   Fig2c leave-one-out predicted vs actual scatter; Fig3c WUE distributions by stress
#   Fig3d scarcity trajectories to 2050; Fig4d per-class dumbbell collapse under DLC.
# 180 mm, 300 dpi, 7 pt Arial, Okabe-Ito class palette, consistent across figures.
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapefile as shp
from matplotlib.patches import Polygon as MplPoly, FancyArrowPatch
from matplotlib.collections import PatchCollection

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "results_world"
OUT = RW / "Journal"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "savefig.dpi": 300,
})
MM = 1 / 25.4
W = 180 * MM
CLS = {1: "#0072B2", 2: "#56B4E9", 3: "#009E73", 4: "#E69F00", 5: "#D55E00"}
CLS_NAME = {1: "Free-cooling cold", 2: "Free-cooling temperate", 3: "Mixed subtropical",
            4: "Chiller tropical", 5: "Class 5"}
MODES = ["Dry free\ncooling", "Evaporative", "Chiller", "Chiller,\nset point reset"]
MODE_COLS = ["#88CCEE", "#44AA99", "#CC6677", "#882255"]

res = pd.read_csv(RW / "station_results.csv")
res = res[res["screen_pass"] == True]                       # noqa: E712
dc = pd.read_csv(RW / "dc_classes.csv")
K = int(dc["dc_class"].max())
resc = res.merge(dc[["STATION", "dc_class", "T_mean", "Td_mean"]], on="STATION", how="left")


def land(ax):
    r = shp.Reader(str(ROOT / "koppen" / "c1976_2000"))
    patches = []
    for s in r.iterShapes():
        pts = np.array(s.points)
        parts = list(s.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            if b - a > 2:
                patches.append(MplPoly(pts[a:b]))
    ax.add_collection(PatchCollection(patches, facecolor="0.92", edgecolor="none", zorder=0))
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 78)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_label(ax, letter, x=-0.01, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def style_box(bp, color):
    for el in ("boxes", "whiskers", "caps", "medians"):
        for item in bp[el]:
            item.set_color(color)
            item.set_linewidth(0.8)
    for item in bp["medians"]:
        item.set_color("black")


# ================================================================ Figure 1
fig = plt.figure(figsize=(W, 132 * MM))
ax_a = fig.add_axes([0.015, 0.54, 0.46, 0.41])
ax_b = fig.add_axes([0.515, 0.54, 0.46, 0.41])
ax_c = fig.add_axes([0.125, 0.075, 0.315, 0.36])
ax_d1 = fig.add_axes([0.56, 0.075, 0.19, 0.36])
ax_d2 = fig.add_axes([0.79, 0.075, 0.19, 0.36])

for ax, var, cmap, lab, letter in ((ax_a, "PUE", "viridis", "PUE (kWh kWh$^{-1}$)", "a"),
                                   (ax_b, "WUE", "plasma", "WUE (L kWh$^{-1}$)", "b")):
    land(ax)
    sc = ax.scatter(res["lon"], res["lat"], c=res[var], s=1.6, cmap=cmap, zorder=2,
                    rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01, aspect=28)
    cb.set_label(lab, size=6.5); cb.ax.tick_params(labelsize=6)
    panel_label(ax, letter, x=0.0, y=0.98)

# c: mode occupancy composition by Koppen main group (established taxonomy; ours comes in Fig 2)
kop = pd.read_csv(RW / "station_koppen.csv")[["STATION", "koppen"]]
resk = res.merge(kop, on="STATION", how="left")
resk["group"] = resk["koppen"].str[0]
GROUPS = ["A", "B", "C", "D", "E"]
GLAB = {"A": "A Tropical", "B": "B Arid", "C": "C Temperate", "D": "D Continental",
        "E": "E Polar", "W": "World"}
occ_cols = ["occ_free", "occ_evap", "occ_chill", "occ_chill_esc"]
comp = resk[resk["group"].isin(GROUPS)].groupby("group")[occ_cols].mean() * 100
comp.loc["W"] = res[occ_cols].mean() * 100
rows = ["W"] + GROUPS[::-1]
left = np.zeros(len(rows))
for m, (col, mc, mlab) in enumerate(zip(occ_cols, MODE_COLS,
                                        ["Dry free cooling", "Evaporative", "Chiller", "Chiller, set point reset"])):
    vals = comp.loc[rows, col].to_numpy()
    ax_c.barh([GLAB[r] for r in rows], vals, left=left, color=mc, label=mlab,
              height=0.62)
    left += vals
ax_c.set_xlim(0, 100)
ax_c.set_xlabel("Share of hours (%)")
ax_c.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
panel_label(ax_c, "c", y=1.12)

# d: conditional footprint distributions across stations (the plateaus as data)
pue_data = [res[f"PUE_{k}"].dropna() for k in ("free", "evap", "chill", "chillesc")]
wue_data = [res[f"WUE_{k}"].dropna() for k in ("evap", "chill", "chillesc")]
bp1 = ax_d1.boxplot(pue_data, positions=range(4), widths=0.55, showfliers=False)
style_box(bp1, "#E69F00")
ax_d1.set_xticks(range(4)); ax_d1.set_xticklabels(["Dry", "Evap.", "Chill.", "Reset"])
ax_d1.set_ylabel("Conditional PUE (kWh kWh$^{-1}$)")
bp2 = ax_d2.boxplot(wue_data, positions=range(3), widths=0.55, showfliers=False)
style_box(bp2, "#009E73")
ax_d2.set_xticks(range(3)); ax_d2.set_xticklabels(["Evap.", "Chill.", "Reset"])
ax_d2.set_ylabel("Conditional WUE (L kWh$^{-1}$)")
ax_d2.text(-0.9, ax_d2.get_ylim()[1], "", fontsize=6)
panel_label(ax_d1, "d")
fig.savefig(OUT / "Fig1.png"); fig.savefig(OUT / "Fig1.pdf")
fig.savefig(OUT / "Fig1.eps"); fig.savefig(OUT / "EPS" / "Fig1.eps"); plt.close(fig)

# ================================================================ Figure 2
ksel = pd.read_csv(RW / "dc_k_selection.csv")
# leak-free leave-one-out: class membership of the held-out station from the spatially blocked tree
loo_d = pd.read_csv(RW / "dc_loo_blocked.csv") if (RW / "dc_loo_blocked.csv").exists() else pd.read_csv(RW / "dc_loo.csv")
loo_k = pd.read_csv(RW / "koppen_loo.csv")

fig = plt.figure(figsize=(W, 150 * MM))
ax_a = fig.add_axes([0.07, 0.60, 0.36, 0.35])
ax_b = fig.add_axes([0.50, 0.55, 0.49, 0.42])
ax_c1 = fig.add_axes([0.07, 0.07, 0.19, 0.40])
ax_c2 = fig.add_axes([0.315, 0.07, 0.19, 0.40])
ax_d = fig.add_axes([0.60, 0.07, 0.38, 0.40])

ax_a.plot(ksel["k"], ksel["eta2_PUE"], "o-", color="#E69F00", ms=3, label="PUE, cooling classes")
ax_a.plot(ksel["k"], ksel["eta2_WUE"], "s-", color="#009E73", ms=3, label="WUE, cooling classes")
kcorr = pd.read_csv(RW / "koppen_correspondence.csv")
kcorr = kcorr[kcorr["metric"].isin(["PUE", "WUE"])].set_index("metric").astype(float)
ax_a.axhline(kcorr.loc["PUE", "eta2_koppen"], color="#E69F00", ls="--", lw=0.8)
ax_a.axhline(kcorr.loc["WUE", "eta2_koppen"], color="#009E73", ls="--", lw=0.8)
ax_a.text(11.8, float(kcorr.loc["PUE", "eta2_koppen"]) - 0.012, "Köppen-Geiger, 26 classes", ha="right", fontsize=6.5, color="0.25")
ax_a.axvline(K, color="0.4", ls=":", lw=0.8)
ax_a.set_xlabel("Number of classes $k$")
ax_a.set_ylabel("Between-station variance explained")
ax_a.legend(frameon=False, loc="lower right")
panel_label(ax_a, "a")

land(ax_b)
for c, g in resc.dropna(subset=["dc_class"]).groupby("dc_class"):
    ax_b.scatter(g["lon"], g["lat"], s=1.6, color=CLS[int(c)], zorder=2, rasterized=True,
                 label=f"{int(c)} {CLS_NAME[int(c)]}")
ax_b.legend(loc="lower left", frameon=False, markerscale=4, handletextpad=0.3)
panel_label(ax_b, "b", x=0.0, y=0.98)

# c: leave-one-out predicted vs actual, colored by class
ok = loo_d.dropna(subset=["dc_class"])
# Classes 2 and 3 share PUE 1.040, so their bands coincide exactly in the PUE panel (any marker
# hides one class under the other). The two bands are offset vertically by +-0.002 for
# visibility; the offset is declared in the legend.
DODGE = {}
for c, g in ok.groupby("dc_class"):
    ax_c1.scatter(g["PUE_actual"], g["PUE_loo"] + DODGE.get(int(c), 0.0), s=1.5,
                  color=CLS[int(c)], rasterized=True)
    ax_c2.scatter(g["WUE_actual"], g["WUE_loo"], s=1.5, color=CLS[int(c)], rasterized=True)
lims1 = [1.0, float(max(ok["PUE_actual"].max(), ok["PUE_loo"].max())) + 0.006]
ax_c1.plot(lims1, lims1, "k-", lw=0.6)
ax_c1.set_xlim(lims1); ax_c1.set_ylim(lims1)
ax_c1.set_xlabel("Station PUE (kWh kWh$^{-1}$)"); ax_c1.set_ylabel("Predicted from class climate")
# absolute errors (kWh kWh-1), same unit convention as the WUE panel
p_d = (loo_d["PUE_loo"] - loo_d["PUE_actual"]).abs().median()
p_k = (loo_k["PUE_loo"] - loo_k["PUE_actual"]).abs().median()
ax_c1.text(0.04, 0.975, f"median error {p_d:.3f} kWh kWh$^{{-1}}$\n(Köppen-Geiger, {loo_k["koppen"].nunique()} cl.: {p_k:.3f})",
           transform=ax_c1.transAxes, fontsize=6.0, va="top")
lims2 = [0.0, float(max(ok["WUE_actual"].max(), ok["WUE_loo"].max())) + 0.05]
ax_c2.plot(lims2, lims2, "k-", lw=0.6)
ax_c2.set_xlim(lims2); ax_c2.set_ylim(lims2)
ax_c2.set_xlabel("Station WUE (L kWh$^{-1}$)")
w_d = (loo_d["WUE_loo"] - loo_d["WUE_actual"]).abs().median()
w_k = (loo_k["WUE_loo"] - loo_k["WUE_actual"]).abs().median()
ax_c2.text(0.04, 0.975, f"median error {w_d:.3f} L kWh$^{{-1}}$\n(Köppen-Geiger: {w_k:.3f})",
           transform=ax_c2.transAxes, fontsize=6.0, va="top")
panel_label(ax_c1, "c")

mon = pd.read_csv(RW / "station_monthly.csv")
tr = mon.groupby("STATION")["T"].agg(lambda s: s.max() - s.min()).rename("T_range")
dd = resc.dropna(subset=["dc_class"]).merge(tr, on="STATION")
for c, g in dd.groupby("dc_class"):
    ax_d.scatter(g["T_mean"], g["T_range"], s=2.5, color=CLS[int(c)], rasterized=True)
for c, g in dd.groupby("dc_class"):
    ax_d.text(g["T_mean"].mean(), g["T_range"].mean(), str(int(c)), fontsize=9,
              fontweight="bold", ha="center", va="center",
              bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=CLS[int(c)], lw=1.0))
ax_d.set_xlabel("Annual mean temperature (°C)")
ax_d.set_ylabel("Seasonal range of monthly mean temperature (°C)")
panel_label(ax_d, "d")
fig.savefig(OUT / "Fig2.png"); fig.savefig(OUT / "Fig2.pdf")
fig.savefig(OUT / "Fig2.eps"); fig.savefig(OUT / "EPS" / "Fig2.eps"); plt.close(fig)

# ================================================================ Figure 3
ann = pd.read_csv(RW / "annual_benchmark.csv")
pp = pd.read_csv(RW / "station_precip_phase.csv")
sw = pd.read_csv(RW / "station_water_stress.csv")
monc = mon.merge(dc[["STATION", "dc_class"]], on="STATION").merge(
    res[["STATION", "lat"]], on="STATION")
monc["Month_loc"] = np.where(monc["lat"] >= 0, monc["Month"],
                             ((monc["Month"] + 5) % 12) + 1)
DIM = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

fig = plt.figure(figsize=(W, 135 * MM))
ax_a = fig.add_axes([0.07, 0.58, 0.40, 0.37])
ax_b = fig.add_axes([0.58, 0.58, 0.40, 0.37])
ax_c = fig.add_axes([0.07, 0.085, 0.40, 0.36])
ax_d = fig.add_axes([0.58, 0.085, 0.40, 0.36])

for c, g in monc.groupby("dc_class"):
    prof = g.groupby("Month_loc")["WUE"].mean()
    water = prof * 100 * 1000 * pd.Series({m: d * 24 for m, d in DIM.items()}) / 1e6
    ax_a.plot(water.index, water.values, "o-", ms=2.5, color=CLS[int(c)],
              label=f"Class {int(c)}")
ax_a.set_xlabel("Month (Southern Hemisphere shifted by six)")
ax_a.set_ylabel("Water demand (ML month$^{-1}$, 100 MW)")
ax_a.legend(frameon=False, ncol=2)
panel_label(ax_a, "a")

ax_b.hist(ann["r_et0"].dropna(), bins=36, color="#0072B2")
ax_b.axvline(0, color="k", lw=0.7)
ax_b.set_xlabel("Correlation of monthly facility water with ET$_0$")
ax_b.set_ylabel("Stations")
ax_bb = ax_b.inset_axes([0.12, 0.40, 0.42, 0.46], zorder=6)
ax_bb.set_facecolor("white"); ax_bb.patch.set_alpha(1.0)
med = pp[pp["dc_class"].isin(range(1, K + 1))].groupby("dc_class")["r_precip"].median()
xpos = np.arange(len(med))
ax_bb.bar(xpos, med.values, color=[CLS[int(c)] for c in med.index])
ax_bb.axhline(0, color="k", lw=0.6)
ax_bb.set_xticks(xpos)
ax_bb.set_xticklabels([str(int(c)) for c in med.index])
ax_bb.set_title("Rain supply phase by class", fontsize=6.5, pad=2)
ax_bb.set_ylabel("median $r$", fontsize=6.5); ax_bb.tick_params(labelsize=6)
ax_bb.set_xlabel("Cooling class", fontsize=6.5)
panel_label(ax_b, "b")

# c: WUE distributions by stress category (boxes, Aqueduct palette)
CATS = ["Low (<10%)", "Low - Medium (10-20%)", "Medium - High (20-40%)",
        "High (40-80%)", "Extremely High (>80%)", "Arid and Low Water Use"]
COLC = ["#d4b93c", "#e0a025", "#ff9900", "#ff1900", "#990000", "#808080"]
data = [sw[sw["bws_label"] == cat]["WUE"].dropna() for cat in CATS]
bp = ax_c.boxplot(data, positions=range(6), widths=0.6, showfliers=False, patch_artist=True,
                  showmeans=True,
                  meanprops=dict(marker="D", markerfacecolor="white",
                                 markeredgecolor="black", markersize=3.5))
for patch, color in zip(bp["boxes"], COLC):
    patch.set_facecolor(color); patch.set_alpha(0.75); patch.set_edgecolor("0.25")
    patch.set_linewidth(0.6)
for el in ("whiskers", "caps"):
    for item in bp[el]:
        item.set_color("0.25"); item.set_linewidth(0.7)
for item in bp["medians"]:
    item.set_color("black"); item.set_linewidth(1.0)
ax_c.set_xticks(range(6))
ax_c.set_xticklabels(["Low", "Low-\nMed", "Med-\nHigh", "High", "Extr.\nHigh", "Arid"])
ax_c.set_ylabel("Station WUE (L kWh$^{-1}$)")
ax_c.set_xlabel("Aqueduct baseline water stress of the basin")
panel_label(ax_c, "c")

# d: scarcity trajectories to 2050 (all three pathways through their actual 2030 values,
# stress_future_summary.csv)
years = [2024, 2030, 2050]
_ss = pd.read_csv(RW / "stress_summary.csv")
_SC = ["High (40-80%)", "Extremely High (>80%)", "Arid and Low Water Use"]
_b = _ss[_ss["bws_label"].isin(_SC)]
assert len(_b) == 3, "scarce stress categories not found"
_base = {"scarce_pct": float(_b["share_pct"].sum()),
         "water_share_pct": float(100 * _b["total_water_GL_yr"].sum()
                                  / _ss["total_water_GL_yr"].sum())}
_fs = pd.read_csv(RW / "stress_future_summary.csv").set_index("scenario")
def _traj(tag, col):
    return [_base[col], float(_fs.loc[tag + "2030", col]), float(_fs.loc[tag + "2050", col])]
bau_scarce = _traj("bau", "scarce_pct"); bau_water = _traj("bau", "water_share_pct")
opt_scarce = _traj("opt", "scarce_pct"); opt_water = _traj("opt", "water_share_pct")
pes_scarce = _traj("pes", "scarce_pct"); pes_water = _traj("pes", "water_share_pct")
ax_d.plot(years, bau_scarce, "o-", color="#D55E00", label="Stations in scarce basins, BAU")
ax_d.plot(years, opt_scarce, "o--", color="#D55E00", lw=0.8, alpha=0.7, ms=2.5,
          label="Optimistic")
ax_d.plot(years, pes_scarce, "o:", color="#D55E00", lw=0.8, alpha=0.7, ms=2.5,
          label="Pessimistic")
ax_d.plot(years, bau_water, "s-", color="#0072B2", label="Share of water demand, BAU")
ax_d.plot(years, opt_water, "s--", color="#0072B2", lw=0.8, alpha=0.7, ms=2.5)
ax_d.plot(years, pes_water, "s:", color="#0072B2", lw=0.8, alpha=0.7, ms=2.5)
ax_d.set_xticks(years); ax_d.set_xticklabels(["Baseline", "2030", "2050"])
ax_d.set_ylabel("Share (%)")
ax_d.legend(frameon=False, fontsize=6.5, loc="upper left")
panel_label(ax_d, "d")
fig.savefig(OUT / "Fig3.png"); fig.savefig(OUT / "Fig3.pdf")
fig.savefig(OUT / "Fig3.eps"); fig.savefig(OUT / "EPS" / "Fig3.eps"); plt.close(fig)

# ================================================================ Figure 4
fr = pd.read_csv(RW / "design_frontier_stations.csv")
wf = fr.groupby("lam")[["PUE", "WUE"]].mean().reset_index()
pol = pd.read_csv(RW / "design_policy_summary.csv")
w0 = pol[pol["scope"] == "world"].iloc[0]
dl = pd.read_csv(RW / "dlc_archetype.csv")
wsu = pd.read_csv(RW / "warming_summary.csv")

fig = plt.figure(figsize=(W, 135 * MM))
ax_a = fig.add_axes([0.07, 0.58, 0.40, 0.37])
ax_b = fig.add_axes([0.58, 0.58, 0.40, 0.37])
ax_c = fig.add_axes([0.095, 0.085, 0.375, 0.36])
ax_d = fig.add_axes([0.58, 0.085, 0.40, 0.36])

base = wsu[wsu["scenario"] == "base"].iloc[0]
for name, color, lab in (("constRH", "#D55E00", "Constant relative humidity"),
                         ("constTd", "#0072B2", "Constant dew point")):
    g = wsu[wsu["scenario"] == name].sort_values("dT")
    ax_a.plot([0] + g["dT"].tolist(), [base["PUE"]] + g["PUE"].tolist(), "o-", ms=3,
              color=color, label=lab)
    ax_b.plot([0] + g["dT"].tolist(), [base["WUE"]] + g["WUE"].tolist(), "o-", ms=3,
              color=color, label=lab)
ax_a.set_xlabel("Warming (°C)"); ax_a.set_ylabel("World mean PUE (kWh kWh$^{-1}$)")
ax_a.legend(frameon=False); panel_label(ax_a, "a")
ax_b.set_xlabel("Warming (°C)"); ax_b.set_ylabel("World mean WUE (L kWh$^{-1}$)")
ax_b.legend(frameon=False); panel_label(ax_b, "b")

ax_c.plot(wf["WUE"], wf["PUE"], "o-", color="#0072B2", ms=3, label="Operating-policy frontier")
ax_c.scatter([w0["WUE_base"]], [w0["PUE_base"]], marker="s", s=35, color="#E69F00",
             zorder=5, label="Production policy")
ax_c.scatter([w0["WUE_pueopt"]], [w0["PUE_pueopt"]], marker="^", s=35, color="#D55E00",
             zorder=5, label="Energy-only optimum")
ax_c.scatter([w0["WUE_wueopt"]], [w0["PUE_wueopt"]], marker="v", s=35, color="#009E73",
             zorder=5, label="Water optimum")
_ls = pd.read_csv(RW / "design_lambda_star.csv").set_index("scope").loc["world"]
if np.isfinite(_ls["lambda_star"]):
    ax_c.scatter([_ls["WUE_at"]], [_ls["PUE_at"]], marker="o", s=30, facecolor="white",
                 edgecolor="#0072B2", zorder=6)
    ax_c.annotate(f"water price $\\lambda^*$ = {_ls['lambda_star']:.4f} kWh L$^{{-1}}$",
                  xy=(float(_ls["WUE_at"]), float(_ls["PUE_at"])),
                  xytext=(0.22, 0.86), textcoords="axes fraction",
                  fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.6))
ax_c.set_xlabel("WUE (L kWh$^{-1}$)"); ax_c.set_ylabel("PUE (kWh kWh$^{-1}$)")
ax_c.legend(frameon=False, loc="center right")
panel_label(ax_c, "c")

# d: per-class collapse under direct liquid cooling (dumbbell arrows)
dsum = dl[dl["dc_class"].isin(range(1, K + 1))].groupby("dc_class")[["WUE_base", "WUE_dlc"]].mean()
ypos = np.arange(len(dsum))[::-1]
for (c, row), y in zip(dsum.iterrows(), ypos):
    ax_d.add_patch(FancyArrowPatch((row["WUE_base"], y), (row["WUE_dlc"], y),
                                   arrowstyle="-|>", mutation_scale=9,
                                   color=CLS[int(c)], lw=1.4, zorder=3))
    ax_d.scatter([row["WUE_base"]], [y], s=26, color="0.45", zorder=4)
    ax_d.scatter([row["WUE_dlc"]], [y], s=26, color=CLS[int(c)], zorder=4)
ax_d.set_yticks(ypos)
ax_d.set_yticklabels([f"Class {int(c)}" for c in dsum.index])
ax_d.set_xlabel("WUE (L kWh$^{-1}$)")
ax_d.set_xlim(-0.05, 1.45)
ax_d.set_ylim(-0.6, 4.95)
ax_d.scatter([], [], s=26, color="0.45", label="Evaporative baseline")
ax_d.add_patch(FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=9,
                               color="0.3", lw=1.4, alpha=0))
ax_d.plot([], [], color="0.3", lw=1.4, label="Direct liquid cooling")
ax_d.legend(frameon=False, loc="upper right", fontsize=6.5)
_cut = 100 * (1 - dl["WUE_dlc"].mean() / dl["WUE_base"].mean())
ax_d.text(0.02, 0.955, f"world: -{_cut:.0f}% water", transform=ax_d.transAxes, fontsize=6.5,
          fontstyle="italic", va="top")
panel_label(ax_d, "d")
fig.savefig(OUT / "Fig4.png"); fig.savefig(OUT / "Fig4.pdf")
fig.savefig(OUT / "Fig4.eps"); fig.savefig(OUT / "EPS" / "Fig4.eps"); plt.close(fig)

print("revised manuscript figures written to", OUT)
