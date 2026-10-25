# Supplementary Fig. 3 (revised): footprint by Köppen class, grouped by main group A to E and
# ordered within group, so the within-group spread is visible. Same pixel size as the original
# (1950 x 825 at 150 dpi) so the SI drawing extent is unchanged.
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "results_world"
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.labelsize": 10.5,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
                     "axes.linewidth": 0.7, "savefig.dpi": 150})

GROUPS = [("A", "A  tropical", "#D55E00"), ("B", "B  arid", "#E69F00"), ("C", "C  temperate", "#009E73"),
          ("D", "D  continental (snow)", "#0072B2"), ("E", "E  polar", "#56B4E9")]
COL = {g: c for g, _, c in GROUPS}

k = pd.read_csv(RW / "koppen_class_table.csv", index_col=0)
k = k[k["n_stations"] >= 25]
order = []
for g, _, _ in GROUPS:
    order += sorted(c for c in k.index if c.startswith(g))
k = k.loc[order]
n = len(order)
y = np.arange(n)[::-1]
colors = [COL[c[0]] for c in order]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
axes[0].barh(y, k["PUE"], xerr=k["PUE_sd"], color=colors, error_kw=dict(lw=0.9, capsize=2.5, ecolor="0.25"))
axes[0].set_xlim(1.0, 1.20)
axes[0].set_xlabel("PUE (kWh kWh$^{-1}$), mean and s.d. across stations")
axes[1].barh(y, k["WUE"], xerr=k["WUE_sd"], color=colors, error_kw=dict(lw=0.9, capsize=2.5, ecolor="0.25"))
axes[1].set_xlim(0, 1.5)
axes[1].set_xlabel("WUE (L kWh$^{-1}$), mean and s.d. across stations")
axes[0].set_yticks(y)
axes[0].set_yticklabels([f"{c}  (n = {int(k.loc[c, 'n_stations'])})" for c in order])
# separators between main groups
bounds = []
for i in range(1, n):
    if order[i][0] != order[i - 1][0]:
        bounds.append((y[i - 1] + y[i]) / 2)
for ax in axes:
    for b in bounds:
        ax.axhline(b, color="0.45", lw=0.7, ls=":")
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
handles = [Patch(fc=c, label=lab) for _, lab, c in GROUPS]
axes[1].legend(handles=handles, loc="lower right", frameon=False, title="Köppen-Geiger main group")
fig.tight_layout()
fig.savefig(RW / "fig_koppen_footprint_grouped.png")
fig.savefig(ROOT / "results_world" / "Journal" / "EPS" / "fig_koppen_footprint_grouped.eps")
print("written", RW / "fig_koppen_footprint_grouped.png", "classes:", order)
