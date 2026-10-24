# Two additional figures for the revised report: the water energy trade off frontier with
# the policy and archetype points, and the warming trajectories.
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_world"
DPI = 150

# ---------------------------------------------------------------- frontier figure
fr = pd.read_csv(OUT / "design_frontier_stations.csv")
wf = fr.groupby("lam")[["PUE", "WUE"]].mean().reset_index()
pol = pd.read_csv(OUT / "design_policy_summary.csv")
w = pol[pol["scope"] == "world"].iloc[0]
dl = pd.read_csv(OUT / "dlc_archetype.csv")
sw = pd.read_csv(OUT / "design_t4_sweep.csv")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
ax.plot(wf["WUE"], wf["PUE"], "o-", color="tab:blue", label="operating policy frontier (lambda sweep)")
ax.scatter([w["WUE_base"]], [w["PUE_base"]], marker="s", s=90, color="tab:orange", zorder=5,
           label="production policy (eq. 22)")
ax.scatter([w["WUE_pueopt"]], [w["PUE_pueopt"]], marker="^", s=90, color="tab:red", zorder=5,
           label="exact PUE optimal (water blind)")
ax.scatter([w["WUE_wueopt"]], [w["PUE_wueopt"]], marker="v", s=90, color="tab:green", zorder=5,
           label="exact WUE optimal")
ax.scatter([dl["WUE_dlc"].mean()], [dl["PUE_dlc"].mean()], marker="*", s=220, color="tab:brown",
           zorder=5, label="direct liquid cooling archetype")
ax.set_xlabel("WUE (L/kWh)"); ax.set_ylabel("PUE (kWh/kWh)")
ax.set_title("The water energy trade off, world means")
ax.legend(fontsize=8, loc="upper left")

cmap = plt.get_cmap("tab10")
for c, g in sw.groupby("dc_class"):
    ax2.plot(g["T4"], g["PUE"], "o-", color=cmap((int(c) - 1) % 10), label=f"DC class {int(c)}")
ax2.set_xlabel("Chiller set point T4 (C)"); ax2.set_ylabel("PUE (kWh/kWh)")
ax2.set_title("Design penalty: PUE against the chiller set point")
ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_frontier.png", dpi=DPI); plt.close(fig)

# ---------------------------------------------------------------- warming figure
su = pd.read_csv(OUT / "warming_summary.csv")
base = su[su["scenario"] == "base"].iloc[0]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for name, color in (("constRH", "tab:red"), ("constTd", "tab:blue")):
    g = su[su["scenario"] == name].sort_values("dT")
    x = [0] + g["dT"].tolist()
    ax.plot(x, [base["PUE"]] + g["PUE"].tolist(), "o-", color=color,
            label=f"{'constant RH' if name == 'constRH' else 'constant dew point'}")
    ax2.plot(x, [base["WUE"]] + g["WUE"].tolist(), "o-", color=color,
             label=f"{'constant RH' if name == 'constRH' else 'constant dew point'}")
ax.set_xlabel("Warming (C)"); ax.set_ylabel("PUE (kWh/kWh)")
ax.set_title("World mean PUE under warming, current designs")
ax.legend(fontsize=9)
ax2.set_xlabel("Warming (C)"); ax2.set_ylabel("WUE (L/kWh)")
ax2.set_title("World mean WUE under warming")
ax2.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_warming.png", dpi=DPI); plt.close(fig)
print("figures written")
