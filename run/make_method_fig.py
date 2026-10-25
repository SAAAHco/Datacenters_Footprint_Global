# Supplementary Fig. 5: the model, its operating logic, the workflow and the evaluation chain.
# 180 mm wide, 300 dpi, Arial; written for reviewer 3 (methods figure). Output in results_world/Journal.
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcfootprint.core.modes import select_mode_economizer, A_WET, A_DRY  # noqa: E402

OUT = ROOT / "results_world" / "Journal"
RW = ROOT / "results_world"
import pandas as pd  # noqa: E402
_res = pd.read_csv(RW / "station_results.csv"); _res = _res[_res["screen_pass"] == True]  # noqa: E712
_ladder = int(_res["n_ladder"].sum()); _infeas = int(_res["n_infeasible"].sum())
_wp = float(_res["PUE"].mean()); _ww = float(_res["WUE"].mean())
_cm = pd.read_csv(RW / "closure_merkel_summary.csv")
_r = _cm[(_cm["scope"] == "world") & (_cm["Me"] == 1.5)].iloc[0]
_me_dp = float(_r["dPUE_mean"]); _me_dw = float(_r["dWUE_mean"])
_lk = pd.read_csv(RW / "koppen_loo.csv")
_cb_p = float(_lk["dPUE_self_pct"].abs().median()); _cb_w = float(_lk["dWUE_self_pct"].abs().median())
_gc = pd.read_csv(RW / "gsa_world_ci.csv").set_index(["scope", "metric"])
_ci = {m: (float(_gc.loc[("world", m), "production_ci_lo"]), float(_gc.loc[("world", m), "production_ci_hi"])) for m in ("PUE", "WUE")}
_sob = 1024 * (pd.read_csv(RW / "gsa_world_sobol.csv")["param"].nunique() + 2)
_K = int(pd.read_csv(RW / "dc_class_table.csv")["dc_class"].max())
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                     "axes.linewidth": 0.6, "savefig.dpi": 300})
MM = 1 / 25.4
MODE_COLS = ["#88CCEE", "#44AA99", "#CC6677", "#882255"]
GREY = "0.3"

fig = plt.figure(figsize=(180 * MM, 222 * MM))
ax_a = fig.add_axes([0.02, 0.575, 0.50, 0.40])
ax_b = fig.add_axes([0.60, 0.635, 0.37, 0.31])
ax_c = fig.add_axes([0.02, 0.01, 0.50, 0.535])
ax_d = fig.add_axes([0.545, 0.01, 0.445, 0.535])


def label(ax, letter, x=0.0, y=1.0):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def box(ax, x, y, text, w=None, fc="#F3F3F3", ec=GREY, fs=5.8, ls="-", bold_first=False):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=0.6, ls=ls))


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=0.7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0,
                                mutation_scale=7))


# ------------------------------------------------------------------ a: tower schematic
ax = ax_a
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
label(ax, "a", x=0.0, y=1.0)
# facility
ax.add_patch(FancyBboxPatch((0.15, 3.3), 1.4, 3.4, boxstyle="round,pad=0.05", fc="#FFF4E0", ec=GREY, lw=0.6))
ax.text(0.85, 5.0, "Facility\n$Q_{DC}$ = 1 kW\nsupply $T_4$\nreturn $T_1$\n(chiller in\nmodes 2, 3)", ha="center", va="center", fontsize=5.2, linespacing=1.3)
# tower
ax.add_patch(FancyBboxPatch((3.4, 3.0), 3.4, 4.6, boxstyle="round,pad=0.05", fc="#E6F2FA", ec=GREY, lw=0.7))
ax.text(5.1, 5.5, "Open evaporative\ncooling tower\n(reversible limit,\nentropy production = 0)", ha="center", va="center", fontsize=6, linespacing=1.3)
# stream 1: warm water in
arrow(ax, 1.55, 6.4, 3.4, 6.4, color="#B03A2E")
ax.text(2.5, 6.62, "1  warm water, $T_1$", ha="center", va="bottom", fontsize=5.4, color="#B03A2E")
ax.text(2.5, 6.18, "pump 0.21 MPa", ha="center", va="top", fontsize=4.9, color=GREY)
# stream 4: cold water out
arrow(ax, 3.4, 3.8, 1.55, 3.8, color="#1F618D")
ax.text(2.5, 4.02, "4  cold water, $T_4$", ha="center", va="bottom", fontsize=5.4, color="#1F618D")
# stream 2: ambient air in
arrow(ax, 9.6, 4.6, 6.8, 4.6, color="#1E8449")
ax.text(8.2, 4.82, "2  ambient air in", ha="center", va="bottom", fontsize=5.6, color="#1E8449")
ax.text(8.2, 4.38, "$T_2$, $RH_2$, $p$ (station hour)", ha="center", va="top", fontsize=5.4, color=GREY)
# stream 5: air out
arrow(ax, 5.1, 7.6, 5.1, 9.5, color="#1E8449")
ax.text(5.35, 8.55, "5  air out, $T_5$ from the closure,\n$RH_5$ = 100% when the sprays run\nfan $\\Delta p$ = 250 Pa on $\\dot V_{A5}$", ha="left", va="center", fontsize=5.4, color=GREY)
# stream 3: makeup water
arrow(ax, 9.6, 3.35, 6.8, 3.35, color="#1F618D")
ax.text(8.35, 3.55, "3  makeup water, $T_3$ = $T_2$", ha="center", va="bottom", fontsize=5.4, color="#1F618D")
# balances
ax.text(0.15, 2.5,
        "Mass:     $\\dot m_{L3}$ = ($w_5$ - $w_2$) $\\dot m_A$,   $w$ = 0.62186 $p_V$/($p$ - $p_V$)\n"
        "Energy:  $\\dot m_A$ = $Q$ / [($h_{A5}$ + $w_5 h_{V5}$) - ($h_{A2}$ + $w_2 h_{V2}$) - ($w_5$ - $w_2$) $h_{L3}$],   $Q$ = $Q_{DC}$(1 + $\\chi$/COP)\n"
        "Closure: entropy production $\\sigma$($T_5$) = 0, bisection on $T_5$ in [min($T_2$, $T_4$), $T_1$]; outlet capped at $T_1$\n"
        "PUE = 1 + [$\\Delta p_{fan} \\dot V_{A5}$ + $\\Delta p_{pump} \\dot V_{L}$ + chiller/COP($T_4$)] / $Q_{DC}$;   WUE = 3,600 $\\dot m_{L3}$ / $Q_{DC}$",
        ha="left", va="top", fontsize=5.3, linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.5))

# ------------------------------------------------------------------ b: mode map
ax = ax_b
T = np.arange(-15, 45.01, 0.25); RH = np.arange(0.5, 100.01, 0.5)
TT, HH = np.meshgrid(T, RH)
_, T4m, _, _, mode, _ = select_mode_economizer(TT.ravel(), HH.ravel(), 101325.0, T4base=26.0)
state = np.where((mode == 2) & (T4m > 26.0), 3, mode)
M = state.reshape(TT.shape)
ax.imshow(M, origin="lower", extent=(T[0], T[-1], RH[0], RH[-1]), aspect="auto",
          cmap=ListedColormap(MODE_COLS), vmin=-0.5, vmax=3.5, interpolation="nearest")
ax.set_xlabel("Ambient temperature $T_2$ (°C)")
ax.set_ylabel("Relative humidity $RH_2$ (%)")
ax.set_xlim(-15, 45); ax.set_ylim(0, 100)
handles = [Patch(fc=MODE_COLS[0], label=f"0  dry free cooling, 27/18 °C ($T_2 \\leq$ {18 - A_DRY:.0f} °C)"),
           Patch(fc=MODE_COLS[1], label=f"1  evaporative, 27/18 °C ($T_{{wb}} \\leq$ {18 - A_WET:.0f} °C)"),
           Patch(fc=MODE_COLS[2], label="2  chiller + evaporative, design loop 50/26 °C"),
           Patch(fc=MODE_COLS[3], label="3  chiller, set point reset above design")]
ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9, edgecolor="0.6", fontsize=5.4,
          handlelength=1.2, handleheight=0.9, borderpad=0.5)
ax.text(0.02, 0.98, f"Reset: $T_4$ = max(design, ceil($T_{{wb}}$ + {A_WET:.0f} K)); residual ladder for\n"
        "infeasible records or air flow > 0.15 m$^3$ s$^{-1}$ kW$^{-1}$", transform=ax.transAxes,
        ha="left", va="top", fontsize=5.2, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))
label(ax, "b", x=-0.22, y=1.06)

# ------------------------------------------------------------------ c: workflow
ax = ax_c
ax.set_xlim(0, 10); ax.set_ylim(0, 13.2); ax.axis("off")
label(ax, "c", x=0.0, y=1.0)
steps = [
    (12.45, "GHCNh hourly archive, 2022 to 2025: 24,072 stations"),
    (11.35, "All four years and the three sensors (T, dew point or RH,\nstation pressure): 8,017 stations"),
    (10.05, "One station per equal-area cell (12,364 km$^2$), fewest missing\nrecords: 3,379 stations, 99.2 M records (Section S1)"),
    (8.75, "Cleaning by physical rule: plausibility, one record per hour,\ndew point first, pressure trust ladder (25 hPa gate)"),
    (7.45, "Coverage screen ($\\geq$ 50% of hours, every month $\\geq$ 30%):\n2,081 stations, 139 countries, 65.2 M station hours"),
    (6.15, "Per-station design: $T_4$ = max(26, floor($p_{99.5}$($T_{wb}$) + 3) + 1) °C,\n$T_1$ = $T_4$ + 24 °C; station pressure"),
    (4.85, f"Hourly solve for every station hour: economizer sequence $\\rightarrow$\nbalances $\\rightarrow$ closure $\\rightarrow$ reset and ladder ({_ladder:,} ladder climbs, {_infeas} infeasible)"),
    (3.55, "Station products: PUE, WUE, mode occupancies, conditional\nfootprints, monthly climatology, design and pressure"),
]
for y, s in steps:
    box(ax, 5.0, y, s, fs=5.7)
ys = [y for y, _ in steps]
for y0, y1 in zip(ys[:-1], ys[1:]):
    h0 = 0.28 if "\n" not in steps[ys.index(y0)][1] else 0.46
    arrow(ax, 5.0, y0 - h0, 5.0, y1 + 0.46)
# analyses row
an = [(1.35, f"Classification and\ntransfer: Ward k = {_K},\nKDE convolution,\nleave-one-out\n(Sections S6, S7)"),
      (3.8, "Benchmark facility\n100 MW: ET0 and\nrainfall calendars,\nAqueduct, warming\n(Sections S9 to S11)"),
      (6.25, "Policies, archetypes,\nwater accounting,\nload profiles\n(Sections S12, S13)"),
      (8.7, f"Uncertainty:\nSobol {_sob:,},\nLHS 4,096\n(Section S3)")]
for x, s in an:
    box(ax, x, 1.35, s, fs=5.5, fc="#EAF4EA")
    arrow(ax, 5.0, 3.09, x, 2.35)

# ------------------------------------------------------------------ d: evaluation chain
ax = ax_d
ax.set_xlim(0, 10); ax.set_ylim(0, 13.2); ax.axis("off")
label(ax, "d", x=0.0, y=1.0)
ev = [
    (12.0, "1  Verification gates (eight automated tests)\nscalar oracle to 1e-6 on 2,000 conditions; Madrid 1.0405 / 1.0136\nwith 7 infeasible; Texas 601 escalated; WUE bit-invariant to the\nauxiliary term; wet bulb vs chart; approach rule and production\nphysics (outlet bound, condenser load) on 30,000 states", "#F3F3F3"),
    (9.95, "2  Configuration calibration (Madrid series, legacy rules)\npartial-pressure entropy reproduces occupancy 12.6/67.2/19.8/0.4\nexactly with 0 escalations; means -0.34% (PUE), +0.67% (WUE)", "#F3F3F3"),
    (8.2, f"3  Closure comparison (all stations, both sprayed modes)\nMerkel finite-transfer method, Me = 0.75 to 3.0, calibrated on nothing:\nworld PUE within {_me_dp:+.3f}, WUE within {_me_dw:+.3f} L kWh$^{{-1}}$ at Me = 1.5", "#F3F3F3"),
    (6.75, f"4  Convolution consistency\ndensity means against record-level means: {_cb_p:.2f}% PUE, {_cb_w:.2f}% WUE", "#F3F3F3"),
    (5.3, f"5  Uncertainty (11 parameters; Sobol {_sob:,}, LHS 4,096)\nworld PUE {_ci['PUE'][0]:.3f} to {_ci['PUE'][1]:.3f}; WUE {_ci['WUE'][0]:.3f} to {_ci['WUE'][1]:.3f} L kWh$^{{-1}}$", "#F3F3F3"),
    (3.5, f"6  Operational record (lower bound respected)\nfleet PUE 1.08 to 1.09 and WUE 1.04 above the world bound\n{_wp:.3f} / {_ww:.2f}; measured facilities above it by their\nexcluded loads (Section 3.6)", "#EAF4EA"),
    (1.35, "Open: metered hourly series of a single facility\n(none public; predicted against observed pending, code deposited)", "white"),
]
for y, s, fc in ev:
    box(ax, 5.0, y, s, fs=5.5, fc=fc, ls="--" if fc == "white" else "-")
hh = {1: 0.28, 2: 0.46, 3: 0.64, 4: 0.82, 5: 1.00}
for (y0, s0, _), (y1, s1, _) in zip(ev[:-1], ev[1:]):
    arrow(ax, 5.0, y0 - hh[s0.count("\n") + 1], 5.0, y1 + hh[s1.count("\n") + 1])
ax.text(9.85, 12.9, "increasing distance from the code $\\downarrow$", ha="right", va="center", fontsize=5.4, color=GREY, style="italic")

fig.savefig(OUT / "SuppFig5_method.png")
fig.savefig(OUT / "SuppFig5_method.pdf")
print("written", OUT / "SuppFig5_method.png")
