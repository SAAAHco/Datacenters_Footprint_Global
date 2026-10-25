# Approach temperatures and operating margins of the production run, for the supplementary
# material: the distribution of the cold water approach to the inlet wet bulb in the sprayed
# modes, the dry bulb margin in the dry mode, the condenser water reset, and the hours each
# rule binds, world and per class. Reads results_world/world_hourly.parquet.
# Outputs: approach_stats.csv (per class and world), approach_hist.csv (histograms).
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_world"
from dcfootprint.core.modes import A_WET, A_DRY, T_SUPPLY   # noqa: E402

t0 = time.time()
res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].set_index("STATION")   # noqa: E712
dc = pd.read_csv(OUT / "dc_classes.csv").set_index("STATION")["dc_class"]

bins_a = np.arange(0.0, 30.5, 0.5)          # approach bins (K)
bins_r = np.arange(-0.5, 15.5, 1.0)         # reset bins (K)
acc = {}


def A(c):
    return acc.setdefault(c, dict(hours=0, dry=0, evap=0, chill=0, reset=0, ladder=0,
                                  appr_evap=np.zeros(len(bins_a) - 1), appr_chill=np.zeros(len(bins_a) - 1),
                                  reset_hist=np.zeros(len(bins_r) - 1), margin_dry=np.zeros(len(bins_a) - 1),
                                  twb_sum=0.0, twb_max=-99.0, bind_wet=0, bind_dry=0))


pf = pq.ParquetFile(OUT / "world_hourly.parquet")
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "Twb", "mode", "reset", "esc"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        if stn not in res.index:
            continue
        c = int(dc.get(stn, 0))
        t4b = float(res.loc[stn, "T4base"])
        T = g["T"].to_numpy(float); tw = g["Twb"].to_numpy(float)
        mode = g["mode"].to_numpy(); reset = g["reset"].to_numpy(float); esc = g["esc"].to_numpy()
        T4 = np.where(mode <= 1, T_SUPPLY, t4b + reset)
        for cc in ({c, 0}):   # a set: the unclassified stations (c = 0) count once in the world row
            a = A(cc)
            a["hours"] += len(g)
            a["dry"] += int((mode == 0).sum()); a["evap"] += int((mode == 1).sum())
            a["chill"] += int((mode == 2).sum()); a["reset"] += int(((mode == 2) & (reset > 0)).sum())
            a["ladder"] += int((esc > 0).sum())
            a["appr_evap"] += np.histogram(np.clip(T4[mode == 1] - tw[mode == 1], 0, 30 - 1e-9), bins_a)[0]
            a["appr_chill"] += np.histogram(np.clip(T4[mode == 2] - tw[mode == 2], 0, 30 - 1e-9), bins_a)[0]
            a["reset_hist"] += np.histogram(np.clip(reset[mode == 2], 0, 14), bins_r)[0]
            a["margin_dry"] += np.histogram(np.clip(T_SUPPLY - T[mode == 0], 0, 30 - 1e-9), bins_a)[0]
            a["twb_sum"] += float(tw.sum()); a["twb_max"] = max(a["twb_max"], float(tw.max()))
            # hours within 1 K of a switching rule (where the approach assumption decides the mode)
            a["bind_wet"] += int((np.abs(tw - (T_SUPPLY - A_WET)) <= 1.0).sum())
            a["bind_dry"] += int((np.abs(T - (T_SUPPLY - A_DRY)) <= 1.0).sum())
print(f"pass done ({time.time()-t0:.0f}s)")


def wq(h, edges, q):
    cw = np.cumsum(h) / h.sum()
    k = min(int(np.searchsorted(cw, q)), len(h) - 1)
    return float(edges[k] + 0.5 * (edges[1] - edges[0]))


rows = []; hrows = []
for c in sorted(acc):
    a = acc[c]; H = a["hours"]
    rows.append(dict(scope="world" if c == 0 else f"class {c}", hours=H,
                     dry_pct=100 * a["dry"] / H, evap_pct=100 * a["evap"] / H, chill_pct=100 * a["chill"] / H,
                     reset_pct_of_hours=100 * a["reset"] / H,
                     reset_pct_of_chiller=100 * a["reset"] / max(a["chill"], 1),
                     reset_mean_K=float((a["reset_hist"][1:] * (bins_r[1:-1] + 0.5)).sum() / max(a["reset_hist"][1:].sum(), 1)),
                     ladder_pct=100 * a["ladder"] / H,
                     appr_evap_p50=wq(a["appr_evap"], bins_a, 0.5), appr_evap_p5=wq(a["appr_evap"], bins_a, 0.05),
                     appr_chill_p50=wq(a["appr_chill"], bins_a, 0.5), appr_chill_p5=wq(a["appr_chill"], bins_a, 0.05),
                     appr_chill_p95=wq(a["appr_chill"], bins_a, 0.95),
                     margin_dry_p50=wq(a["margin_dry"], bins_a, 0.5),
                     twb_mean=a["twb_sum"] / H, twb_max=a["twb_max"],
                     within1K_of_wet_rule_pct=100 * a["bind_wet"] / H,
                     within1K_of_dry_rule_pct=100 * a["bind_dry"] / H))
    for name, h, edges in (("approach_evap", a["appr_evap"], bins_a), ("approach_chill", a["appr_chill"], bins_a),
                           ("reset_K", a["reset_hist"], bins_r), ("margin_dry", a["margin_dry"], bins_a)):
        for k in range(len(h)):
            hrows.append(dict(scope="world" if c == 0 else f"class {c}", variable=name,
                              bin_lo=edges[k], bin_hi=edges[k + 1], hours=int(h[k])))
st = pd.DataFrame(rows); st.to_csv(OUT / "approach_stats.csv", index=False)
pd.DataFrame(hrows).to_csv(OUT / "approach_hist.csv", index=False)
pd.set_option("display.width", 250)
print(st.round(2).to_string(index=False))
print(f"[done] {time.time()-t0:.0f}s")
