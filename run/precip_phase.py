# Tier 2 package A, step 2: the supply side of the monsoon question.
# Correlates each station's monthly facility water calendar with the monthly precipitation
# climatology extracted from the raw GHCNh archive (both totals and wet hour frequency, the
# latter robust to reporting convention differences). Demand versus ET0 is established
# (in phase essentially everywhere); this asks whether water AVAILABILITY also peaks with
# demand, and where. Outputs: results_world/station_precip_phase.csv and printed summary.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_world"

pm = pd.read_csv(OUT / "station_precip_monthly.csv")
mon = pd.read_csv(OUT / "station_monthly.csv")
ann = pd.read_csv(OUT / "annual_benchmark.csv")[["STATION", "r_et0"]]
dc = pd.read_csv(OUT / "dc_classes.csv")[["STATION", "dc_class", "koppen", "lat"]]

DIM = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
hours = pd.Series({m: d * 24 for m, d in DIM.items()})
water = mon.pivot_table(index="STATION", columns="Month", values="WUE").mul(hours, axis=1)

tot = pm.groupby("STATION").agg(rep=("n_reported", "sum"), rec=("n_records", "sum"),
                                psum=("precip_sum", "sum"))
tot["rep_frac"] = tot["rep"] / tot["rec"]
usable = tot[(tot["rep_frac"] >= 0.05) & (tot["psum"] > 10.0)].index
print(f"{len(tot)} stations extracted; {len(usable)} usable "
      f"(excluded: {int((tot['rep_frac'] < 0.05).sum())} under 5% reporting, "
      f"{int(((tot['rep_frac'] >= 0.05) & (tot['psum'] <= 10.0)).sum())} with no measured rain)")

P = pm.pivot_table(index="STATION", columns="Month", values="precip_sum")
WET = (pm.assign(wf=pm["wet_records"] / pm["n_records"].clip(lower=1))
       .pivot_table(index="STATION", columns="Month", values="wf"))

rows = []
for stn in usable:
    if stn not in water.index:
        continue
    W = water.loc[stn].reindex(range(1, 13)).to_numpy(float)
    Pv = P.loc[stn].reindex(range(1, 13)).to_numpy(float)
    Wf = WET.loc[stn].reindex(range(1, 13)).to_numpy(float)
    ok = np.isfinite(W) & np.isfinite(Pv)
    if ok.sum() < 12:
        continue
    r_p = float(np.corrcoef(W, Pv)[0, 1]) if Pv.std() > 0 else np.nan
    r_w = float(np.corrcoef(W, Wf)[0, 1]) if np.isfinite(Wf).all() and Wf.std() > 0 else np.nan
    rows.append(dict(STATION=stn, r_precip=r_p, r_wet=r_w,
                     annual_precip_mm=float(Pv.sum() / 4.0)))
ph = pd.DataFrame(rows).merge(dc, on="STATION").merge(ann, on="STATION")
ph.to_csv(OUT / "station_precip_phase.csv", index=False)
v = ph.dropna(subset=["r_precip"])
print(f"\n{len(v)} stations analyzed")
print(f"r(monthly facility water, monthly precipitation): median {v['r_precip'].median():+.2f}")
print(f"  supply in phase (r > 0.3):   {100 * (v['r_precip'] > 0.3).mean():.0f}%")
print(f"  neutral (-0.3 to 0.3):       {100 * (v['r_precip'].between(-0.3, 0.3)).mean():.0f}%")
print(f"  supply anti phase (r < -0.3): {100 * (v['r_precip'] < -0.3).mean():.0f}%")
print(f"wet hour frequency version: median r {v['r_wet'].median():+.2f}, "
      f"agreement in sign with totals {100 * ((v['r_wet'] * v['r_precip']) > 0).mean():.0f}%")

print("\nby Koppen group:")
v["kg"] = v["koppen"].str[0]
for kg, g in v.groupby("kg"):
    print(f"  {kg} ({len(g)}): median r {g['r_precip'].median():+.2f}, "
          f"in phase {100 * (g['r_precip'] > 0.3).mean():.0f}%, "
          f"anti phase {100 * (g['r_precip'] < -0.3).mean():.0f}%")
print("\nkey classes:")
for k in ("Am", "Aw", "As", "Cwa", "Cwb", "Af", "Csa", "Csb", "Cfa", "BSh", "BWh"):
    g = v[v["koppen"] == k]
    if len(g) >= 5:
        print(f"  {k} ({len(g)}): median r {g['r_precip'].median():+.2f}, "
              f"in phase {100 * (g['r_precip'] > 0.3).mean():.0f}%, "
              f"anti phase {100 * (g['r_precip'] < -0.3).mean():.0f}%")
print("\nby DC class:")
for c, g in v.groupby("dc_class"):
    print(f"  class {int(c)} ({len(g)}): median r {g['r_precip'].median():+.2f}, "
          f"in phase {100 * (g['r_precip'] > 0.3).mean():.0f}%, "
          f"anti phase {100 * (g['r_precip'] < -0.3).mean():.0f}%")
both = v[(v["r_et0"] > 0.5)]
print(f"\nstations with demand strongly in phase with ET0 (r_et0 > 0.5): {len(both)}")
print(f"  of these, supply also in phase: {100 * (both['r_precip'] > 0.3).mean():.0f}%, "
      f"supply anti phase: {100 * (both['r_precip'] < -0.3).mean():.0f}%")
