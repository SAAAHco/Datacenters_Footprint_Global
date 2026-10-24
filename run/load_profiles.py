# Tier 3 item 10: load profiles. The model is linear in IT load at each instant, so a load
# profile L(t) turns the footprint means into load weighted means: intensity per kWh of ITE
# actually delivered. Three archetypes on local solar time (longitude / 15):
#   flat        constant load (the base case, AI training clusters)
#   inference   diurnal demand peaking 15:00 local, trough 03:00 (2:1 peak to trough)
#   night       the same curve inverted (load shifted to the cool hours, the operational
#               lever a scheduler could pull)
# Output: results_world/load_profiles.csv and printed summary (world and DC class means).
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_world"
t0 = time.time()

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].set_index("STATION")   # noqa: E712
dc = pd.read_csv(OUT / "dc_classes.csv").set_index("STATION")["dc_class"]
K = int(dc.max())
lon = res["lon"]

PROFILES = {
    "flat": lambda h: np.ones_like(h, dtype=float),
    "inference": lambda h: 0.75 + 0.25 * np.cos(2 * np.pi * (h - 15.0) / 24.0),
    "night": lambda h: 0.75 + 0.25 * np.cos(2 * np.pi * (h - 3.0) / 24.0),
}

acc = {}
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "Hour", "PUE", "WUE"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        if stn not in lon.index:
            continue
        h_loc = (g["Hour"].to_numpy(float) + float(lon.loc[stn]) / 15.0) % 24.0
        P = g["PUE"].to_numpy(float); W = g["WUE"].to_numpy(float)
        ok = np.isfinite(P) & np.isfinite(W)
        a = acc.setdefault(stn, {k: [0.0, 0.0, 0.0] for k in PROFILES})
        for name, fn in PROFILES.items():
            w = fn(h_loc[ok])
            a[name][0] += float((w * P[ok]).sum())
            a[name][1] += float((w * W[ok]).sum())
            a[name][2] += float(w.sum())
print(f"pass done ({time.time()-t0:.0f}s)")

rows = []
for stn, a in acc.items():
    r = dict(STATION=stn, dc_class=int(dc.get(stn, -1)))
    for name in PROFILES:
        s = a[name]
        r[f"PUE_{name}"] = s[0] / s[2]
        r[f"WUE_{name}"] = s[1] / s[2]
    rows.append(r)
lp = pd.DataFrame(rows)
lp.to_csv(OUT / "load_profiles.csv", index=False)

print("\nload weighted footprint (means over stations):")
for scope, g in [("world", lp)] + [(f"class {c}", lp[lp["dc_class"] == c]) for c in range(1, K + 1)]:
    print(f"  {scope}: flat PUE {g['PUE_flat'].mean():.4f} WUE {g['WUE_flat'].mean():.4f} | "
          f"inference {g['PUE_inference'].mean():.4f} / {g['WUE_inference'].mean():.4f} | "
          f"night {g['PUE_night'].mean():.4f} / {g['WUE_night'].mean():.4f}")
dP = 100 * (lp["WUE_night"] - lp["WUE_inference"]) / lp["WUE_inference"]
print(f"\nnight vs inference WUE: median {dP.median():+.2f}%  p5 {dP.quantile(0.05):+.2f}%  "
      f"p95 {dP.quantile(0.95):+.2f}%")
print(f"[done] {time.time()-t0:.0f}s")
