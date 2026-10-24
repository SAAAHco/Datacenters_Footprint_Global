# Tier 1 point 4b: re-derivation of the practical air flow bound VCAP under the production
# configuration. The original 0.15 m3/s per kW was set under the legacy physics as "about
# twice the largest air flow of any healthy operating state in the series". The production
# configuration moved the entropy root, so the healthy flow distribution moved with it.
# Method: evaluate the laddered model on the (T, RH) grid for every (design, pressure level)
# group of stations, weight each cell by the class climate mass actually observed at those
# stations, and read the healthy (non escalated, feasible) flow distribution.
# Outputs: results_world/vcap_rederivation.csv and the printed summary.
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_world"

from dcfootprint.core.properties import V2_PARTIALP        # noqa: E402
from dcfootprint.core.modes import run_policy, VCAP            # noqa: E402

t0 = time.time()
T_E = np.arange(-40.0, 50.5, 0.5); RH_E = np.arange(0.5, 101.5, 1.0)
T_C = (T_E[:-1] + T_E[1:]) / 2; RH_C = (RH_E[:-1] + RH_E[1:]) / 2
P_STEP = 2500.0

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].set_index("STATION")   # noqa: E712

# histogram of climate mass per (design, pressure level) group
groups = {}
p_acc = {}
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
total_hours = 0
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "T", "RH", "p_pa"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        if stn not in res.index:
            continue
        total_hours += len(g)
        key = (float(res.loc[stn, "T4base"]),
               int(round(float(np.median(g["p_pa"])) / P_STEP)))
        h, _, _ = np.histogram2d(g["T"], g["RH"], bins=[T_E, RH_E])
        groups[key] = groups.get(key, 0) + h
print(f"{len(groups)} (design, pressure) groups, {total_hours:,} panel hours "
      f"({time.time()-t0:.0f}s)")

TT, RR = np.meshgrid(T_C, RH_C, indexing="ij")
flows = []; weights = []; escal = []
for (t4b, plev), H in groups.items():
    r = run_policy(TT.ravel(), RR.ravel(), T4base=t4b, p=plev * P_STEP,
                   props=V2_PARTIALP, max_rungs=12)
    w = H.ravel() / 1.0
    m = (w > 0) & np.isfinite(r["PUE"])
    flows.append(r["VAmix5"][m]); weights.append(w[m]); escal.append(r["esc"][m])
V = np.concatenate(flows); W = np.concatenate(weights); E = np.concatenate(escal)

healthy = E == 0
Vh = V[healthy]; Wh = W[healthy]
order = np.argsort(Vh)
cw = np.cumsum(Wh[order]) / Wh[order].sum()
q = lambda p_: float(Vh[order][np.searchsorted(cw, p_)])
vmax_healthy = float(Vh.max())
print(f"healthy flow distribution (m3/s per kW): p50 {q(0.5):.4f}  p99 {q(0.99):.4f}  "
      f"p99.9 {q(0.999):.4f}  max {vmax_healthy:.4f}")
vcap_new = 2.0 * vmax_healthy
print(f"re-derived bound: 2 x max healthy = {vcap_new:.3f}  (current VCAP {VCAP})")

# impact of changing the bound: climate mass whose unescalated flow lies between the bounds
lo_b, hi_b = sorted([vcap_new, VCAP])
mass_between = float(Wh[(Vh > lo_b) & (Vh <= hi_b)].sum())
share = mass_between / W.sum()
print(f"climate mass with healthy flow in ({lo_b:.3f}, {hi_b:.3f}]: {100*share:.4f}% "
      f"of panel hours (~{share*total_hours:,.0f} records would change escalation status)")

pd.DataFrame([dict(vcap_current=VCAP, healthy_p50=q(0.5), healthy_p99=q(0.99),
                   healthy_p999=q(0.999), healthy_max=vmax_healthy,
                   vcap_rederived=vcap_new, mass_between_pct=100 * share,
                   records_affected_est=share * total_hours,
                   total_hours=total_hours)]).to_csv(OUT / "vcap_rederivation.csv", index=False)
print(f"[done] {time.time()-t0:.0f}s")
