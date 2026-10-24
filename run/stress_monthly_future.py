# Tier 2 package B: peak on peak water stress, and the overlay under future projections.
# Part 1: joins every station to its sub basin in the Aqueduct 4.0 baseline_monthly layer
# (16,395 basins) and crosses the monthly facility water calendar with the monthly basin
# water stress: correlation, peak coincidence (is the facility's heaviest month inside the
# basin's three most stressed months), and the stress amplification factor (water weighted
# mean monthly stress score minus the unweighted mean).
# Part 2: joins the future_annual layer and re-runs the scarcity accounting under the 2030
# and 2050 projections (business as usual primary, optimistic and pessimistic as range).
# Outputs: results_world/stress_peak_on_peak.csv, stress_future.csv, future_summary tables
# printed, plus stress_future_summary.csv.
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
GDB = (ROOT / "aqueduct" / "Aqueduct40_waterrisk_download_Y2023M07D05" / "GDB"
       / "Aq40_Y2023D07M05.gdb")
t0 = time.time()

res = pd.read_csv(OUT / "station_water_stress.csv")   # has lat, lon, dc_class, bws (annual)
SCARCE_LABELS = ["High (40-80%)", "Extremely High (>80%)", "Arid and Low Water Use"]
res["scarce"] = res["bws_label"].isin(SCARCE_LABELS)
mon = pd.read_csv(OUT / "station_monthly.csv")
DIM = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
hours = pd.Series({m: d * 24 for m, d in DIM.items()})
water = mon.pivot_table(index="STATION", columns="Month", values="WUE")
water = water.mul(hours, axis=1) * 100 * 1000 / 1e6        # ML per month, 100 MW
pts = shapely.points(res["lon"].to_numpy(), res["lat"].to_numpy())


def join_layer(layer, columns):
    meta, _, wkb, fields = ogr_read(str(GDB), layer=layer, columns=columns)
    geoms = shapely.from_wkb(wkb)
    cols = dict(zip(meta["fields"], fields))
    tree = STRtree(geoms)
    gi = np.empty(len(pts), int); dist = np.zeros(len(pts))
    for i, pt in enumerate(pts):
        idx = tree.query(pt, predicate="within")
        if len(idx):
            gi[i] = int(idx[0])
        else:
            gi[i] = int(tree.nearest(pt))
            dist[i] = float(shapely.distance(geoms[gi[i]], pt))
    return {c: np.asarray(v)[gi] for c, v in cols.items()}, dist


# ------------------------------------------------------------ part 1: monthly stress
mcols = [f"bws_{m:02d}_score" for m in range(1, 13)]
mdat, dist1 = join_layer("baseline_monthly", ["pfaf_id"] + mcols)
S = np.column_stack([mdat[c] for c in mcols]).astype(float)   # score 0..5, sentinels < 0
S[S < 0] = np.nan
print(f"monthly layer joined ({time.time()-t0:.0f}s); nearest fallback {int((dist1>0).sum())}")

rows = []
for i, stn in enumerate(res["STATION"]):
    if stn not in water.index:
        continue
    W = water.loc[stn].reindex(range(1, 13)).to_numpy(float)
    s = S[i]
    ok = np.isfinite(s) & np.isfinite(W)
    if ok.sum() < 12 or np.nanstd(s) == 0 or np.nanstd(W) == 0:
        r = np.nan; coin = np.nan; amp = np.nan
    else:
        r = float(np.corrcoef(W, s)[0, 1])
        top3 = np.argsort(s)[-3:]
        coin = float(int(np.argmax(W)) in set(top3.tolist()))
        amp = float((W * s).sum() / W.sum() - s.mean())
    rows.append(dict(STATION=stn, dc_class=res["dc_class"].iloc[i],
                     bws_label=res["bws_label"].iloc[i], scarce=res["scarce"].iloc[i],
                     r_stress=r, peak_coincidence=coin, stress_amplification=amp,
                     s_annual_mean=float(np.nanmean(s))))
pk = pd.DataFrame(rows)
pk.to_csv(OUT / "stress_peak_on_peak.csv", index=False)
v = pk.dropna(subset=["r_stress"])
print(f"\npeak on peak ({len(v)} stations with seasonal stress data):")
print(f"  r(monthly water, monthly stress): median {v['r_stress'].median():+.2f}, "
      f"share positive {100 * (v['r_stress'] > 0).mean():.0f}%, "
      f"share > 0.3: {100 * (v['r_stress'] > 0.3).mean():.0f}%")
print(f"  peak coincidence (max water month in basin's top 3 stress months): "
      f"{100 * v['peak_coincidence'].mean():.0f}% of stations")
print(f"  stress amplification (water weighted minus mean score): "
      f"median {v['stress_amplification'].median():+.3f}, p90 {v['stress_amplification'].quantile(0.9):+.3f}")
print("  by DC class (median r, median amplification):")
for c, g in v.groupby("dc_class"):
    print(f"    class {int(c)}: r {g['r_stress'].median():+.2f}  "
          f"amp {g['stress_amplification'].median():+.3f}  "
          f"coincidence {100 * g['peak_coincidence'].mean():.0f}%")
sc = v[v["scarce"] == True]     # noqa: E712
print(f"  scarce basin stations only: r median {sc['r_stress'].median():+.2f}, "
      f"coincidence {100 * sc['peak_coincidence'].mean():.0f}%")

# ------------------------------------------------------------ part 2: future stress
fcols = []
for scn in ("bau", "opt", "pes"):
    for yr in ("30", "50"):
        fcols += [f"{scn}{yr}_ws_x_c", f"{scn}{yr}_ws_x_l"]
fdat, dist2 = join_layer("future_annual", ["pfaf_id"] + fcols)
fut = pd.DataFrame({"STATION": res["STATION"], "dc_class": res["dc_class"],
                    "bws_cat_now": res["bws_cat"], "bws_label_now": res["bws_label"],
                    "water_ML_yr": res["water_ML_yr"]})
for c in fcols:
    fut[c] = fdat[c]
fut.to_csv(OUT / "stress_future.csv", index=False)
print(f"\nfuture layer joined ({time.time()-t0:.0f}s)")

# scarce = category 3 (High) or 4 (Extremely High) or the arid special category (-1 in
# baseline; in the future layer arid basins carry category 4 labels "Extremely High"
# per Aqueduct convention, checked below)
print("future label values (bau50):", pd.Series(fut["bau50_ws_x_l"]).value_counts().to_dict())


def scarce_mask(cat_series, labels):
    lab = pd.Series(labels).astype(str)
    return (pd.to_numeric(cat_series, errors="coerce") >= 3) | lab.str.contains("Arid", case=False)


now_scarce = res["scarce"].to_numpy(bool)
summary = []
for scn in ("bau", "opt", "pes"):
    for yr in ("30", "50"):
        m = scarce_mask(fut[f"{scn}{yr}_ws_x_c"], fut[f"{scn}{yr}_ws_x_l"]).to_numpy()
        red = m & fut["dc_class"].isin([4, 5]).to_numpy()
        wshare = fut.loc[m, "water_ML_yr"].sum() / fut["water_ML_yr"].sum()
        summary.append(dict(scenario=f"{scn}20{yr}",
                            scarce_stations=int(m.sum()),
                            scarce_pct=100 * m.mean(),
                            newly_scarce=int((m & ~now_scarce).sum()),
                            exit_scarce=int((~m & now_scarce).sum()),
                            red_zone=int(red.sum()), red_zone_pct=100 * red.mean(),
                            water_share_pct=100 * wshare))
sm = pd.DataFrame(summary)
sm.to_csv(OUT / "stress_future_summary.csv", index=False)
print("\nscarcity accounting, baseline vs projections "
      f"(baseline: scarce {int(now_scarce.sum())} = {100*now_scarce.mean():.1f}%, "
      f"red zone 181 = 8.7%, water share 27.8%):")
print(sm.round(1).to_string(index=False))

mig = pd.crosstab(res["bws_label"],
                  pd.Series(fut["bau50_ws_x_l"]).astype(str).str.split(" \\(").str[0],
                  rownames=["baseline"], colnames=["bau 2050"])
print("\ncategory migration baseline to BAU 2050:")
print(mig.to_string())
print(f"[done] {time.time()-t0:.0f}s")
