# Tier 2 package A, step 1: extract precipitation for the 2,081 screened stations from the
# raw GHCNh psv archive (year folders on disk), aggregated to per station calendar month.
# Only the hourly 'precipitation' column is used (the multi hour accumulation columns are
# inconsistent across networks); two robust quantities are kept per station month:
#   precip_sum   sum of reported hourly precipitation (mm, missing treated as no report)
#   wet_records  count of records with precipitation > 0
#   n_records    all records, n_reported: records with a non missing precipitation value
# The analysis step (precip_phase.py) uses the monthly SHAPE, not absolute totals, so
# reporting convention differences between networks matter little; stations with almost no
# reported precipitation are excluded there and counted.
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"G:\Datacenter_Global")
OUT = ROOT / "results_world"
YEARS = ["2022", "2023", "2024", "2025"]


def one_station(stn):
    rows = []
    for y in YEARS:
        f = ROOT / y / f"GHCNh_{stn}_{y}.psv"
        if not f.exists():
            continue
        try:
            d = pd.read_csv(f, sep="|", usecols=["Month", "precipitation"],
                            engine="c", low_memory=False)
        except Exception as e:
            return stn, None, f"{y}:{type(e).__name__}"
        p = pd.to_numeric(d["precipitation"], errors="coerce")
        g = pd.DataFrame({"Month": d["Month"], "p": p})
        agg = g.groupby("Month").agg(
            precip_sum=("p", lambda x: float(np.nansum(x))),
            wet_records=("p", lambda x: int((x > 0).sum())),
            n_reported=("p", lambda x: int(x.notna().sum())),
            n_records=("p", "size"))
        agg["Year"] = int(y)
        rows.append(agg.reset_index())
    if not rows:
        return stn, None, "no files"
    allr = pd.concat(rows)
    out = allr.groupby("Month").agg(
        precip_sum=("precip_sum", "sum"), wet_records=("wet_records", "sum"),
        n_reported=("n_reported", "sum"), n_records=("n_records", "sum")).reset_index()
    out["STATION"] = stn
    return stn, out, None


def main():
    res = pd.read_csv(OUT / "station_results.csv")
    stations = res[res["screen_pass"] == True]["STATION"].tolist()   # noqa: E712
    print(f"{len(stations)} stations, {len(YEARS)} years", flush=True)
    t0 = time.time()
    done, fails, parts = 0, [], []
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one_station, s): s for s in stations}
        for fut in cf.as_completed(futs):
            stn, df, err = fut.result()
            done += 1
            if err:
                fails.append((stn, err))
            if df is not None:
                parts.append(df)
            if done % 200 == 0:
                print(f"{done}/{len(stations)}  {time.time()-t0:.0f}s", flush=True)
    allm = pd.concat(parts, ignore_index=True)
    allm = allm[["STATION", "Month", "precip_sum", "wet_records", "n_reported", "n_records"]]
    allm.to_csv(OUT / "station_precip_monthly.csv", index=False)
    print(f"wrote {len(allm)} station months for {allm['STATION'].nunique()} stations")
    print(f"failures: {len(fails)}", fails[:10])
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
