# World production run: clean allData_World_cellSize_1.csv with the locked policy, apply
# the per station design rule, run the corrected model with the escalation ladder under the
# v2 (Table 6, per site pressure) properties, and write per station results.
#
# Outputs (results_world/):
#   station_results.csv   one row per station: cleaning diagnostics, design point, screen
#                         status, aggregate footprint, mode occupancy, v1 comparison means
#   station_monthly.csv   per station calendar month climatology (screened stations)
#   world_hourly.parquet  cleaned hourly series with per record PUE, WUE, mode, esc
#   manifest.json         model version, property set, policy constants, input identity
#
# Run with the anaconda python (needs pyarrow):
#   C:\Users\workt\anaconda3\python.exe run\world_pipeline.py
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dcfootprint.core.properties import V2_PARTIALP, twb_c         # noqa: E402
from dcfootprint.core.modes import (run_policy, design_t4base_twb,  # noqa: E402
                                    A_WET, A_DRY, VCAP)
from dcfootprint.data import ghcnh                                 # noqa: E402

SRC = Path(os.environ.get("DCF_SRC", ROOT / "allData_World_cellSize_1.csv"))
WS_LIST = ROOT / "WS_list.xlsx"
OUT = Path(os.environ.get("DCF_OUT", ROOT / "results_world"))
OUT.mkdir(exist_ok=True, parents=True)

MAX_RUNGS = 10   # residual ladder depth (infeasible record or air flow above VCAP after the
                 # approach rule has set the mode and the condenser water set point)

if os.environ.get("DCF_SMOKE"):      # end to end smoke test on a truncated extract
    ghcnh.MIN_USABLE_HOURS = 1000
    ghcnh.MIN_MONTH_COVERAGE = 0.0

t0 = time.time()

# ---------------------------------------------------------------- stage 1: stream + clean
read_opts = pacsv.ReadOptions(block_size=1 << 27)
convert_opts = pacsv.ConvertOptions(column_types={
    "STATION": pa.string(),
    "Year": pa.int32(), "Month": pa.int32(), "Day": pa.int32(),
    "Hour": pa.int32(), "Minute": pa.int32(),
    "temperature": pa.float64(), "dew_point_temperature": pa.float64(),
    "station_level_pressure": pa.float64(), "relative_humidity": pa.float64(),
})

store = {}           # station -> list of cleaned station year DataFrames
raw_rows = {}        # station -> raw row count
carry = None
n_total = 0


def process_run(df):
    stn = df["STATION"].iloc[0]
    raw_rows[stn] = raw_rows.get(stn, 0) + len(df)
    cleaned = ghcnh.clean_block(df)
    if cleaned is not None and len(cleaned):
        store.setdefault(stn, []).append(cleaned)


reader = pacsv.open_csv(str(SRC), read_options=read_opts, convert_options=convert_opts)
for batch in reader:
    df = batch.to_pandas()
    if not len(df):
        continue
    n_total += len(df)
    if carry is not None:
        if df["STATION"].iloc[0] == carry["STATION"].iloc[-1]:
            df = pd.concat([carry, df], ignore_index=True)
        else:
            process_run(carry)
        carry = None
    codes = (df["STATION"] != df["STATION"].shift()).cumsum()
    groups = [g for _, g in df.groupby(codes, sort=False)]
    carry = groups.pop()
    for g in groups:
        process_run(g)
if carry is not None:
    process_run(carry)
print(f"[stage1] {n_total:,} raw rows, {len(store)} stations cleaned, "
      f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- stage 2: per station run
ws = pd.read_excel(WS_LIST, engine="openpyxl")
ws = ws.rename(columns={"Elevation (m)": "elev"}).set_index("WS_id")

schema = pa.schema([
    ("STATION", pa.string()), ("Year", pa.int16()), ("Month", pa.int8()),
    ("Day", pa.int8()), ("Hour", pa.int8()),
    ("T", pa.float32()), ("RH", pa.float32()), ("p_pa", pa.float32()),
    ("Twb", pa.float32()),
    ("PUE", pa.float32()), ("WUE", pa.float32()),
    ("mode", pa.int8()), ("reset", pa.int8()), ("esc", pa.int8()),
])
writer = pq.ParquetWriter(str(OUT / "world_hourly.parquet"), schema, compression="snappy")

rows_out = []
monthly_out = []
n_done = 0
for stn in sorted(store.keys()):
    d = pd.concat(store.pop(stn), ignore_index=True)
    n_usable = len(d)
    meta = ws.loc[stn] if stn in ws.index else None
    lat = float(meta["Latitude"]) if meta is not None else np.nan
    lon = float(meta["Longitude"]) if meta is not None else np.nan
    elev = float(meta["elev"]) if meta is not None else np.nan
    loc = str(meta["Location"]) if meta is not None else ""

    p_pa, p_policy, p_med, p_baro = ghcnh.resolve_pressure(
        d["p_hpa"].to_numpy(float), elev,
        t_mean_c=float(np.nanmean(d["T"].to_numpy(float))))
    passed, reasons = ghcnh.screen_station(d["Month"].to_numpy(), n_usable)

    row = dict(STATION=stn, lat=lat, lon=lon, elev=elev, location=loc,
               n_raw=raw_rows.get(stn, 0), n_usable=n_usable,
               p_policy=p_policy, p_median_hpa=round(p_med, 1) if np.isfinite(p_med) else np.nan,
               p_baro_hpa=round(p_baro, 1) if np.isfinite(p_baro) else np.nan,
               screen_pass=passed, screen_reasons=reasons)

    if passed:
        T2 = d["T"].to_numpy(float)
        HR2 = d["RH"].to_numpy(float)
        twb = twb_c(T2, HR2, p_pa, props=V2_PARTIALP)
        t4base, twb995 = design_t4base_twb(twb, a_wet=A_WET)
        td995 = float(np.nanpercentile(d["Td_eff"].to_numpy(float), ghcnh.TD_DESIGN_PCT))
        r = run_policy(T2, HR2, T4base=t4base, p=p_pa, props=V2_PARTIALP,
                       max_rungs=MAX_RUNGS, a_wet=A_WET, a_dry=A_DRY)
        ok = ~np.isnan(r["PUE"]) & ~np.isnan(r["WUE"])

        # four operating states: dry, evaporative, chiller at the design set point, chiller
        # with the condenser water reset above it (the climate forced escalation)
        state = np.where(r["mode"] == 2, np.where(r["reset"] > 0, 3, 2), r["mode"])
        occ = [float((state == m).mean()) for m in range(4)]
        rs = r["reset"][state == 3]
        row.update(
            Td_p995=round(td995, 2), Twb_p995=round(twb995, 2), T4base=t4base,
            n_infeasible=int((~ok).sum()),
            n_escalated=int((state == 3).sum()),
            max_rung=int(rs.max()) if rs.size else 0,
            reset_mean_K=float(rs.mean()) if rs.size else 0.0,
            n_ladder=int((r["esc"] > 0).sum()),
            PUE=float(r["PUE"][ok].mean()), WUE=float(r["WUE"][ok].mean()),
            PUE_std=float(r["PUE"][ok].std()), WUE_std=float(r["WUE"][ok].std()),
            occ_free=occ[0], occ_evap=occ[1], occ_chill=occ[2], occ_chill_esc=occ[3],
            Twb_mean=float(np.mean(twb)), Twb_max=float(np.max(twb)),
        )
        for m, name in ((0, "free"), (1, "evap"), (2, "chill"), (3, "chillesc")):
            sel = (state == m) & ok
            row[f"PUE_{name}"] = float(r["PUE"][sel].mean()) if sel.any() else np.nan
            row[f"WUE_{name}"] = float(r["WUE"][sel].mean()) if sel.any() else np.nan

        dm = pd.DataFrame({"Month": d["Month"].to_numpy(), "PUE": r["PUE"], "WUE": r["WUE"],
                           "T": T2, "RH": HR2})
        g = dm.groupby("Month").agg(PUE=("PUE", "mean"), WUE=("WUE", "mean"),
                                    T=("T", "mean"), RH=("RH", "mean"), n=("PUE", "size"))
        for m, gr in g.iterrows():
            monthly_out.append(dict(STATION=stn, Month=int(m), PUE=gr["PUE"], WUE=gr["WUE"],
                                    T=gr["T"], RH=gr["RH"], n=int(gr["n"])))

        tbl = pa.table({
            "STATION": pa.array([stn] * n_usable, pa.string()),
            "Year": d["Year"].to_numpy(np.int16), "Month": d["Month"].to_numpy(np.int8),
            "Day": d["Day"].to_numpy(np.int8), "Hour": d["Hour"].to_numpy(np.int8),
            "T": d["T"].to_numpy(np.float32), "RH": d["RH"].to_numpy(np.float32),
            "p_pa": p_pa.astype(np.float32), "Twb": twb.astype(np.float32),
            "PUE": r["PUE"].astype(np.float32), "WUE": r["WUE"].astype(np.float32),
            "mode": r["mode"].astype(np.int8), "reset": r["reset"].astype(np.int8),
            "esc": r["esc"].astype(np.int8),
        }, schema=schema)
        writer.write_table(tbl)

    rows_out.append(row)
    n_done += 1
    if n_done % 250 == 0:
        print(f"[stage2] {n_done} stations, {time.time()-t0:.0f}s", flush=True)

writer.close()
res = pd.DataFrame(rows_out)
res.to_csv(OUT / "station_results.csv", index=False)
pd.DataFrame(monthly_out).to_csv(OUT / "station_monthly.csv", index=False)

manifest = dict(
    generated="run world_pipeline.py",
    input=dict(path=str(SRC), size_bytes=os.path.getsize(SRC), rows=n_total),
    model=dict(package_version="0.2.0", properties="v2-table6-partialp",
               aux_term="corrected dp*Vdot (250 Pa fan, 0.21 MPa pump)",
               closure="reversible entropy null, bisection 60 iters",
               policy=dict(name="water preferring economizer sequence with approach temperatures",
                           a_wet_K=A_WET, a_dry_K=A_DRY,
                           dry="T2 <= 18 - a_dry on the 27/18 loop",
                           evaporative="Twb <= 18 - a_wet on the 27/18 loop",
                           chiller="T4 = max(T4base, ceil(Twb + a_wet)), T1 = T4 + 24",
                           wet_bulb="adiabatic saturation temperature with the model's own fits"),
               ladder=f"residual: VCAP {VCAP} m3/s/kW or infeasible, max_rungs {MAX_RUNGS}",
               design_rule="T4base = max(26, floor(p99.5 Twb + a_wet)+1), T1 = T4 + 24",
               states="occ_chill = chiller at design set point, occ_chill_esc = chiller with reset above it"),
    cleaning=dict(T_valid=ghcnh.T_VALID, Td_over_T_tol=ghcnh.TD_OVER_T_TOL,
                  RH_valid="(0, 100]", p_valid_hpa=ghcnh.P_VALID_HPA,
                  p_trust_hpa=ghcnh.P_TRUST_HPA, p_record_window_hpa=ghcnh.P_RECORD_WINDOW_HPA,
                  dedup="one per station hour: completeness, then minute circular distance, then minute",
                  humidity="Td primary (Magnus), reported RH fallback",
                  screen=dict(min_hours=ghcnh.MIN_USABLE_HOURS,
                              min_month_coverage=ghcnh.MIN_MONTH_COVERAGE)),
)
with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)

npass = int(res["screen_pass"].sum())
print(f"\n[done] stations {len(res)}, screened in {npass}, out {len(res)-npass}, "
      f"{time.time()-t0:.0f}s")
if npass:
    r = res[res["screen_pass"]]
    print(f"[done] world mean PUE {r['PUE'].mean():.4f}  WUE {r['WUE'].mean():.4f}  "
          f"(station level, unweighted)")
    print(f"[done] occupancy means: free {r['occ_free'].mean()*100:.1f}%  "
          f"evap {r['occ_evap'].mean()*100:.1f}%  chill {r['occ_chill'].mean()*100:.1f}%  "
          f"esc {r['occ_chill_esc'].mean()*100:.1f}%")
    print(f"[done] reset records total {int(r['n_escalated'].sum()):,}  "
          f"max reset {int(r['max_rung'].max())} K  ladder climbs {int(r['n_ladder'].sum()):,}  "
          f"infeasible left {int(r['n_infeasible'].sum())}")
