# World analysis stage: the old project's analyses (CLAUDE.md section 5) applied to the
# world run output, with the country layer on top (first two characters of the station id,
# FIPS 10-4 country code). Reads only results_world/ (produced from
# allData_World_cellSize_1.csv). No other data file is touched.
#
# Outputs (results_world/):
#   country_results.csv      per country: stations, mean PUE and WUE, occupancy, spreads
#   variance_decomp_world.csv  nested variance decomposition, world scale
#                            (country > station > year > month > hour slot > synoptic)
#   horizon_world.csv        averaging horizon curve, averaged over stations
#   annual_benchmark.csv     100 MW benchmark: annual energy overhead and water per station,
#                            monthly water calendar, ET0 correlation (FAO 56 Hargreaves,
#                            per station latitude)
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
res = res[res["screen_pass"] == True].copy()          # noqa: E712
res["country"] = res["STATION"].str[:2]
print(f"{len(res)} screened stations, {res['country'].nunique()} countries")

# ------------------------------------------------------------------ country aggregation
agg = res.groupby("country").agg(
    n_stations=("STATION", "size"),
    PUE=("PUE", "mean"), PUE_min=("PUE", "min"), PUE_max=("PUE", "max"),
    WUE=("WUE", "mean"), WUE_min=("WUE", "min"), WUE_max=("WUE", "max"),
    occ_free=("occ_free", "mean"), occ_evap=("occ_evap", "mean"),
    occ_chill=("occ_chill", "mean"), occ_chill_esc=("occ_chill_esc", "mean"),
    T4base_max=("T4base", "max"), esc_records=("n_escalated", "sum"),
).sort_values("n_stations", ascending=False)
agg.to_csv(OUT / "country_results.csv")

# --------------------------------------------- nested variance decomposition, world scale
# Telescoping sums of squares over: country > station > year > month > hour slot > residual
# (synoptic). Within-station components are computed station by station from the hourly
# parquet (one pass, low memory); the country and station components come from the per
# station means.
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
comp = {y: dict(Interannual=0.0, Seasonal=0.0, Diurnal=0.0, Synoptic=0.0) for y in ("PUE", "WUE")}
st_stats = []   # per station: n, mean PUE, mean WUE

hor_rows = []   # averaging horizon accumulators
HOR_WINDOWS = [("1h", 1.0), ("1D", 24.0), ("7D", 168.0), ("MS", 730.0), ("YS", 8760.0)]

for i in range(pf.num_row_groups):
    d = pf.read_row_group(i).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        n = len(g)
        rec = {"STATION": stn, "n": n}
        for y in ("PUE", "WUE"):
            v = g[y].to_numpy(np.float64)
            ok = ~np.isnan(v)
            v = v[ok]
            if not v.size:
                continue
            gg = g.loc[ok, ["Year", "Month", "Hour"]].copy()
            gg["v"] = v
            m_st = v.mean()
            m_yr = gg.groupby("Year")["v"].transform("mean").to_numpy()
            m_mo = gg.groupby(["Year", "Month"])["v"].transform("mean").to_numpy()
            m_sl = gg.groupby(["Year", "Month", "Hour"])["v"].transform("mean").to_numpy()
            comp[y]["Interannual"] += float(((m_yr - m_st) ** 2).sum())
            comp[y]["Seasonal"] += float(((m_mo - m_yr) ** 2).sum())
            comp[y]["Diurnal"] += float(((m_sl - m_mo) ** 2).sum())
            comp[y]["Synoptic"] += float(((v - m_sl) ** 2).sum())
            rec[f"mean_{y}"] = m_st
            rec[f"n_{y}"] = int(v.size)
        st_stats.append(rec)

        # averaging horizon per station (variance of block means / native variance)
        ts = pd.to_datetime(dict(year=g["Year"], month=g["Month"], day=g["Day"],
                                 hour=g["Hour"]))
        for y in ("PUE", "WUE"):
            s = pd.Series(g[y].to_numpy(np.float64), index=ts).dropna().sort_index()
            if len(s) < 100:
                continue
            v0 = s.var(ddof=1)
            if not v0 > 0:
                continue
            row = {"STATION": stn, "metric": y}
            for rule, hrs in HOR_WINDOWS:
                if rule == "1h":
                    row[rule] = 1.0
                    continue
                a = s.resample(rule).agg(["mean", "count"])
                med = a["count"].median()
                keep = a[a["count"] >= max(2, 0.5 * med)]["mean"].dropna()
                row[rule] = float(keep.var(ddof=1) / v0) if len(keep) >= 3 else np.nan
            hor_rows.append(row)
    if (i + 1) % 25 == 0:
        print(f"row group {i+1}/{pf.num_row_groups}, {time.time()-t0:.0f}s", flush=True)

st = pd.DataFrame(st_stats)
st["country"] = st["STATION"].str[:2]
vd_rows = []
for y in ("PUE", "WUE"):
    sub = st.dropna(subset=[f"mean_{y}"])
    n_i = sub[f"n_{y}"].to_numpy(float)
    m_i = sub[f"mean_{y}"].to_numpy(float)
    m0 = float((n_i * m_i).sum() / n_i.sum())
    cmeans = sub.groupby("country").apply(
        lambda x: pd.Series(dict(n=x[f"n_{y}"].sum(),
                                 m=(x[f"n_{y}"] * x[f"mean_{y}"]).sum() / x[f"n_{y}"].sum())),
        include_groups=False)
    ss_country = float((cmeans["n"] * (cmeans["m"] - m0) ** 2).sum())
    mc_of_st = sub["country"].map(cmeans["m"]).to_numpy(float)
    ss_station = float((n_i * (m_i - mc_of_st) ** 2).sum())
    parts = dict(Country=ss_country, Station=ss_station, **comp[y])
    sst = sum(parts.values())
    vd_rows.append(dict(metric=y, **{k: 100 * v / sst for k, v in parts.items()},
                        total_var=sst / n_i.sum()))
vd = pd.DataFrame(vd_rows)
vd.to_csv(OUT / "variance_decomp_world.csv", index=False)
print(vd.to_string(index=False))

hor = pd.DataFrame(hor_rows)
hor_summary = hor.groupby("metric")[[w for w, _ in HOR_WINDOWS]].median()
hor.to_csv(OUT / "horizon_world_stations.csv", index=False)
hor_summary.to_csv(OUT / "horizon_world.csv")
print(hor_summary.to_string())

# ------------------------------------------------- 100 MW benchmark and the ET0 calendar
# Monthly water and energy of the 100 MW continuous benchmark from the monthly climatology;
# ET0 by FAO 56 Hargreaves with per station latitude, from daily Tmin, Tmax, Tmean.
mon = pd.read_csv(OUT / "station_monthly.csv")
DIM = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
IT_MW = 100.0

lat_map = res.set_index("STATION")["lat"]

def et0_monthly(stn_df, lat_deg):
    """Monthly ET0 (mm) by Hargreaves from the hourly record of one station."""
    d = stn_df
    ts = pd.to_datetime(dict(year=d["Year"], month=d["Month"], day=d["Day"]))
    g = pd.DataFrame({"date": ts, "T": d["T"].to_numpy(np.float64)}).groupby("date")["T"]
    daily = pd.DataFrame({"tmax": g.max(), "tmin": g.min(), "tmean": g.mean()})
    daily = daily[(daily["tmax"] - daily["tmin"]) > 0]
    doy = daily.index.dayofyear.to_numpy()
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    dec = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1, 1)
    ws = np.arccos(x)
    Ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec)
                                          + np.cos(phi) * np.cos(dec) * np.sin(ws))
    et0 = 0.0023 * (daily["tmean"].to_numpy() + 17.8) * np.sqrt(
        daily["tmax"].to_numpy() - daily["tmin"].to_numpy()) * Ra / 2.45
    et0 = np.maximum(et0, 0)
    out = pd.DataFrame({"Month": daily.index.month, "et0": et0}).groupby("Month")["et0"].mean()
    return out * pd.Series(DIM)   # mean daily ET0 times days in month

ann_rows = []
pf = pq.ParquetFile(OUT / "world_hourly.parquet")
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=["STATION", "Year", "Month", "Day", "Hour", "T"]).to_pandas()
    for stn, g in d.groupby("STATION", sort=False):
        if stn not in lat_map.index:
            continue
        m = mon[mon["STATION"] == stn].set_index("Month")
        if len(m) < 12:
            continue
        hours = pd.Series({k: v * 24 for k, v in DIM.items()})
        water_ML = (m["WUE"] * IT_MW * 1000 * hours) / 1e6          # ML per month
        energy_GWh = (m["PUE"] * IT_MW * hours) / 1000              # GWh per month
        try:
            et0 = et0_monthly(g, float(lat_map.loc[stn]))
        except Exception:
            continue
        both = pd.concat([water_ML.rename("w"), et0.rename("e")], axis=1).dropna()
        r = float(both["w"].corr(both["e"])) if len(both) == 12 else np.nan
        s, w = water_ML.reindex(range(1, 13)), None
        summer = float(s.loc[[6, 7, 8]].sum()); winter = float(s.loc[[12, 1, 2]].sum())
        ann_rows.append(dict(
            STATION=stn, country=stn[:2], lat=float(lat_map.loc[stn]),
            energy_GWh_yr=float(energy_GWh.sum()), water_ML_yr=float(water_ML.sum()),
            overhead_GWh_yr=float(energy_GWh.sum() - IT_MW * 8766 / 1000),
            r_et0=r, summer_winter_water=(summer / winter if winter > 0 else np.nan),
            apr_sep_share=float(s.loc[[4, 5, 6, 7, 8, 9]].sum() / s.sum()),
        ))
    if (i + 1) % 25 == 0:
        print(f"[annual] row group {i+1}/{pf.num_row_groups}, {time.time()-t0:.0f}s", flush=True)

ann = pd.DataFrame(ann_rows)
ann.to_csv(OUT / "annual_benchmark.csv", index=False)
print(f"\n[done] {time.time()-t0:.0f}s  countries {agg.shape[0]}  "
      f"annual rows {len(ann)}  r_et0 median {ann['r_et0'].median():.2f}")
