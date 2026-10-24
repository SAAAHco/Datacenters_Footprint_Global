# GHCNh adapter: cleaning policy for the world extract (allData_World_cellSize_1.csv).
#
# Policy locked with the author on 2026-08-25:
#   QC        record level plausibility filters (the raw QC flags are source dependent and
#             sometimes absent, so they cannot be relied on):
#               T outside [-75, 60] C            -> T invalid
#               Td outside [-100, T + 0.5] C     -> Td invalid
#               reported RH outside (0, 100] %   -> RH invalid
#               p outside [450, 1085] hPa        -> p invalid
#   Dedup     one record per station hour: highest completeness score over (T, Td, RH, p),
#             tie break smallest circular distance of the minute to the top of the hour,
#             then earliest minute.
#   Humidity  dew point is primary (GHCNh RH is itself derived from T and Td): RH is
#             recomputed from (T, Td) with the Magnus form used by the model, Td capped at T
#             so RH <= 100. Reported RH is the fallback when Td is invalid.
#   Pressure  per station trust ladder:
#               channel trusted when the station has >= 100 plausible pressure records and
#               |median(p) - barometric(elevation)| <= 25 hPa (elevation sentinels 999.9,
#               8191 and 9999 are treated as missing metadata):
#                 trusted   -> record p, gap filled with the station median when the record
#                              deviates more than 40 hPa from the median or is missing
#                 untrusted with valid elevation -> constant barometric pressure
#                 untrusted without elevation    -> constant station median (unverified),
#                              or standard 1013.25 hPa when there is no usable channel
#   Screen    a station enters the panel only if it has >= 17,532 usable hours (50% of the
#             2022 to 2025 hours) and every calendar month (pooled over the four years) has
#             >= 30% of its potential hours usable.
#   Design    T4base = max(26, floor(p99.5 of the usable dew point series) + 1), T1 = T4 + 24
#             (the robust percentile replacement of the series maximum rule).
import numpy as np
import pandas as pd

from ..core.properties import pvsat_pa, tdew_c

T_VALID = (-75.0, 60.0)
TD_MIN = -100.0
TD_OVER_T_TOL = 0.5
P_VALID_HPA = (450.0, 1085.0)
P_RECORD_WINDOW_HPA = 40.0
P_TRUST_HPA = 25.0
P_MIN_RECORDS = 100
ELEV_SENTINELS = (999.9, 8191.0, 9999.0)
MIN_USABLE_HOURS = 17532          # 50% of the 35,064 hours in 2022 to 2025
MIN_MONTH_COVERAGE = 0.30
TD_DESIGN_PCT = 99.5
HOURS_PER_MONTH = {               # potential hours per calendar month, 2022 to 2025 pooled
    1: 4 * 31 * 24, 2: (28 * 3 + 29) * 24, 3: 4 * 31 * 24, 4: 4 * 30 * 24,
    5: 4 * 31 * 24, 6: 4 * 30 * 24, 7: 4 * 31 * 24, 8: 4 * 31 * 24,
    9: 4 * 30 * 24, 10: 4 * 31 * 24, 11: 4 * 30 * 24, 12: 4 * 31 * 24,
}


def p_barometric_hpa(elev_m, t_mean_c=15.0):
    """Barometric pressure from elevation, U.S. Standard Atmosphere 1976 convention.

    Hydrostatic equilibrium dp = -rho g dz with the ideal gas rho = p/(R T), integrated
    at constant temperature: p = p0 exp(-g z / (R T)), with p0 = 101325 Pa at sea level,
    g = 9.81 m/s2, R = 287 J/(kg K), and T(K) = 273.16 + T(degC) the station mean air
    temperature (project convention agreed 2026; falls back to 15 degC when no mean is
    available, i.e. the standard sea level temperature).
    """
    t = t_mean_c if (t_mean_c is not None and np.isfinite(t_mean_c)) else 15.0
    return 1013.25 * np.exp(-9.81 * elev_m / (287.0 * (273.16 + t)))


def clean_block(df):
    """Plausibility filter, humidity resolution and hourly dedup for one station year block.

    df columns: Year, Month, Day, Hour, Minute, temperature, dew_point_temperature,
    station_level_pressure, relative_humidity. Returns a reduced dataframe with one row per
    usable station hour: Year, Month, Day, Hour, T, RH, Td_eff, p_hpa (p may be NaN).
    """
    T = df["temperature"].to_numpy(float)
    Td = df["dew_point_temperature"].to_numpy(float)
    RH = df["relative_humidity"].to_numpy(float)
    P = df["station_level_pressure"].to_numpy(float)
    Mi = df["Minute"].to_numpy(float)

    t_ok = (T >= T_VALID[0]) & (T <= T_VALID[1])
    T = np.where(t_ok, T, np.nan)
    td_ok = t_ok & (Td >= TD_MIN) & (Td <= T + TD_OVER_T_TOL)
    Td = np.where(td_ok, Td, np.nan)
    rh_ok = (RH > 0.0) & (RH <= 100.0)
    RH = np.where(rh_ok, RH, np.nan)
    p_ok = (P >= P_VALID_HPA[0]) & (P <= P_VALID_HPA[1])
    P = np.where(p_ok, P, np.nan)

    # humidity resolution: Td primary, reported RH fallback
    with np.errstate(invalid="ignore", divide="ignore"):
        Td_cap = np.minimum(Td, T)
        rh_from_td = 100.0 * pvsat_pa(Td_cap) / pvsat_pa(T)
        RH_final = np.where(~np.isnan(Td), rh_from_td, RH)
        pV = RH_final / 100.0 * pvsat_pa(T)
        Td_eff = np.where(~np.isnan(Td), Td_cap,
                          np.where(RH_final >= 100.0, T, tdew_c(pV)))

    usable = ~np.isnan(T) & ~np.isnan(RH_final)
    if not usable.any():
        return None

    score = ((~np.isnan(T)).astype(np.int8) + (~np.isnan(Td)).astype(np.int8)
             + (~np.isnan(RH)).astype(np.int8) + (~np.isnan(P)).astype(np.int8))
    circ = np.minimum(Mi, 60.0 - Mi)

    out = pd.DataFrame({
        "Year": df["Year"].to_numpy(np.int16), "Month": df["Month"].to_numpy(np.int8),
        "Day": df["Day"].to_numpy(np.int8), "Hour": df["Hour"].to_numpy(np.int8),
        "T": T, "RH": RH_final, "Td_eff": Td_eff, "p_hpa": P,
        "_score": score, "_circ": circ, "_minute": Mi,
    })
    out = out[usable]
    out = out.sort_values(["Year", "Month", "Day", "Hour", "_score", "_circ", "_minute"],
                          ascending=[True, True, True, True, False, True, True], kind="stable")
    out = out.drop_duplicates(subset=["Year", "Month", "Day", "Hour"], keep="first")
    out = out.drop(columns=["_score", "_circ", "_minute"])
    for c in ("T", "RH", "Td_eff", "p_hpa"):
        out[c] = out[c].astype(np.float32)
    return out


def resolve_pressure(p_hpa, elev_m, t_mean_c=np.nan):
    """Apply the station pressure trust ladder. Returns (p_pa, policy, p_median, p_baro).

    t_mean_c: station mean air temperature in degC, used in the NASA 1976 barometric
    formula (see p_barometric_hpa)."""
    valid = p_hpa[~np.isnan(p_hpa)]
    p_med = float(np.median(valid)) if valid.size >= P_MIN_RECORDS else np.nan
    elev_valid = (elev_m is not None and np.isfinite(elev_m)
                  and not any(abs(elev_m - s) < 0.05 for s in ELEV_SENTINELS)
                  and -430.0 <= elev_m <= 6000.0)
    p_baro = p_barometric_hpa(elev_m, t_mean_c) if elev_valid else np.nan

    if np.isfinite(p_med) and elev_valid and abs(p_med - p_baro) <= P_TRUST_HPA:
        p = np.where(np.isnan(p_hpa) | (np.abs(p_hpa - p_med) > P_RECORD_WINDOW_HPA),
                     p_med, p_hpa)
        policy = "record"
    elif elev_valid:
        p = np.full_like(p_hpa, p_baro)
        policy = "barometric"
    elif np.isfinite(p_med):
        p = np.full_like(p_hpa, p_med)
        policy = "median-unverified"
    else:
        p = np.full_like(p_hpa, 1013.25)
        policy = "default"
    return p * 100.0, policy, p_med, p_baro


def screen_station(month, n_usable):
    """Coverage screen. month: int array of the usable hours. Returns (passed, reasons)."""
    reasons = []
    if n_usable < MIN_USABLE_HOURS:
        reasons.append(f"hours {n_usable} < {MIN_USABLE_HOURS}")
    mc = pd.Series(month).value_counts()
    worst = min(mc.get(m, 0) / HOURS_PER_MONTH[m] for m in range(1, 13))
    if worst < MIN_MONTH_COVERAGE:
        reasons.append(f"worst month coverage {worst:.2f} < {MIN_MONTH_COVERAGE}")
    return len(reasons) == 0, "; ".join(reasons)


def design_t4base(td_eff):
    """T4base from the robust dew point percentile rule."""
    td995 = float(np.nanpercentile(td_eff, TD_DESIGN_PCT))
    return max(26.0, float(np.floor(td995) + 1.0)), td995
