# Cooling mode selection and the escalation ladder.
#
# Two policies live here.
#
# LEGACY (select_mode, run_with_escalation): port of select_mode_tx and run_with_escalation
# from the corrected Texas module, generalized to carry the per record pressure and the
# property set through to the model. The thresholds (free cooling curve, eq. 22 evaporative
# boundary, 37.5 C escalation) are those of the regional study. They are kept unchanged so
# the validation gates reproduce the Madrid and Texas baselines bit for bit. They are NOT
# the production policy any more: the free cooling curve gives negative humidity limits
# above 14.4 C, and the only feasibility test of the legacy policy (cold water above the
# dew point) lets an evaporative tower deliver water below the inlet wet bulb, which no
# direct contact tower can do.
#
# PRODUCTION (select_mode_economizer, run_policy): the standard economizer sequence written
# with explicit approach temperatures to the ambient state the tower actually works against:
#   dry (free) cooling on the 27/18 C loop     when T2  <= 18 - A_DRY   (dry bulb approach)
#   evaporative cooling on the 27/18 C loop    when Twb <= 18 - A_WET   (wet bulb approach)
#   chiller, tower on the design loop          otherwise, with the condenser water set point
#       reset upward hour by hour to T4 = max(T4base, ceil(Twb + A_WET)) when the design
#       set point cannot be held (condenser water reset, as a plant control does)
# The dry mode is preferred whenever it can hold the loop (the water preferring sequence);
# the energy only alternative, which never runs dry, is evaluated as a policy scenario in
# run/design_optimization.py. The residual ladder (infeasible record or air flow above VCAP)
# is kept for the rare states the approach rule does not catch.
import os

import numpy as np

from .tower import model_vec
from .properties import V1, V2_PARTIALP, twb_c

VCAP = 0.15   # m3/s per kW_ITE: practical air flow bound; the degenerate saturated states
              # where the reversible outlet collapses onto the inlet climb the ladder instead.

T_SUPPLY = 18.0   # C, cold water of the tower only loop (27/18)
T_RETURN = 27.0   # C, warm water of the tower only loop
LOOP_SPLIT = 24.0 # K, chiller loop split (T1 = T4 + 24)
# approach temperatures of the production policy (the sensitivity run sets DCF_A_WET and
# DCF_A_DRY in the environment; the manifest of each run records the values used)
A_WET = float(os.environ.get("DCF_A_WET", 3.0))   # K, cold water above the inlet wet bulb, sprayed modes
A_DRY = float(os.environ.get("DCF_A_DRY", 5.0))   # K, cold water above the inlet dry bulb, dry mode


def select_mode_economizer(T2, HR2, p, T4base=26.0, a_wet=A_WET, a_dry=A_DRY,
                           props=V2_PARTIALP, twb=None):
    """Production mode selection (economizer sequence with approach temperatures).

    Returns T1, T4, evapYN, chillingYN, mode (0 dry, 1 evaporative, 2 chiller) and the wet
    bulb. T4base is the station design set point; chiller records whose wet bulb plus
    approach exceeds it are reset to the first whole degree that can be held.
    """
    T2 = np.asarray(T2, float); HR2 = np.asarray(HR2, float)
    p = np.broadcast_to(np.asarray(p, float), T2.shape)
    if twb is None:
        twb = twb_c(T2, HR2, p, props=props)
    dry = T2 <= T_SUPPLY - a_dry
    wet = (~dry) & (twb <= T_SUPPLY - a_wet)
    tower_only = dry | wet
    T4c = np.maximum(float(T4base), np.ceil(twb + a_wet))
    T4 = np.where(tower_only, T_SUPPLY, T4c)
    T1 = np.where(tower_only, T_RETURN, T4c + LOOP_SPLIT)
    evap = np.where(dry, 0.0, 1.0)
    chill = np.where(tower_only, 0.0, 1.0)
    mode = np.where(dry, 0, np.where(wet, 1, 2))
    return T1, T4, evap, chill, mode, twb


def run_policy(T2, HR2, T4base=26.0, p=1e5, props=V2_PARTIALP, max_rungs=10,
               a_wet=A_WET, a_dry=A_DRY):
    """Run the production policy and the residual ladder. Same return keys as
    run_with_escalation plus 'twb' (inlet wet bulb) and 'reset' (degrees of condenser
    water reset above the design set point, chiller records only; the ladder rungs are
    counted in 'esc' as before)."""
    T2 = np.asarray(T2, float); HR2 = np.asarray(HR2, float)
    p = np.broadcast_to(np.asarray(p, float), T2.shape).copy()
    T1, T4, evap, chill, mode, twb = select_mode_economizer(
        T2, HR2, p, T4base=T4base, a_wet=a_wet, a_dry=a_dry, props=props)
    T1 = T1.copy(); T4 = T4.copy(); evap = evap.copy(); chill = chill.copy(); mode = mode.copy()
    T5, WUE, PUEnoIT, Tdew2, ex = model_vec(T1, T2, HR2, T2, T4, np.full_like(T2, 100.0),
                                            evap, chill, p=p, props=props, extras=True)
    esc = np.zeros(T2.shape, int)
    wet_ok = twb <= T_SUPPLY - a_wet
    T4c = np.maximum(float(T4base), np.ceil(twb + a_wet))
    for rung in range(1, max_rungs + 1):
        need = np.isnan(PUEnoIT) | (ex["VAmix5"] > VCAP)
        if not need.any():
            break
        idx = np.where(need)[0]
        was_dry = mode[idx] == 0
        was_wet = mode[idx] == 1
        to_wet = was_dry & wet_ok[idx]
        to_chill = (was_dry & ~wet_ok[idx]) | was_wet
        # dry -> evaporative (if the wet bulb allows) or chiller; evaporative -> chiller at
        # the reset set point; chiller -> one more degree
        T4n = np.where(to_wet, T_SUPPLY, np.where(to_chill, T4c[idx], T4[idx] + 1.0))
        T1n = np.where(to_wet, T_RETURN, T4n + LOOP_SPLIT)
        T4[idx] = T4n; T1[idx] = T1n
        evap[idx] = 1.0
        chill[idx] = np.where(to_wet, 0.0, 1.0)
        mode[idx] = np.where(to_wet, 1, 2)
        esc[idx] = rung
        t5, w, pn, td, e2 = model_vec(T1[idx], T2[idx], HR2[idx], T2[idx], T4[idx],
                                      np.full(len(idx), 100.0), evap[idx], chill[idx],
                                      p=p[idx], props=props, extras=True)
        T5[idx] = t5; WUE[idx] = w; PUEnoIT[idx] = pn
        for k in ("fan", "pump", "VAmix5"):
            ex[k][idx] = e2[k]
    reset = np.where(mode == 2, T4 - float(T4base), 0.0)
    return dict(T1=T1, T4=T4, evap=evap, chill=chill, mode=mode, esc=esc,
                T5=T5, WUE=WUE, PUE=PUEnoIT + 1.0, Tdew2=Tdew2, twb=twb, reset=reset,
                fan=ex["fan"], pump=ex["pump"], VAmix5=ex["VAmix5"],
                mL1=1.0 / (props.hL(T1) - props.hL(T4)))


def design_t4base_twb(twb, a_wet=A_WET, pct=99.5, floor_c=26.0):
    """Station design set point from the wet bulb: one whole degree above the approach
    requirement at the given percentile of the cleaned wet bulb series, floored at 26 C.
    Returns (T4base, twb percentile)."""
    tw = float(np.nanpercentile(np.asarray(twb, float), pct))
    return max(float(floor_c), float(np.floor(tw + a_wet) + 1.0)), tw


def select_mode(T2, HR2, T4base=26.0):
    """Mode selection with a per site chiller set point T4base (T1 = T4base + 24).

    Returns T1, T4, evapYN, chillingYN, mode. With T4base = 26 this reproduces the original
    Madrid select_mode exactly (50/26 chiller, 55/31 escalated).
    """
    T2 = np.asarray(T2, float); HR2 = np.asarray(HR2, float)
    T4b = float(T4base); T1b = T4b + 24.0
    T1 = np.empty_like(T2); T4 = np.empty_like(T2)
    evap = np.zeros_like(T2); chill = np.zeros_like(T2); mode = np.zeros(T2.shape, int)
    free = (T2 < 18) & (HR2 < (125 - 13 * T2 + 0.3 * T2**2))
    evapcond = (((HR2 <= 67.5) & (T2 < 24.3))
                | ((HR2 > 67.5) & (HR2 <= 86.4) & (T2 < 37.6 - 0.2 * HR2))
                | ((HR2 > 86.4) & (T2 < 69 - 0.6 * HR2)))
    is_free = free
    is_evap = (~free) & evapcond
    is_ch2 = (~free) & (~evapcond) & (T2 < 37.5)
    is_ch3 = (~free) & (~evapcond) & (T2 >= 37.5)
    T1[is_free] = 27; T4[is_free] = 18; evap[is_free] = 0; chill[is_free] = 0; mode[is_free] = 0
    T1[is_evap] = 27; T4[is_evap] = 18; evap[is_evap] = 1; chill[is_evap] = 0; mode[is_evap] = 1
    T1[is_ch2] = T1b; T4[is_ch2] = T4b; evap[is_ch2] = 1; chill[is_ch2] = 1; mode[is_ch2] = 2
    T1[is_ch3] = T1b + 5; T4[is_ch3] = T4b + 5; evap[is_ch3] = 1; chill[is_ch3] = 1; mode[is_ch3] = 3
    return T1, T4, evap, chill, mode


def run_with_escalation(T2, HR2, T4base=26.0, p=1e5, props=V1, max_rungs=6):
    """Run the model with the station design and rescue records the way an automatic
    control would. Two triggers per rung: (a) infeasible records (condensation or negative
    entropy production) and (b) records whose reversible air flow exceeds VCAP. Free records
    switch to evaporative, evaporative records switch to the chiller base mode, chiller
    records climb +1 C on T1 and T4. Returns a dict of arrays."""
    T2 = np.asarray(T2, float); HR2 = np.asarray(HR2, float)
    p = np.broadcast_to(np.asarray(p, float), T2.shape).copy()
    T1, T4, evap, chill, mode = select_mode(T2, HR2, T4base)
    T5, WUE, PUEnoIT, Tdew2, ex = model_vec(T1, T2, HR2, T2, T4, np.full_like(T2, 100.0),
                                            evap, chill, p=p, props=props, extras=True)
    esc = np.zeros(T2.shape, int)
    for rung in range(1, max_rungs + 1):
        need = np.isnan(PUEnoIT) | (ex["VAmix5"] > VCAP)
        if not need.any():
            break
        idx = np.where(need)[0]
        was_free = (evap[idx] == 0) & (chill[idx] == 0)
        was_evap = (evap[idx] == 1) & (chill[idx] == 0)
        evap[idx] = np.where(was_free, 1.0, evap[idx])
        T1[idx] = np.where(was_evap, T4base + 24.0, np.where(was_free, T1[idx], T1[idx] + 1.0))
        T4[idx] = np.where(was_evap, T4base, np.where(was_free, T4[idx], T4[idx] + 1.0))
        chill[idx] = np.where(was_free, 0.0, 1.0)
        mode[idx] = np.where(was_free, 1, np.where(was_evap, 2, mode[idx]))
        esc[idx] = rung
        t5, w, pn, td, e2 = model_vec(T1[idx], T2[idx], HR2[idx], T2[idx], T4[idx],
                                      np.full(len(idx), 100.0), evap[idx], chill[idx],
                                      p=p[idx], props=props, extras=True)
        T5[idx] = t5; WUE[idx] = w; PUEnoIT[idx] = pn
        for k in ("fan", "pump", "VAmix5"):
            ex[k][idx] = e2[k]
    ex["mL1"] = 1.0 / (props.hL(T1) - props.hL(T4))
    return dict(T1=T1, T4=T4, evap=evap, chill=chill, mode=mode, esc=esc,
                T5=T5, WUE=WUE, PUE=PUEnoIT + 1.0, Tdew2=Tdew2,
                fan=ex["fan"], pump=ex["pump"], VAmix5=ex["VAmix5"], mL1=ex["mL1"])
