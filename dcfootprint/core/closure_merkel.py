# Merkel closure of the cooling tower: the standard finite transfer model of the fill
# literature (Merkel 1925; Kloppers and Kroger 2005), used as the independent check of the
# reversible closure of tower.py.
#
# The production model places the outlet air where the entropy production of the tower
# vanishes, which is the reversible limit of an infinitely large exchanger. The Merkel
# method instead gives a tower of finite size, characterized by its Merkel number
#     Me = KaV/L = cpw * integral from T4 to T1 of dTw / (h_sat(Tw) - h_a(Tw)),
# with h_sat the enthalpy of saturated air at the local water temperature and h_a the air
# enthalpy along the counterflow operating line h_a(Tw) = h2 + (L cpw / G) (Tw - T4), the
# Lewis factor taken as one and the evaporated water neglected in the energy balance. Here
# the loop temperatures T1 and T4 are prescribed by the operating policy, so the unknown is
# the air flow G that a tower of the given Me needs to hold the loop at the ambient state.
# The larger the air flow, the flatter the operating line and the smaller the Me required;
# the smallest tower that can do the job at all has Me_min = Me(G -> infinity), and the
# wet bulb limit appears by itself: Me_min is finite only when h_sat(T4) exceeds h2, that
# is, when the cold water sits above the inlet wet bulb. Given G, the outlet air is
# saturated at the enthalpy h5 = h2 + Q/G (the Merkel outlet assumption, shared with the
# production model), from which the evaporated water, the fan power and WUE follow exactly
# as in tower.py. The dry (unsprayed) mode has no Merkel analogue and keeps the reversible
# solution.
import numpy as np

from .properties import V2_PARTIALP, RA, RV, K0, EPS_W, pvsat_pa
from .tower import DP_FAN, DP_PUMP, cop, LEGACY_PHYSICS

N_SIMPSON = 32      # intervals of the Merkel integral over the water temperature range
N_BISECT = 50       # bisection steps on the operating line slope


def _wsat(T, p):
    ps = pvsat_pa(T)
    return EPS_W * ps / (p - ps)


def _hsat(T, p, props):
    return props.hA(T) + _wsat(T, p) * props.hV(T)


def merkel_sprayed(T1, T2, HR2, T4, p, Me, props=V2_PARTIALP, chill=0.0, legacy=None):
    """Merkel solution for sprayed records. Arrays broadcast over records. chill marks
    chiller-mode records, whose tower load is Q_DC (1 + 1/COP) under the production physics
    (legacy: Q_DC in every mode, as in tower.model_vec).

    Returns dict with WUE (L per kWh), fan (kW per kW), pump (kW per kW), VAmix5 (m3/s per
    kW), T5 (C), G (kg dry air per s per kW), Me_min, feasible (bool). Infeasible records
    (Me below Me_min, the tower too small to hold T4 at this ambient state) carry NaN.
    """
    T1 = np.asarray(T1, float); T2 = np.asarray(T2, float); HR2 = np.asarray(HR2, float)
    T4 = np.asarray(T4, float); p = np.asarray(p, float); Me = np.asarray(Me, float)
    T1, T2, HR2, T4, p, Me = np.broadcast_arrays(T1, T2, HR2, T4, p, Me)
    legacy = LEGACY_PHYSICS if legacy is None else bool(legacy)
    chill = np.broadcast_to(np.asarray(chill, float), T1.shape)
    q = np.ones_like(T1) if legacy else 1.0 + chill / cop(T4)   # tower load per kW of computing
    pV2 = HR2 / 100.0 * pvsat_pa(T2)
    w2 = EPS_W * pV2 / (p - pV2)
    h2 = props.hA(T2) + w2 * props.hV(T2)
    cpw = props.hL(T4 + 1.0) - props.hL(T4)           # kJ/kg/K of the liquid fit
    hL1 = props.hL(T1); hL4 = props.hL(T4)
    mL = q / (hL1 - hL4)                               # kg/s per kW of computing
    # Simpson nodes over the water temperature range
    s = np.linspace(0.0, 1.0, N_SIMPSON + 1)           # (n+1,)
    Tw = T4[..., None] + s * (T1 - T4)[..., None]      # (..., n+1)
    hs = _hsat(Tw, p[..., None], props)                # (..., n+1)
    drive0 = hs - h2[..., None]                        # driving enthalpy at infinite air flow
    wts = np.ones(N_SIMPSON + 1); wts[1:-1:2] = 4.0; wts[2:-1:2] = 2.0
    dT = (T1 - T4) / N_SIMPSON

    def me_of(x):
        # x = Q/G in kJ per kg dry air: enthalpy rise of the air over the tower
        den = drive0 - x[..., None] * s
        den = np.where(den > 1e-9, den, 1e-9)
        return cpw * dT / 3.0 * (wts / den).sum(-1)

    feasible = drive0.min(-1) > 1e-9                   # T4 above the wet bulb, in enthalpy terms
    me_min = np.where(feasible, me_of(np.zeros_like(T1)), np.inf)
    feasible = feasible & (me_min <= Me)
    # pinch: largest x that keeps the driving force positive everywhere
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s > 0, drive0 / np.where(s > 0, s, 1.0), np.inf)
    x_pinch = np.where(feasible, ratio.min(-1), 1.0)
    lo = np.zeros_like(T1); hi = x_pinch * (1.0 - 1e-9)
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        too_small = me_of(mid) < Me                    # Me rises with x
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    x = 0.5 * (lo + hi)
    G = q / x                                          # kg dry air per s per kW of computing
    h5 = h2 + x
    # saturated outlet at enthalpy h5
    tlo = np.minimum(T2, T4) - 5.0; thi = T1 + 0.0
    for _ in range(N_BISECT):
        tm = 0.5 * (tlo + thi)
        high = _hsat(tm, p, props) > h5
        thi = np.where(high, tm, thi); tlo = np.where(high, tlo, tm)
    T5 = 0.5 * (tlo + thi)
    pV5 = pvsat_pa(T5)
    w5 = EPS_W * pV5 / (p - pV5)
    mL3 = G * (w5 - w2)
    WUE = 3600.0 * mL3
    VAmix5 = G * RA * (K0 + T5) / (p - pV5)
    fan = VAmix5 * DP_FAN / 1000.0
    pump = (mL / 1000.0) * DP_PUMP / 1000.0
    nan = ~feasible
    out = dict(WUE=np.where(nan, np.nan, WUE), fan=np.where(nan, np.nan, fan), pump=pump,
               VAmix5=np.where(nan, np.nan, VAmix5), T5=np.where(nan, np.nan, T5),
               G=np.where(nan, np.nan, G), Me_min=me_min, feasible=feasible)
    return out


def pue_from(fan, pump, chill, T4):
    return 1.0 + fan + pump + np.asarray(chill, float) / cop(T4)
