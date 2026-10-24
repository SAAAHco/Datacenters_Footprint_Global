# Finite transfer coefficient closure (effectiveness form), the tropical robustness check
# required by CLAUDE.md 8.2.
#
# The production model closes the tower at the reversible limit (zero entropy production),
# which is ill conditioned near saturation: the entropy production curve is nearly flat at
# its zero, and in near saturated inlet states the reversible outlet collapses onto the
# inlet. This module closes the tower instead with a fixed air side effectiveness, the
# bluntest member of the effectiveness NTU family:
#
#     h5 = h2 + eps (hsat(T1) - h2)
#
# where h denotes the enthalpy of the moist air per unit of dry air, hsat(T1) is the
# enthalpy of saturated air at the entering water temperature (the counterflow limit), the
# outlet is saturated (RH5 = 100, as in the production model), and eps is a fixed tower
# characteristic calibrated once, per operating loop, so that the closure reproduces the
# reversible results in the well conditioned regime (the Madrid series). Away from
# saturation the two closures then agree by construction; in the degenerate near saturated
# states the effectiveness closure keeps the outlet enthalpy bounded away from the inlet, so
# the air flow stays finite where the reversible closure diverges. The difference between
# the two closures at the tropical stations measures how much of the reported footprint
# rests on the reversible idealization.
#
# Free cooling records (no spray) are evaluated with the unchanged reversible model. Sprayed
# records whose effectiveness solution would produce entropy (varE < -1e-6) beyond the
# reversible limit are capped at the reversible solution and counted.
import numpy as np

from .properties import V1, RA, RV, K0, EPS_W, pvsat_pa
from .tower import model_vec, DP_FAN, DP_PUMP
from .modes import select_mode, VCAP


def _wsat(T, p):
    ps = pvsat_pa(T)
    return EPS_W * ps / (p - ps)


def _hmoist_sat(T, p, props):
    """Enthalpy of saturated moist air per unit dry air, kJ per kg dry air."""
    return props.hA(T) + _wsat(T, p) * props.hV(T)


def model_vec_entu(T1, T2, HR2, T3, T4, HR5, evapYN, chillingYN,
                   p=1e5, props=V1, eps_evap=0.7, eps_chill=0.7,
                   n_iter=60, extras=False):
    """Same signature and outputs as tower.model_vec, closed by effectiveness instead of
    zero entropy production for the sprayed records. eps_evap applies to the evaporative
    loop (27/18), eps_chill to the chiller loops."""
    T1 = np.asarray(T1, float); T2 = np.asarray(T2, float); HR2 = np.asarray(HR2, float)
    T3 = np.asarray(T3, float); T4 = np.asarray(T4, float); HR5 = np.asarray(HR5, float)
    evapYN = np.asarray(evapYN, float); chillingYN = np.asarray(chillingYN, float)
    p = np.broadcast_to(np.asarray(p, float), T2.shape).copy()

    # reversible solution everywhere: free cooling records use it directly, sprayed records
    # use it as the physical cap
    T5r, WUEr, PUEr, Tdew2, exr = model_vec(T1, T2, HR2, T3, T4, HR5, evapYN, chillingYN,
                                            p=p, props=props, n_iter=n_iter, extras=True)

    sprayed = evapYN == 1
    eps = np.where(chillingYN == 1, float(eps_chill), float(eps_evap))

    hV2 = props.hV(T2); hA2 = props.hA(T2)
    pV2 = HR2 / 100 * pvsat_pa(T2)
    w2 = EPS_W * pV2 / (p - pV2)
    h2m = hA2 + w2 * hV2
    h1m = _hmoist_sat(T1, p, props)
    target = h2m + eps * (h1m - h2m)

    hL1 = props.hL(T1); hL3 = props.hL(T3); hL4 = props.hL(T4)
    sL1 = props.sL(T1); sL3 = props.sL(T3); sL4 = props.sL(T4)
    sV2 = props.sV(T2, pV2); sA2 = props.sA(T2, p, pV2)
    QDC = 1.0
    mL1 = QDC / (hL1 - hL4); mL4 = mL1

    # bisection on T5 in [Tdew2, T1]: hmoist_sat is increasing in T, the root is unique
    lo = np.where(np.isfinite(Tdew2), Tdew2, T2 - 30.0) + 1e-6
    hi = T1.copy()
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        f = _hmoist_sat(mid, p, props) - target
        high = f > 0
        hi = np.where(high, mid, hi)
        lo = np.where(high, lo, mid)
    T5 = (lo + hi) / 2

    pV5 = pvsat_pa(T5)
    w5 = EPS_W * pV5 / (p - pV5)
    hV5 = props.hV(T5); hA5 = props.hA(T5)
    sV5 = props.sV(T5, pV5); sA5 = props.sA(T5, p, pV5)
    mA = mL1 * (hL1 - hL4) / ((hA5 + w5 * hV5) - (hA2 + w2 * hV2) - (w5 - w2) * hL3)
    VAmix5 = mA * RA * (K0 + T5) / (p - pV5)
    mV5 = pV5 / RV * VAmix5 / (K0 + T5)
    VAmix2 = mA * RA * (K0 + T2) / (p - pV2)
    mV2 = pV2 / RV * VAmix2 / (K0 + T2)
    mL3 = (w5 - w2) * mA
    WUE = mL3 / QDC * 3600
    fan = VAmix5 * DP_FAN / 1000.0
    pump = (mL1 / 1000.0) * DP_PUMP / 1000.0
    PUEnoIT = fan + pump + chillingYN / (6.88 + 0.18 * 18 - 0.12 * T4)
    varE = (sL4 * mL4 + (sA5 * mA + sV5 * mV5) - sL1 * mL1
            - (sA2 * mA + sV2 * mV2) - sL3 * mL3)

    # cap: effectiveness solutions beyond the reversible limit are unphysical
    capped = sprayed & (varE < -1e-6)
    # condensation infeasibility is unchanged
    bad = T4 < Tdew2

    T5 = np.where(capped, T5r, T5)
    WUE = np.where(capped, WUEr, WUE)
    PUEnoIT = np.where(capped, PUEr, PUEnoIT)
    VAmix5 = np.where(capped, exr["VAmix5"], VAmix5)
    fan = np.where(capped, exr["fan"], fan)

    # free cooling records: reversible model throughout
    T5 = np.where(sprayed, T5, T5r)
    WUE = np.where(sprayed, WUE, WUEr)
    PUEnoIT = np.where(sprayed, PUEnoIT, PUEr)
    VAmix5 = np.where(sprayed, VAmix5, exr["VAmix5"])
    fan = np.where(sprayed, fan, exr["fan"])

    T5 = np.where(bad, np.nan, T5)
    WUE = np.where(bad, np.nan, WUE)
    PUEnoIT = np.where(bad, np.nan, PUEnoIT)
    if extras:
        ex = dict(fan=fan, pump=np.where(bad, np.nan, pump), VAmix5=VAmix5, mL1=mL1,
                  capped=capped, eps=eps)
        return T5, WUE, PUEnoIT, Tdew2, ex
    return T5, WUE, PUEnoIT, Tdew2


def run_with_escalation_entu(T2, HR2, T4base=26.0, p=1e5, props=V1,
                             eps_evap=0.7, eps_chill=0.7, max_rungs=10):
    """The escalation ladder of modes.run_with_escalation, driven by the effectiveness
    closure. Identical triggers: infeasible records and air flow above VCAP."""
    T2 = np.asarray(T2, float); HR2 = np.asarray(HR2, float)
    p = np.broadcast_to(np.asarray(p, float), T2.shape).copy()
    T1, T4, evap, chill, mode = select_mode(T2, HR2, T4base)
    T5, WUE, PUEnoIT, Tdew2, ex = model_vec_entu(
        T1, T2, HR2, T2, T4, np.full_like(T2, 100.0), evap, chill,
        p=p, props=props, eps_evap=eps_evap, eps_chill=eps_chill, extras=True)
    esc = np.zeros(T2.shape, int)
    n_capped = int(ex["capped"].sum())
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
        t5, w, pn, td, e2 = model_vec_entu(
            T1[idx], T2[idx], HR2[idx], T2[idx], T4[idx], np.full(len(idx), 100.0),
            evap[idx], chill[idx], p=p[idx], props=props,
            eps_evap=eps_evap, eps_chill=eps_chill, extras=True)
        T5[idx] = t5; WUE[idx] = w; PUEnoIT[idx] = pn
        for k in ("fan", "pump", "VAmix5"):
            ex[k][idx] = e2[k]
        n_capped = int(n_capped + e2["capped"].sum())
    return dict(T1=T1, T4=T4, evap=evap, chill=chill, mode=mode, esc=esc,
                T5=T5, WUE=WUE, PUE=PUEnoIT + 1.0, Tdew2=Tdew2,
                VAmix5=ex["VAmix5"], n_capped=n_capped)
