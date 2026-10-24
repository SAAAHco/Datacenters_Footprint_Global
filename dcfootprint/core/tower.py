# Vectorized cooling tower model with the corrected auxiliary power term.
#
# Port of Old_analysis/Texas Analysis/Texas_analysis_package/cooling_tower_vectorized_corr.py
# (itself validated against the original scalar coolingTower class to ~1e-6), generalized in
# two ways only:
#   - the thermodynamic property fits are supplied by a property set object (see properties.py),
#   - the atmospheric pressure p is a per record input instead of the hardcoded 1e5 Pa.
# With props = V1 and p = 1e5 the arithmetic is expression for expression identical to the
# corrected Texas module, so WUE and T5 reproduce it bit for bit (validation gates 1 and 5).
#
# The solver closure is unchanged: the outlet air state sits at zero entropy production,
# found by bisection on T5 between T2 and 50 C. Feasibility: infeasible when T4 < Tdew2
# (condensation) or the entropy balance cannot be satisfied (varE < -1e-6).
import numpy as np

from .properties import V1, RA, RV, K0, EPS_W, pvsat_pa

DP_FAN = 250.0     # Pa, effective fan pressure rise (efficiency folded in)
DP_PUMP = 0.21e6   # Pa, effective loop pressure rise (0.2 MPa spray + 0.01 MPa head)


def cop(T4):
    """Chiller COP correlation, eq. (15)."""
    return 6.88 + 0.18 * 18 - 0.12 * np.asarray(T4, float)


def model_vec(T1, T2, HR2, T3, T4, HR5, evapYN, chillingYN,
              p=1e5, props=V1, n_iter=60, extras=False):
    T1 = np.asarray(T1, float); T2 = np.asarray(T2, float); HR2 = np.asarray(HR2, float)
    T3 = np.asarray(T3, float); T4 = np.asarray(T4, float); HR5 = np.asarray(HR5, float)
    evapYN = np.asarray(evapYN, float); chillingYN = np.asarray(chillingYN, float)
    p = np.broadcast_to(np.asarray(p, float), T2.shape).copy()

    hL1 = props.hL(T1); sL1 = props.sL(T1)
    hV2 = props.hV(T2); hA2 = props.hA(T2)
    pVSat2 = 610.78 * np.exp(17.27 * T2 / (237.3 + T2)); pV2 = HR2 / 100 * pVSat2
    w2 = EPS_W * pV2 / (p - pV2)
    sV2 = props.sV(T2, pV2)
    sA2 = props.sA(T2, p, pV2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Tdew2 = np.where(HR2 == 100, T2, 237.3 / (17.27 / np.log(pV2 / 610.78) - 1))
    hL3 = props.hL(T3); sL3 = props.sL(T3)
    hL4 = props.hL(T4); sL4 = props.sL(T4)
    QDC = 1.0
    mL1 = QDC / (hL1 - hL4); mL4 = mL1

    T5max = np.full_like(T2, 50.0); T5min = T2.copy()

    def eval_T5(T5):
        hV5 = props.hV(T5); hA5 = props.hA(T5)
        pVSat5 = 610.78 * np.exp(17.27 * T5 / (237.3 + T5)); pV5e = HR5 / 100 * pVSat5
        w5e = EPS_W * pV5e / (p - pV5e)
        w5 = np.where(evapYN == 0, w2, w5e)
        pV5 = np.where(evapYN == 0, p * w2 / (EPS_W + w2), pV5e)
        sV5 = props.sV(T5, pV5)
        sA5 = props.sA(T5, p, pV5)
        mA = mL1 * (hL1 - hL4) / ((hA5 + w5 * hV5) - (hA2 + w2 * hV2) - (w5 - w2) * hL3)
        VAmix5 = mA * RA * (K0 + T5) / (p - pV5)
        rho5 = p / RA / (K0 + T5) * (1 - pV5 / p * (1 - RA / RV))
        mMix5 = rho5 * VAmix5
        mV5 = pV5 / RV * VAmix5 / (K0 + T5)
        VAmix2 = mA * RA * (K0 + T2) / (p - pV2)
        mV2 = pV2 / RV * VAmix2 / (K0 + T2)
        mL3 = (w5 - w2) * mA
        WUE = mL3 / QDC * 3600
        # corrected auxiliary power term, P = dp * volumetric flow
        fan = VAmix5 * DP_FAN / 1000.0            # kW per kW_ITE
        pump = (mL1 / 1000.0) * DP_PUMP / 1000.0  # kW per kW_ITE
        PUEnoIT = fan + pump + chillingYN / (6.88 + 0.18 * 18 - 0.12 * T4)
        varE = sL4 * mL4 + (sA5 * mA + sV5 * mV5) - sL1 * mL1 - (sA2 * mA + sV2 * mV2) - sL3 * mL3
        return WUE, PUEnoIT, varE, fan, pump, VAmix5

    T5 = (T5max + T5min) / 2
    for _ in range(n_iter):
        T5 = (T5max + T5min) / 2
        _, _, varE, _, _, _ = eval_T5(T5)
        hot = varE < 0
        T5max = np.where(hot, T5, T5max)
        T5min = np.where(hot, T5min, T5)
    T5 = (T5max + T5min) / 2
    WUE, PUEnoIT, varE, fan, pump, VAmix5 = eval_T5(T5)

    bad = (T4 < Tdew2) | (varE < -1e-6)
    T5 = np.where(bad, np.nan, T5)
    WUE = np.where(bad, np.nan, WUE)
    PUEnoIT = np.where(bad, np.nan, PUEnoIT)
    if extras:
        ex = dict(fan=np.where(bad, np.nan, fan), pump=np.where(bad, np.nan, pump),
                  VAmix5=np.where(bad, np.nan, VAmix5), mL1=mL1)
        return T5, WUE, PUEnoIT, Tdew2, ex
    return T5, WUE, PUEnoIT, Tdew2
