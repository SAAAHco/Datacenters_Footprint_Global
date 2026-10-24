# Thermodynamic properties and psychrometrics.
#
# Two property sets:
#   V1 (legacy): the REFPROP fits hardcoded in Old_analysis/cooling_tower_vectorized.py,
#       polynomials in T(K), valid at p = 0.1 MPa only. Used by the validation gates so the
#       Madrid and Texas baselines are reproduced bit for bit.
#   V2 (table6): the fits of prelim_ideas.docx Table 6, polynomials in T(degC) with explicit
#       pressure terms (p in MPa) on the dry air entropy, fitted over 0.05 to 0.12 MPa and
#       276 to 372 K. Same datum as V1 (checked: h_a, s_a, s_L, s_v agree with V1 at 25 C and
#       0.1 MPa to the fit precision). The vapour entropy fit carries no pressure terms; it
#       reproduces the saturation line s_v(T, psat(T)) (checked at 25 C and 50 C), so the
#       optional partial pressure variant applies s_v(T, pV) = s_v_sat(T) - Rv ln(pV/psat(T)).
import numpy as np

# Gas constants, project convention (J/kg/K)
RA = 8.31 / 28.97e-3
RV = 8.31 / 18.05e-3
K0 = 273.16          # K offset used throughout the original model
EPS_W = 0.62186      # humidity ratio coefficient


def pvsat_pa(T):
    """Magnus saturation vapour pressure, T in degC, result in Pa."""
    return 610.78 * np.exp(17.27 * T / (237.3 + T))


def tdew_c(pV):
    """Dew point in degC from vapour pressure in Pa (inverse Magnus)."""
    return 237.3 / (17.27 / np.log(pV / 610.78) - 1.0)


class V1Properties:
    """Legacy fits at 0.1 MPa, T in K. Pressure arguments are accepted and ignored."""
    name = "v1-legacy-0.1MPa"
    eL = (-1143.79850778, 4.18765442)
    eV = (1.89674720e+03, 2.54423376e+00, -1.22199599e-03)
    eA = (124.0129842, 1.00763588)
    sLc = (-5.59476137e+00, 2.59636662e-02, -2.00282656e-05)
    sVc = (2.13062766e+01, -6.41403034e-02, 7.17739861e-05)
    sAc = (2.45260622e+00, 6.22899746e-03, -4.78974682e-06)

    def hL(self, T):
        return self.eL[0] + self.eL[1] * (T + K0)

    def hV(self, T):
        Tk = T + K0
        return self.eV[0] + self.eV[1] * Tk + self.eV[2] * Tk**2

    def hA(self, T):
        return self.eA[0] + self.eA[1] * (T + K0)

    def sL(self, T):
        Tk = T + K0
        return self.sLc[0] + self.sLc[1] * Tk + self.sLc[2] * Tk**2

    def sV(self, T, pV=None):
        Tk = T + K0
        return self.sVc[0] + self.sVc[1] * Tk + self.sVc[2] * Tk**2

    def sA(self, T, p=None, pV=None):
        Tk = T + K0
        return self.sAc[0] + self.sAc[1] * Tk + self.sAc[2] * Tk**2


class V2Properties:
    """Table 6 fits, T in degC, p in Pa (converted to MPa internally for the s_a terms).

    vapour_partial_pressure: if True, the vapour entropy is shifted from the saturation line
    to the actual partial pressure, and the dry air entropy is evaluated at the dry air
    partial pressure p - pV instead of the total pressure. This is the explicit test the
    project brief requires (CLAUDE.md 8.1, point 3); default False matches the paper.
    """
    name = "v2-table6"

    def __init__(self, vapour_partial_pressure=False):
        self.vpp = bool(vapour_partial_pressure)
        if self.vpp:
            self.name = "v2-table6-partialp"

    def hL(self, T):
        return 4.1887 * T

    def hV(self, T):
        return 2500.4620 + 1.8784 * T + 1.2279e-3 * T**2

    def hA(self, T):
        return 399.2857 + 1.0074 * T

    def sL(self, T):
        return 1.5153e-2 * T - 2.1274e-5 * T**2

    def sV(self, T, pV=None):
        s = 9.1540 - 2.6506e-2 * T + 1.1160e-4 * T**2 - 2.6642e-7 * T**3
        if self.vpp and pV is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                s = s - (RV / 1000.0) * np.log(pV / pvsat_pa(T))
        return s

    def sA(self, T, p, pV=None):
        if self.vpp and pV is not None:
            pa = np.asarray(p, float) - pV
        else:
            pa = np.asarray(p, float)
        pM = pa / 1e6
        return (4.3983 + 3.6069e-3 * T - 4.7568e-6 * T**2
                - 1.0982e1 * pM + 6.7642e1 * pM**2 - 1.7960e2 * pM**3)


V1 = V1Properties()
V2 = V2Properties()
V2_PARTIALP = V2Properties(vapour_partial_pressure=True)


def twb_c(T, RH, p, props=V2_PARTIALP, n_iter=40):
    """Thermodynamic wet bulb (adiabatic saturation temperature) in degC.

    Solves, with the same Magnus saturation law and the same enthalpy fits as the tower
    model, the adiabatic saturation balance
        hA(Twb) + ws(Twb) hV(Twb) - (ws(Twb) - w) hL(Twb) = hA(T) + w hV(T)
    by bisection between the dew point (or T - 60 when the air is dry enough that the dew
    point is undefined) and the dry bulb T. The left side rises monotonically with Twb, so
    the root is unique. T in degC, RH in percent, p in Pa; arrays broadcast.
    """
    T = np.asarray(T, float); RH = np.asarray(RH, float)
    p = np.broadcast_to(np.asarray(p, float), T.shape)
    pV = RH / 100.0 * pvsat_pa(T)
    w = EPS_W * pV / (p - pV)
    h_in = props.hA(T) + w * props.hV(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        tdew = np.where(RH >= 100.0, T, tdew_c(pV))
    lo = np.where(np.isfinite(tdew), np.minimum(tdew, T), T - 60.0)
    hi = T.copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        ps = pvsat_pa(mid)
        ws = EPS_W * ps / (p - ps)
        f = props.hA(mid) + ws * props.hV(mid) - (ws - w) * props.hL(mid) - h_in
        high = f > 0
        hi = np.where(high, mid, hi)
        lo = np.where(high, lo, mid)
    return 0.5 * (lo + hi)
