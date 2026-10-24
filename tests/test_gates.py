# Validation gates (CLAUDE.md section 9). All must pass before any global number leaves
# the repo. Gates 3 and 4 read the original raw series and take a couple of minutes; run
# with:  python -m pytest tests -s
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dcfootprint.core.properties import V1, V2_PARTIALP, twb_c, pvsat_pa, tdew_c
from dcfootprint.core.tower import model_vec
from dcfootprint.core.modes import (select_mode, run_with_escalation, run_policy,
                                    design_t4base_twb, A_WET, A_DRY, T_SUPPLY)

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "Old_analysis"
# The regional baselines (scalar oracle, legacy module, Madrid and Texas series) belong to the
# companion regional study and are not redistributed; their gates are skipped when absent.
_needs_old = pytest.mark.skipif(not (OLD / "cooling_tower_tradeoff.py").exists(),
                                reason="regional baseline inputs (Old_analysis/) not present; available from the authors")
_needs_madrid = pytest.mark.skipif(not (OLD / "todosLosDatos.txt").exists(),
                                   reason="Madrid half-hourly series not present")
_needs_texas = pytest.mark.skipif(not (OLD / "Texas Analysis" / "Texas_abreviatedData.txt").exists(),
                                  reason="Texas hourly series not present")


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- gate 1: scalar oracle
@_needs_old
def test_gate1_vectorized_vs_scalar_oracle():
    oracle = _load_module(OLD / "cooling_tower_tradeoff.py", "ct_oracle")
    rng = np.random.default_rng(0)
    n = 2000
    Ts = rng.uniform(-5, 45, n); Hs = rng.uniform(5, 100, n)
    T1, T4, ev, ch, md = select_mode(Ts, Hs, T4base=26.0)
    T5v, WUEv, PUEv, Tdewv = model_vec(T1, Ts, Hs, Ts, T4, np.full(n, 100.0), ev, ch,
                                       p=1e5, props=V1)
    assert all((md == k).sum() > 0 for k in range(4)), "sample must span all four modes"

    n_mismatch = 0
    dP, dW, dT = [], [], []
    for i in range(n):
        c = oracle.coolingTower(T1[i], Ts[i], Hs[i], Ts[i], T4[i], 100, ev[i], ch[i])
        c.results()
        for arr, scal, store in ((PUEv[i], c.PUEnoIT, dP), (WUEv[i], c.WUE, dW), (T5v[i], c.T5, dT)):
            a_nan = bool(np.isnan(arr)); s_nan = bool(np.isnan(scal))
            if a_nan and s_nan:
                continue
            if a_nan != s_nan:
                n_mismatch += 1
                continue
            store.append(abs(arr - scal))
    assert n_mismatch == 0, f"{n_mismatch} feasibility mismatches vs oracle"
    # the oracle carries the OLD aux term, so PUE is compared on WUE/T5 only; the corrected
    # PUE is validated by gates 4 and 6. WUE and T5 depend only on the entropy solver.
    # Tolerance: the original validation harness itself recorded max |dWUE| = 6.5e-6
    # (Old_analysis/RESULTS.md section 0), so 1e-5 is the faithful bound.
    assert max(dW) < 1e-5, f"max |dWUE| = {max(dW):.3e}"
    assert max(dT) < 1e-3, f"max |dT5| = {max(dT):.3e}"


# ------------------------------------------------------- gate 5: WUE and T5 invariance
@_needs_old
def test_gate5_wue_t5_bit_identical_to_old_aux_module():
    old = _load_module(OLD / "cooling_tower_vectorized.py", "ct_old_vec")
    rng = np.random.default_rng(1)
    n = 5000
    Ts = rng.uniform(-10, 47, n); Hs = rng.uniform(3, 100, n)
    T1, T4, ev, ch, md = select_mode(Ts, Hs, T4base=26.0)
    HR5 = np.full(n, 100.0)
    T5n, WUEn, _, Tdn = model_vec(T1, Ts, Hs, Ts, T4, HR5, ev, ch, p=1e5, props=V1)
    T5o, WUEo, _, Tdo = old.model_vec(T1, Ts, Hs, Ts, T4, HR5, ev, ch)
    assert np.array_equal(T5n, T5o, equal_nan=True), "T5 not bit identical to old module"
    assert np.array_equal(WUEn, WUEo, equal_nan=True), "WUE not bit identical to old module"
    assert np.array_equal(Tdn, Tdo, equal_nan=True)


# ---------------------------------------------------------- gate 6: reference condition
def test_gate6_reference_condition():
    T5, WUE, PUEnoIT, Tdew, ex = model_vec(
        np.array([50.0]), np.array([25.0]), np.array([30.0]), np.array([25.0]),
        np.array([26.0]), np.array([100.0]), np.array([1.0]), np.array([1.0]),
        p=1e5, props=V1, extras=True)
    aux = float(ex["fan"][0] + ex["pump"][0])
    assert abs(T5[0] - 36.8) < 0.05, f"T5 = {T5[0]:.3f}"
    assert abs(WUE[0] - 1.28) < 0.005, f"WUE = {WUE[0]:.4f}"
    # exact corrected value is 0.00449 kWh/kWh (fan 0.00239 + pump 0.00210); the 0.005 in
    # CLAUDE.md section 9 is a one figure rounding of it
    assert abs(aux - 0.0045) < 0.0005, f"aux = {aux:.5f}"
    assert abs(ex["pump"][0] - 0.0021) < 0.0002, f"pump = {ex['pump'][0]:.5f}"


# ------------------------------------------------- gate 7: wet bulb against the chart
def test_gate7_wet_bulb_matches_psychrometric_chart():
    # thermodynamic wet bulb at sea level, reference values from the ASHRAE chart
    pts = [(25.0, 30.0, 14.4), (35.0, 50.0, 26.1), (10.0, 50.0, 5.5), (40.0, 10.0, 18.6),
           (0.0, 80.0, -1.1), (30.0, 100.0, 30.0)]
    for T, RH, ref in pts:
        tw = float(twb_c(T, RH, 101325.0))
        assert abs(tw - ref) < 0.3, f"Twb({T}, {RH}) = {tw:.2f}, chart {ref}"
    rng = np.random.default_rng(3)
    T = rng.uniform(-30, 48, 20000); RH = rng.uniform(1, 100, 20000)
    p = rng.uniform(60000, 105000, 20000)
    tw = twb_c(T, RH, p)
    pV = RH / 100 * pvsat_pa(T)
    td = np.where(RH >= 100, T, tdew_c(pV))
    assert np.all(tw <= T + 1e-6) and np.all(tw >= td - 1e-6), "wet bulb outside [Tdew, T]"


# ---------------------------------------- gate 8: production policy respects the approach
def test_gate8_policy_holds_the_approach_everywhere():
    rng = np.random.default_rng(4)
    n = 30000
    T = rng.uniform(-30, 48, n); RH = rng.uniform(1, 100, n); p = rng.uniform(60000, 105000, n)
    for t4b in (26.0, 30.0):
        r = run_policy(T, RH, T4base=t4b, p=p, props=V2_PARTIALP, max_rungs=10)
        ok = ~np.isnan(r["PUE"]) & ~np.isnan(r["WUE"])
        assert ok.mean() > 0.999, f"{int((~ok).sum())} records infeasible after the ladder"
        sprayed = r["evap"] == 1
        assert np.all(r["T4"][sprayed] >= r["twb"][sprayed] + A_WET - 1e-9), \
            "sprayed record with cold water below wet bulb plus approach"
        dry = r["mode"] == 0
        assert np.all(T[dry] <= T_SUPPLY - A_DRY + 1e-9), "dry record above the dry bulb limit"
        assert np.all(r["T4"][dry] == T_SUPPLY) and np.all(r["evap"][dry] == 0)
        chill = r["mode"] == 2
        assert np.all(r["T4"][chill] >= t4b) and np.all(r["T1"][chill] == r["T4"][chill] + 24)
        assert np.all(r["reset"][chill] == r["T4"][chill] - t4b)
        assert all((r["mode"] == k).sum() > 0 for k in range(3))
    # design rule: one degree above the approach requirement at the 99.5th percentile
    tw = np.array([10.0, 20.0, 22.4, 24.9])
    assert design_t4base_twb(tw, a_wet=3.0, pct=100.0)[0] == 28.0
    assert design_t4base_twb(np.array([5.0, 12.0]), a_wet=3.0, pct=100.0)[0] == 26.0


# ------------------------------------------------------------- gate 3: Madrid baseline
@pytest.mark.slow
@_needs_madrid
def test_gate3_madrid_baseline():
    cols = ["IdEstacion", "Fecha", "Anio", "Dia", "HoraMin", "T", "HR"]
    df = pd.read_csv(OLD / "todosLosDatos.txt", sep="\t", usecols=[1, 2, 3, 4, 5, 6, 7],
                     names=cols, header=0)
    df = df.dropna(subset=["T", "HR"]).reset_index(drop=True)
    n0 = len(df)
    assert n0 > 1_700_000

    T1, T4, ev, ch, md = select_mode(df["T"].values, df["HR"].values, T4base=26.0)
    T5, WUE, PUEnoIT, Tdew2 = model_vec(T1, df["T"].values, df["HR"].values, df["T"].values,
                                        T4, np.full(n0, 100.0), ev, ch, p=1e5, props=V1)
    PUE = PUEnoIT + 1.0
    n_inf = int(np.isnan(PUE).sum())
    ok = ~np.isnan(PUE) & ~np.isnan(WUE)
    pue_m = float(PUE[ok].mean()); wue_m = float(WUE[ok].mean())
    print(f"\n[gate3] Madrid rows {n0:,}  infeasible {n_inf}  "
          f"PUE {pue_m:.4f}  WUE {wue_m:.4f}")
    assert n_inf < 10, f"{n_inf} infeasible records"
    assert abs(pue_m - 1.039) < 0.002, f"Madrid mean PUE {pue_m:.4f} (target about 1.039)"
    assert abs(wue_m - 1.014) < 0.002, f"Madrid mean WUE {wue_m:.4f} (target 1.014)"


# -------------------------------------------------------------- gate 4: Texas baseline
@pytest.mark.slow
@_needs_texas
def test_gate4_texas_baseline():
    path = OLD / "Texas Analysis" / "Texas_abreviatedData.txt"
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    stations = sorted(df["Station"].unique())
    t4base = {"El Paso": 26.0, "ElPaso": 26.0, "Lubbock": 26.0,
              "Dallas": 28.0, "Houston": 28.0}
    targets = {  # (PUE, WUE) frozen in CLAUDE.md section 6.1
        "El Paso": (1.056, 1.05), "ElPaso": (1.056, 1.05), "Lubbock": (1.058, 0.99),
        "Dallas": (1.087, 1.08), "Houston": (1.112, 1.15)}

    tot_esc = 0
    pooled_pue = []; pooled_wue = []
    print()
    for st in stations:
        g = df[df["Station"] == st]
        T2 = g["Temperature"].values.astype(float)
        HR2 = g["Relative Humidity"].values.astype(float)
        r = run_with_escalation(T2, HR2, T4base=t4base[st], p=1e5, props=V1)
        ok = ~np.isnan(r["PUE"]) & ~np.isnan(r["WUE"])
        assert ok.all(), f"{st}: {int((~ok).sum())} records still infeasible after the ladder"
        pue_m = float(r["PUE"].mean()); wue_m = float(r["WUE"].mean())
        n_esc = int((r["esc"] > 0).sum()); tot_esc += n_esc
        pooled_pue.append(r["PUE"]); pooled_wue.append(r["WUE"])
        tp, tw = targets[st]
        print(f"[gate4] {st:8s} n {len(g):7,}  PUE {pue_m:.4f} (target {tp})  "
              f"WUE {wue_m:.4f} (target {tw})  escalated {n_esc}")
        assert abs(pue_m - tp) < 0.005, f"{st} PUE {pue_m:.4f} vs {tp}"
        assert abs(wue_m - tw) < 0.006, f"{st} WUE {wue_m:.4f} vs {tw}"
    pue_p = float(np.concatenate(pooled_pue).mean())
    wue_p = float(np.concatenate(pooled_wue).mean())
    print(f"[gate4] pooled PUE {pue_p:.4f} (target 1.078)  WUE {wue_p:.4f} (target 1.067)  "
          f"escalated {tot_esc} (target 601)")
    assert abs(pue_p - 1.078) < 0.002
    assert abs(wue_p - 1.067) < 0.002
    assert tot_esc == 601, f"escalated {tot_esc} != 601"
