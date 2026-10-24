# Changelog

## v2.0.0 (2026-10-24)

Second production run, accompanying the manuscript "The world's climates sort evaporative data
center cooling into four water and energy regimes" (submitted to Energy Conversion and
Management: X). Every number, table and figure of that manuscript comes from this version.
The weather record, the cleaning policy and the station panel are unchanged from v1.0.0
(2,081 stations, 65,217,135 cleaned station hours, 2022 to 2025).

### Model and operating policy (dcfootprint/core)

- `properties.py`: new `twb_c()`, the thermodynamic wet bulb (adiabatic saturation temperature)
  computed with the model's own enthalpy fits at the station pressure.
- `modes.py`: new production policy `select_mode_economizer()` / `run_policy()`, the
  economizer sequence with explicit approach temperatures: dry free cooling on the 27/18 C loop
  when T2 <= 18 - A_dry (A_dry = 5 K); evaporative cooling when Twb <= 18 - A_wet
  (A_wet = 3 K); otherwise the chiller with the condenser water reset hour by hour to
  T4 = max(T4_design, ceil(Twb + A_wet)), T1 = T4 + 24. The fixed +5 C escalation above
  37.5 C ambient is gone; the residual ladder (infeasible record or air flow above
  0.15 m3 s-1 kW-1) is kept. The v1 rules (`select_mode()`, `run_with_escalation()`) remain in
  the module for the verification gates only. Reason for the change: the v1 feasibility test
  bounded the cold water by the dew point and allowed sprayed operation with cold water below
  the inlet wet bulb (3.9% of evaporative hours in v1), and the v1 dry-cooling curve gave
  negative humidity limits above 14.4 C.
- New design rule (`design_t4base_twb()`): T4_design = max(26, floor(p99.5(Twb) + A_wet) + 1)
  instead of the dew point rule. 1,086 stations keep 50/26 C; the most humid reach 61/37 C.
- `closure_merkel.py` (new): the Merkel finite transfer closure (Lewis factor 1, saturated
  outlet), solved for the air flow a tower of prescribed Merkel number needs to hold the
  loop. Replaces the effectiveness closure of v1 (`closure_entu.py`, kept for the record but
  no longer used), which was calibrated on the reversible model itself.
- The approach temperatures can be overridden for sensitivity runs through the environment
  (`DCF_A_WET`, `DCF_A_DRY`); `world_pipeline.py` also accepts `DCF_SRC`, `DCF_OUT` and
  `DCF_SMOKE`.

### Verification (tests)

- Two new gates: wet bulb against psychrometric chart values and bounds (gate 7), and the
  approach rule on 30,000 random ambient states at two design set points (gate 8). The five
  v1 gates run unchanged under the legacy fits and rules.

### Analyses (run)

- `world_pipeline.py`: writes `Twb` and `reset` per hour to `world_hourly.parquet`; the
  per-station columns `occ_chill_esc`, `n_escalated`, `max_rung` and `PUE_chillesc` /
  `WUE_chillesc` now describe chiller hours with the condenser water reset above design;
  new columns `Twb_p995`, `Twb_mean`, `Twb_max`, `reset_mean_K`, `n_ladder`. The legacy
  comparison columns `PUE_v1` / `WUE_v1` are dropped.
- `closure_check.py`: rewritten as the Merkel comparison at every station and sprayed mode for
  Me = 0.75, 1.0, 1.5, 2.0 and 3.0 (`closure_merkel.csv`, `closure_merkel_summary.csv`).
- `classification_hardening.py`: leakage-free tests. The split-half test assigns the held-out
  half by a threshold tree on climate normals (v1 used the held-out stations' own footprint
  features); new spatially blocked cross-validation (15 degree blocks), confusion matrix,
  per-class precision and recall, and the footprint error of assigning class means through
  the rule (`hardening_blocked_cv.csv`, `hardening_tree_confusion.csv`,
  `hardening_tree_classwise.csv`).
- `dc_classification.py`: the Koppen-Geiger benchmark is read from the current run instead of
  hard-coded; the class-number criterion now selects four classes.
- `design_optimization.py`: candidate maps under the new policy; the water price lambda* at
  which the frontier returns to the production water use is solved by bisection
  (`design_lambda_star.csv`); the v1 "eq. 23" policy is dropped; the direct liquid cooling
  archetype follows the same approach rules (dry to 25 C ambient, sprayed to a 27 C wet bulb).
- `gsa_world.py`: eleven parameters (the two approach temperatures added, +-1 K and +-2 K),
  mode selection per draw, 13,312 Saltelli evaluations.
- `warming_scenarios.py`: design erosion counted against the wet bulb design rule; four
  operating states.
- `approach_stats.py` (new): approach temperature distributions, reset statistics and the
  share of hours within 1 K of each switching rule (`approach_stats.csv`, `approach_hist.csv`).
- `run_chain.py` (new): runs the downstream stages in groups with one log per stage.
- Figure scripts updated for four classes and the four operating states; `make_method_fig.py`
  reads its numbers from the result files.

### Results of record (results/)

World means (2,081 stations): PUE 1.066 (area weighted 1.078), WUE 0.708 L kWh-1 (0.818);
dry 43.0%, evaporative 19.1%, chiller 38.0% of hours (reset above design in 0.13%). Four
classes: free-cooling cold (639 stations), mixed temperate (755), chiller subtropical (275),
chiller tropical (386). The 5 K / 8 K sensitivity run is in `results/sensitivity_a5_8/`.

## v1.0.0 (2026-08-31)

First production run (v1 rules: minimum-PUE mode boundaries, dew point design rule,
effectiveness closure check). Superseded; the station means of that run are kept with the
project for traceability but are not part of this release.
