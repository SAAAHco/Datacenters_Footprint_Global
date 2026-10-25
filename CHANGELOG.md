# Changelog

## v3.0.0 (2026-10-25)

Third production run, the run of record of the manuscript "Water and energy footprint of evaporative data center cooling across the world's climates: a global hourly thermodynamic assessment and a cooling classification" (Energy Conversion and Management: X).
It supersedes v2.0.0 after a pre-submission audit found two conventions of the tower solver,
inherited from the regional study, that a cooling-tower reviewer would reject. Every number,
table and figure of the manuscript comes from this version. Weather record, cleaning policy,
station panel, operating policy, design rule and analysis scripts are those of v2.0.0.

### Physics (dcfootprint/core/tower.py, production default)

- Condenser load: in the chiller mode the tower now rejects the computing load plus the
  compressor work, `Q = Q_DC (1 + chillingYN/COP(T4))`, because the condenser rejects the
  evaporator load together with the compressor work. v2 rejected `Q_DC` in every mode, so
  chiller-mode evaporation and auxiliary power were low by the factor 1 + 1/COP (14% at the
  50/26 loop, 18% at 61/37). The outlet state does not depend on the load, so the correction
  scales WUE and the fan and pump terms of chiller hours exactly.
- Outlet bracket: the bisection on the outlet air temperature now runs between the coldest and
  the warmest surfaces the air meets, `min(T2, T4)` and `T1`, so the outlet air can never leave
  hotter than the water it cooled. v2 bisected between `T2` and 50 C, which placed the outlet
  above the 27 C water of the tower-only loop in 99 to 100% of dry and evaporative hours and
  pinned it at 50 C for the stations designed above 50/26. Where the zero of entropy production
  lies above the warm water (the rule on the 27/18 C loop) the outlet now sits at the warm water
  temperature, saturated when the sprays run, which is the counterflow limit of the Merkel
  method as the tower grows without bound.
- Legacy switch: `DCF_LEGACY_PHYSICS=1` (set by `tests/conftest.py`) or
  `model_vec(..., legacy=True)` restores the v2 conventions, under which the frozen regional
  gates (scalar oracle, Madrid and Texas baselines, reference condition) still reproduce their
  values bit for bit. `tower.PHYSICS` is a one-line description recorded in `manifest.json`.
- The same tower load enters the Merkel closure (`closure_merkel.py`, `chill` argument, used by
  `run/closure_check.py`) and the parametrized model of `run/gsa_world.py`, so the closure
  comparison and the uncertainty propagation are consistent with the production run.

### Verification (tests)

- New gate 9 (`test_gate9_production_physics`): on 30,000 random ambient states at two design
  set points the outlet air never leaves above the warm water and never below
  `min(T2, T4)`; at the reference condition the chiller-mode WUE and auxiliary power under the
  production physics equal the legacy values times exactly 1 + 1/COP(26 C) with the outlet
  temperature unchanged; on the 27/18 C loop the outlet sits at 27 C in both tower-only modes.
  Eight gates in all; `conftest.py` sets the legacy switch so the regional gates are unchanged.

### Analyses and figures (run)

- `run_chain.py`: `world_analysis` now runs alone before the other analyses, because
  `aqueduct_overlay` and `precip_phase` read the annual benchmark it writes (a start-order
  race when the results folder is empty).
- `world_pipeline.py`: records `physics` and package version 0.3.0 in `manifest.json`.
- `make_method_fig.py`: panel (a) shows the tower load and the outlet bracket; panel (c) reads
  the class count and the Sobol sample size from the results; panel (d) lists eight gates.
- `make_ms_figs.py`: class names of this run; the water-price annotation of Fig. 4c is placed
  in axes coordinates.

### Results of record (results/)

World means (2,081 stations, 65,217,135 cleaned station hours): PUE 1.067 (area weighted
1.079), WUE 0.787 L kWh-1 (0.917); dry 43.0%, evaporative 19.1%, chiller 37.9% of hours
(condenser water reset above design in 0.13%); per-mode plateaus PUE 1.017 / 1.010 / 1.151 and
WUE 0 / 1.18 / 1.46 L kWh-1 (1.52 with the reset); no record needed the residual ladder and
none remained infeasible. Against v2.0.0 the PUE values move by less than 0.001 and the WUE
values rise by 10% at the world mean (14% in the chiller-dominated tropics). The class-number
criterion again selects four classes, but the Ward partition moved its boundaries: free-cooling
cold (197 stations, formerly 639), free-cooling temperate (856; named "mixed temperate" in
v2, 755), mixed subtropical (614; named "chiller subtropical" in v2, 275) and chiller tropical
(388). Energy-only control costs 51% more water for 0.0029 of PUE; the water price at which the
water-preferring sequence becomes energy-optimal is 0.0166 kWh L-1; the direct liquid cooling
archetype cuts world water use by 64%. The Merkel comparison at Me = 1.5 moves the world mean
PUE by +0.011 and WUE by +4.1% at the median station, and the evaporative-mode plateau of the
finite tower converges on the bounded closure from above (1.39, 1.28, 1.17 against 1.18 L kWh-1
at Me = 1.5, 2, 3). The 5 K / 8 K sensitivity run (`results/sensitivity_a5_8/`) gives PUE 1.079
and WUE 0.895. The cleaned hourly record of this run is deposited as `world_hourly_v3.parquet`.

### Review round of 25 October 2026 (same version; added before publication)

Three independent reviews of the submission package were run against the built documents and
the pipeline. The fixes that touched the code and the results of record:

- `run/approach_stats.py`: the world row of `approach_stats.csv` double-counted the hours of
  the stations without a class assignment; the hour-weighted world shares are now computed on
  every hour once (44.4 / 19.3 / 36.3% dry / evaporative / chiller).
- `run/classification_hardening.py` writes `station_blocked_labels.csv` (class of every station
  from the spatially blocked tree), and `run/dc_classification.py` adds a leak-free
  leave-one-out (`dc_loo_blocked.csv`): the held-out station is predicted from the pooled
  climate of the class the blocked tree assigns it, not from the class its own clustering
  features earned (median errors 0.012 kWh kWh-1 and 0.093 L kWh-1; membership agrees with the
  reference partition for 89.1% of stations). `dc_loo.csv` (reference membership) is kept for
  comparison. Figure 2c and Table S23 use the leak-free file.
- `run/weighting_check.py` (new): country-capped subsampling of the classification (no country
  above 50 stations, 50 draws) with a same-size random-subsample control
  (`hardening_weighting.csv`, `hardening_weighting_draws.csv`).
- `run/reset_down_check.py` (new): chiller-term sensitivity to a downward condenser water reset
  (`reset_down_sensitivity.csv`, `reset_down_stations.csv`).
- `run/closure_bound_stats.py` (new): share of hours in which the outlet air sits at the
  warm-water cap, per mode and per design, and the chiller-mode plateau under a 5.5 K condenser
  range (`closure_bound_stats.csv`, `closure_bound_stations.csv`, `closure_split_sensitivity.csv`).
- `dcfootprint/core/tower.py`: optional switch `DCF_DRY_PUMP_MPA` (default 0.21, the production
  convention) charging the dry mode a different pump duty; `run/design_optimization.py` reads its
  inputs from `results_world/` and writes to `DCF_OUT` when set. The policy comparison with the
  dry mode charged 0.06 MPa is in `results/sensitivity_drypump_0.06MPa/`.
- `run/make_ms_figs.py`: Figure 2d plots the two normals the planner rule uses (annual mean
  temperature and seasonal range); Figure 2c limits follow the data; legend wording "cooling
  classes"; Figure 3b inset drawn above the axis line. `run/aqueduct_overlay.py`,
  `run/make_si_fig3.py`, `run/make_method_fig.py`: label and layout fixes.
- `hardening_tree_rules.txt` header reads "Cooling class threshold rules".

No production number changed; the eight gates pass.

## v2.0.0 (2026-10-24)

Second production run, accompanying the manuscript "Water and energy footprint of evaporative data center cooling across the world's climates: a global hourly thermodynamic assessment and a cooling classification" (submitted to Energy Conversion and
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
