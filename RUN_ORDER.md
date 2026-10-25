# How to reproduce every result (v3.0.0)

Environment: Python 3.12 or 3.13 with `requirements.txt`. Inputs at the repository root:
`allData_World_cellSize_1.csv` (stage 0 output), `WS_list.xlsx`, `koppen/c1976_2000`
(World Bank Koppen-Geiger map), `aqueduct/` (WRI Aqueduct 4.0 geodatabase),
`gridmix/owid-energy-data.csv`, the GHCNh year folders `2022/` to `2025/` (precipitation
only), and `Old_analysis/` (the regional baselines of gates 1, 3, 4 and 5, available from the
authors; without it those gates are skipped and gates 6 to 9 run self-contained). The test
configuration sets `DCF_LEGACY_PHYSICS=1`; production scripts run the v3 physics by default and
record it in `manifest.json`.

Stage 0 (once): `python wsData.py` then `python dataExtract.py`.

Validation (must pass before anything else is trusted):

    python -m pytest tests -q

Production run (about 60 min) and the sensitivity run (same length, in parallel if wanted):

    python run/world_pipeline.py
    set DCF_A_WET=5 && set DCF_A_DRY=8 && set DCF_OUT=results_world\sens_a5_8 && python run/world_pipeline.py

Downstream chain (groups run in order; the scripts inside a group run concurrently, one log
per stage in `results_world/logs/`):

    python run/run_chain.py classify analyses figures

which is equivalent to running, in this order:

    classify:  assign_koppen -> world_koppen_analysis -> dc_classification
    analyses:  world_analysis first (alone); then closure_check, gsa_world, design_optimization,
               warming_scenarios, aqueduct_overlay, precip_phase (needs precip_extract once),
               classification_hardening, representativeness, vcap_rederive,
               water_scenarios, load_profiles, approach_stats; then stress_monthly_future
    figures:   make_world_figs, make_tier_figs, make_ms_figs, make_method_fig, make_si_fig3;
               then make_si_data, build_master_workbook

Review-round sensitivities (each reads `results_world/` and writes its own CSVs there):

    python run/weighting_check.py
    python run/reset_down_check.py
    python run/closure_bound_stats.py
    set DCF_DRY_PUMP_MPA=0.06 && set DCF_OUT=results_world\sens_drypump && python run/design_optimization.py

Auxiliary, kept for the record: `run/madrid_pressure_variants.py` (the property configuration
calibration under the legacy rules, Supplementary Section S2). Every stage writes to
`results_world/`; the manifest written by `world_pipeline.py` records the model version, the
policy constants and every cleaning constant.
