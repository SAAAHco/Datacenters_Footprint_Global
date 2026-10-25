# Global water and energy footprint of evaporative data center cooling

Code and results for the article "Water and energy footprint of evaporative data center cooling across the world's climates: a global hourly thermodynamic assessment and a cooling classification" (Z. Ashkanani, R. Sánchez, R. Mohtar, R. Albatayneh;
submitted to Energy Conversion and Management: X). This repository holds the thermodynamic
model package, the verification gates, one script per analysis, every analysis table and
figure of the study, and the documented execution order, sufficient to reproduce every number
and figure from the raw weather archive. Version 3.0.0 is the run of record of that article;
see `CHANGELOG.md` for what changed from the earlier runs.

## What is here

| Path | Content |
| --- | --- |
| `dcfootprint/` | The model package. `core/properties.py` (REFPROP fits, partial-pressure entropy, wet bulb), `core/tower.py` (mass, energy and entropy balances, bisection closure, auxiliary power, chiller COP; production physics v3: the tower rejects Q_DC(1 + 1/COP) in the chiller mode and the outlet air is bracketed between min(T2, T4) and T1, with the v2 conventions behind `DCF_LEGACY_PHYSICS`), `core/modes.py` (the production operating policy: economizer sequence with approach temperatures, condenser water reset, residual ladder, wet-bulb design rule; the legacy rules kept for the gates), `core/closure_merkel.py` (the Merkel finite-transfer closure used as the independent check), `core/closure_entu.py` (the v1 effectiveness check, unused), `data/ghcnh.py` (cleaning policy, pressure trust ladder, coverage screen). |
| `tests/` | Eight verification gates (pytest). Gates 1, 3, 4 and 5 need the regional baseline inputs (`Old_analysis/`, available from the authors) and skip without them; gates 6 to 9 run self-contained. `conftest.py` sets `DCF_LEGACY_PHYSICS=1` so that the frozen regional gates run under the v2 conventions; gate 9 tests the production physics explicitly. |
| `run/` | One script per analysis, and `run_chain.py`, which runs the downstream stages in groups (`RUN_ORDER.md`). |
| `wsData.py`, `dataExtract.py` | Stage 0: station screening, equal-area grid selection and record extraction from the raw GHCNh archive. |
| `WS_list.xlsx`, `World_cellSize_1.csv` | The 24,072-station inventory and the 3,379-station world selection it produces. |
| `results/` | Every output of the run of record: per-station results and monthly climatologies, the cooling classification, all analysis tables (CSV), the consolidated `World_Results_Master.xlsx`, `manifest.json` (model version, policy constants, cleaning constants), the figures of the article (`figures/`), run logs (`logs/`), and the 5 K / 8 K approach sensitivity run (`sensitivity_a5_8/`). |
| `requirements.txt` | The package versions of the analysis environment. |

## The cleaned hourly record

`world_hourly.parquet` (1.02 GB, deposited as `world_hourly_v3.parquet`; 65,217,135 cleaned station hours with temperature, humidity,
pressure, wet bulb, PUE, WUE, operating mode, condenser water reset and ladder rung per hour)
exceeds the GitHub file limit and is deposited with the Zenodo archive of this repository
(DOI in the article's data availability statement). `run/world_pipeline.py` regenerates it
bit-identically from the raw archive.

## Source data (not redistributed here)

- GHCNh v1.1.0 hourly weather archive: NOAA NCEI (years 2022 to 2025).
- Köppen-Geiger 1976 to 2000 world map: World Bank Data Catalog.
- Aqueduct 4.0 baseline, monthly and future water risk layers: World Resources Institute.
- National generation mixes: Our World in Data energy dataset.

Place them as: GHCNh year folders `2022/` to `2025/` (per-station `.psv` files),
`koppen/c1976_2000` shapefile, `aqueduct/` geodatabase, `gridmix/owid-energy-data.csv`.

## How to reproduce everything

See `RUN_ORDER.md`. In short: stage 0 extraction (`wsData.py`, `dataExtract.py`), then
`python -m pytest tests -q`, then `python run/world_pipeline.py` (about one hour), then
`python run/run_chain.py classify analyses figures`.

## Which file holds which number

| Article item | File |
| --- | --- |
| Table 1 (what is new) | `CHANGELOG.md` |
| Table 2 (the four classes), Fig. 2 | `results/dc_class_table.csv`, `results/dc_classes.csv`, `results/gsa_world_ci.csv` (intervals), `results/dc_k_selection.csv` |
| Table 3 (water accounting) | `results/water_scenarios_summary.csv` |
| Section 3.1 world means, plateaus, occupancy; Fig. 1 | `results/station_results.csv`, `results/representativeness.csv`, `results/approach_stats.csv` |
| Section 3.1 Merkel comparison (Tables S6, S7) | `results/closure_merkel_summary.csv`, `results/closure_merkel.csv` |
| Section 3.2 transfer tests (Tables S15 to S23) | `results/koppen_*.csv`, `results/hardening_*.csv`, `results/dc_loo.csv`, `results/hardening_tree_rules.txt` |
| Section 3.3 benchmark facility, ET0 and rain (Fig. 3a, 3b; Table S25) | `results/annual_benchmark.csv`, `results/station_monthly.csv`, `results/station_precip_phase.csv` |
| Section 3.4 water stress and projections (Fig. 3c, 3d; Tables S26 to S29) | `results/stress_summary.csv`, `results/stress_dc_crosstab.csv`, `results/country_stress.csv`, `results/stress_future_summary.csv`, `results/stress_peak_on_peak.csv` |
| Section 3.4 warming (Fig. 4a, 4b; Table S30) | `results/warming_summary.csv`, `results/warming_scenarios.csv` |
| Section 3.5 policies, lambda*, set point sweep, liquid cooling (Fig. 4c, 4d; Tables S33 to S38) | `results/design_policy_summary.csv`, `results/design_lambda_star.csv`, `results/design_frontier.csv`, `results/design_t4_sweep.csv`, `results/dlc_archetype.csv`, `results/load_profiles.csv` |
| Uncertainty (Tables S8, S9) | `results/gsa_world_sobol.csv`, `results/gsa_world_ci.csv` |
| Sensitivity to the approaches (Table S4) | `results/sensitivity_a5_8/station_results.csv` |

## Conventions

- Properties configuration `v2-table6-partialp`: water vapor entropy at its partial pressure,
  dry air entropy at the dry-air partial pressure; corrected auxiliary power term (250 Pa fan,
  0.21 MPa pump); per-station pressure trust ladder.
- Operating policy (`results/manifest.json`): dry if T2 <= 18 - 5 K; else evaporative if
  Twb <= 18 - 3 K; else chiller at T4 = max(T4_design, ceil(Twb + 3 K)), T1 = T4 + 24 K;
  design rule T4_design = max(26, floor(p99.5(Twb) + 3) + 1) C; residual ladder with the
  0.15 m3 s-1 kW-1 practical air-flow bound.
- Physics (v3): the tower rejects Q_DC in the dry and evaporative modes and Q_DC(1 + 1/COP) in
  the chiller mode (the condenser load); the outlet air temperature is bisected between
  min(T2, T4) and T1. `DCF_LEGACY_PHYSICS=1` restores the v2 conventions (load Q_DC in every
  mode, bracket T2 to 50 C), which the frozen gates were produced under.
- Barometric pressure: U.S. Standard Atmosphere 1976,
  p = 101325 exp(-9.81 z / (287 (273.16 + Tmean_C))) Pa.

## License

Code (`dcfootprint/`, `tests/`, `run/`, `wsData.py`, `dataExtract.py`): MIT License (see
`LICENSE`). Data (`results/`, `WS_list.xlsx`, `World_cellSize_1.csv`): CC BY 4.0, with
attribution to the article this repository accompanies.
