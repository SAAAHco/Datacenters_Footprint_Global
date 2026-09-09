# Five climate classes govern global data center water and energy

Code and data for the article "Five climate classes govern global data center water and energy" (under review). This repository contains the thermodynamic model package, the
validation gates, one script per analysis, and every analysis table and figure of the study,
together with the documented execution order below, sufficient to reproduce every number and
figure from the raw archive.

This repository is anonymized for double-blind peer review. Author information will be added
on acceptance, along with the archived release DOI referenced by the article.

## What is here

| Path | Content |
| --- | --- |
| `dcfootprint/` | The model package. `core/properties.py` (property sets, partial pressure entropy), `core/tower.py` (the evaporative tower model: mass, energy and entropy balances, bisection closure, auxiliary power, COP), `core/modes.py` (operating mode selection and the escalation ladder), `core/closure_entu.py` (the finite-transfer closure check), `data/ghcnh.py` (cleaning policy, pressure trust ladder with the U.S. Standard Atmosphere 1976 barometric form, coverage screen, per-station design rule). |
| `tests/` | The five validation gates (pytest): solver agreement with the scalar oracle, the frozen Madrid and Texas regional baselines, auxiliary-term invariance, and the reference operating condition. |
| `run/` | One script per analysis; execution order below. |
| `wsData.py`, `dataExtract.py` | Stage 0: station screening, equal-area grid selection and record extraction from the raw GHCNh archive. |
| `WS_list.xlsx`, `World_cellSize_1.csv` | The 24,072-station inventory and the 3,379-station world selection it produces. |
| `results/` | Every analysis output of the study: per-station results, the five-class cooling classification, all analysis tables (CSV), the consolidated `World_Results_Master.xlsx`, run logs, the machine-readable `manifest.json` (model version and every cleaning and policy constant), and all analysis figures. |
| `requirements.txt` | The package versions of the analysis environment (Python 3.13). |

## The cleaned hourly record

`world_hourly.parquet` (783 MB; 65,217,135 cleaned station hours with per-hour footprint and
mode) exceeds the GitHub 100 MB file limit and is therefore not in the repository tree. It is
deposited as a release asset of this repository (GitHub releases accept files to 2 GB) or in
the accompanying archive deposit; `run/world_pipeline.py` regenerates it bit-identically from
the raw archive.

## Source data (not redistributed here)

- GHCNh v1.1.0 hourly weather archive: NOAA NCEI (years 2022 to 2025).
- Koppen-Geiger 1976 to 2000 world map: World Bank Data Catalog.
- Aqueduct 4.0 baseline, monthly and future water risk layers: World Resources Institute.
- National generation mixes: Our World in Data energy dataset.

Place them as: GHCNh year folders `2022/` to `2025/` (per-station `.psv` files),
`koppen/c1976_2000` shapefile, `aqueduct/` geodatabase, `gridmix/owid-energy-data.csv`.

## How to reproduce everything

Environment: Python 3.13 with `requirements.txt` (conda or pip).

Stage 0, extraction from the raw archive (run once):

    python wsData.py          # station screening + equal-area selection -> World_cellSize_1.csv
    python dataExtract.py     # record extraction -> allData_World_cellSize_1.csv (~4.4 GB)

Validation (must pass before anything else is trusted):

    python -m pytest tests -q     # 5 gates, ~30 s

The regional baseline inputs of gates 2 and 3 (the Madrid half-hourly series and the Texas
hourly series, with the original scalar oracle) are the subject of a companion regional paper
and are available from the authors on request; the remaining gates run self-contained.

Main chain, in this order (later stages read earlier outputs; every stage writes to
`results/` and appends to its own log):

     1. run/world_pipeline.py            cleaning + footprint run, ~2 h -> station_results,
                                         station_monthly, world_hourly.parquet, manifest.json
     2. run/assign_koppen.py             Koppen class per station
     3. run/world_analysis.py            country tables, variance, horizon, benchmark + ET0
     4. run/world_koppen_analysis.py     correspondence, Hellinger, leave-one-out
     5. run/dc_classification.py         the five cooling classes, planner rule, map
     6. run/aqueduct_overlay.py          baseline water stress overlay
     7. run/stress_monthly_future.py     monthly peak-on-peak + 2030/2050 projections
     8. run/precip_extract.py            precipitation from the raw archive (~12 min, once)
     9. run/precip_phase.py              supply-side phase analysis
    10. run/warming_scenarios.py         +1/+2/+3 C density shifts, two humidity conventions
    11. run/classification_hardening.py  bootstrap, split-half, threshold tree
    12. run/closure_check.py             finite-transfer closure robustness (classes 4 and 5)
    13. run/gsa_world.py                 Sobol indices + propagated intervals (~7 min)
    14. run/representativeness.py        screen bias, composition and area weighting
    15. run/vcap_rederive.py             practical air-flow bound re-derivation
    16. run/water_scenarios.py           blowdown + grid source water accounting
    17. run/design_optimization.py       exact policies, frontier, T4 sweep, DLC archetype
    18. run/load_profiles.py             diurnal load-profile scenarios
    19. run/make_world_figs.py           analysis figures
    20. run/make_tier_figs.py            frontier and warming figures
    21. run/make_ms_figs.py              the four manuscript figures (Fig1 to Fig4)
    22. run/make_si_data.py              the Supplementary Information data tables
    23. run/build_master_workbook.py     cross-checks + World_Results_Master.xlsx

Auxiliary, kept for the record: `run/madrid_pressure_variants.py` (the property-configuration
calibration reported in Supplementary Note 2).

Note for chaining stages under Windows PowerShell 5.1: launch each stage as
`cmd /c "python script > log 2>&1"`; piping a native command's stderr through PowerShell
(`2>&1 | Out-File`) sets `$?` false on mere warnings and silently breaks conditional chains.

## Conventions

- Properties configuration `v2-table6-partialp`: water vapor entropy at its partial pressure,
  dry air entropy at the dry-air partial pressure; corrected auxiliary power term; per-station
  pressure trust ladder; design rule T4 = max(26, floor(p99.5(Td)) + 1) with T1 = T4 + 24;
  escalation ladder with the 0.15 m3 s-1 kW-1 practical air-flow bound.
- Barometric pressure: U.S. Standard Atmosphere 1976,
  p = 101325 exp(-9.81 z / (287 (273.16 + Tmean_C))) Pa.
- `results/manifest.json` records the model version and every cleaning and policy constant of
  the production run.

## License

Code (the `dcfootprint` package, `tests/`, `run/`, `wsData.py`, `dataExtract.py`): MIT
License (see `LICENSE`). Data (`results/`, `WS_list.xlsx`, `World_cellSize_1.csv`):
CC BY 4.0, with attribution to the article this repository accompanies. The copyright
holder reads "The Authors" while the article is under double-anonymized review and will
be completed on acceptance.

## Availability during review

This repository stays private while the article is under double-anonymized review; the
article cites it only as "[repository DOI to be completed]". If reviewer access is needed,
an anonymized mirror (for example anonymous.4open.science) will be provided. On acceptance
the repository is made public, a release is archived with a DOI (for example via the
Zenodo-GitHub integration, together with the `world_hourly.parquet` release asset), and
that DOI replaces the placeholder in the Data and Code availability statements.
