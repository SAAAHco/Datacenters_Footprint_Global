# Consolidate every result of the global phase into ONE file:
#   results_world/World_Results_Master.xlsx
# Three stages:
#   1. Chain health check: every stage log must exist and end without a traceback.
#   2. Verification: new results against the pre NASA-formula backup
#      (results_world/pre_nasa_2026-08-30), reported on a Verification sheet.
#   3. The workbook: README + Verification + Manifest + one sheet per result table.
# No formulas are written (data consolidation only), Arial throughout, frozen and
# filtered headers.
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "results_world"
BK = RW / "pre_nasa_2026-08-30"
OUT = RW / "World_Results_Master.xlsx"
t0 = time.time()

F_BODY = Font(name="Arial", size=10)
F_HEAD = Font(name="Arial", size=10, bold=True)
F_TITLE = Font(name="Arial", size=13, bold=True)
FILL = PatternFill("solid", fgColor="E2E2E2")

# ---------------------------------------------------------------- 1. chain health
LOGS = ["pipeline_run.log", "analysis_run.log", "koppen_run.log", "dc_classification.log",
        "aqueduct.log", "stress_future.log", "precip_phase.log", "warming.log",
        "hardening.log", "closure_check.log", "gsa_world.log", "representativeness.log",
        "vcap.log", "water_scenarios.log", "design_opt.log", "load_profiles.log",
        "figs.log", "tier_figs.log"]
pipe_mtime = (RW / "pipeline_run.log").stat().st_mtime
health = []
for lg in LOGS:
    p = RW / lg
    if not p.exists():
        health.append((lg, "MISSING"))
        continue
    if lg != "pipeline_run.log" and p.stat().st_mtime < pipe_mtime:
        health.append((lg, "STALE (older than the pipeline run)"))
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    health.append((lg, "TRACEBACK" if "Traceback" in txt else "ok"))
n_bad = sum(1 for _, s in health if s != "ok")
print(f"chain health: {len(health)} logs, {n_bad} problems")
for lg, s in health:
    if s != "ok":
        print("  PROBLEM:", lg, s)

# ---------------------------------------------------------------- 2. verification
ver_rows = [("check", "old (power law)", "new (NASA 1976)", "difference", "verdict")]

def add_check(name, old, new, tol, fmt="{:.4f}"):
    d = new - old
    verdict = "ok" if abs(d) <= tol else "REVIEW"
    ver_rows.append((name, fmt.format(old), fmt.format(new), f"{d:+.4f}", verdict))

sn = pd.read_csv(RW / "station_results.csv")
so = pd.read_csv(BK / "station_results.csv")
sn2 = sn[sn["screen_pass"] == True]      # noqa: E712
so2 = so[so["screen_pass"] == True]      # noqa: E712
add_check("stations screened in", len(so2), len(sn2), 0, "{:.0f}")
add_check("world mean PUE", so2["PUE"].mean(), sn2["PUE"].mean(), 0.001)
add_check("world mean WUE", so2["WUE"].mean(), sn2["WUE"].mean(), 0.003)
m = so2[["STATION", "PUE", "WUE", "p_policy"]].merge(
    sn2[["STATION", "PUE", "WUE", "p_policy"]], on="STATION", suffixes=("_o", "_n"))
add_check("max |station dPUE|", 0.0, float((m["PUE_n"] - m["PUE_o"]).abs().max()), 0.02)
add_check("max |station dWUE|", 0.0, float((m["WUE_n"] - m["WUE_o"]).abs().max()), 0.06)
pol_ch = int((m["p_policy_o"] != m["p_policy_n"]).sum())
ver_rows.append(("stations with pressure policy change", "-", str(pol_ch), "-",
                 "ok" if pol_ch <= 40 else "REVIEW"))
for f, cols, tol in (("koppen_correspondence.csv", ["eta2_koppen"], 0.005),
                     ("dc_k_selection.csv", ["eta2_PUE", "eta2_WUE"], 0.01)):
    o = pd.read_csv(BK / f); n = pd.read_csv(RW / f)
    for c in cols:
        add_check(f"{f}:{c} (first row)", float(o[c].iloc[0]), float(n[c].iloc[0]), tol)
gn = pd.read_csv(RW / "gsa_world_ci.csv"); go = pd.read_csv(BK / "gsa_world_ci.csv")
wn = gn[(gn["scope"] == "world")]; wo = go[(go["scope"] == "world")]
for met in ("PUE", "WUE"):
    add_check(f"GSA world {met} mean",
              float(wo[wo["metric"] == met]["mean"].iloc[0]),
              float(wn[wn["metric"] == met]["mean"].iloc[0]), 0.003)
print(f"verification checks: {len(ver_rows)-1}, "
      f"review flags: {sum(1 for r in ver_rows[1:] if r[4] == 'REVIEW')}")

# ---------------------------------------------------------------- 3. the workbook
SHEETS = [
    # (sheet name <=31 chars, file, description)
    ("station_results", "station_results.csv", "Per station master table: footprint, occupancy, design, pressure policy, screen"),
    ("station_monthly", "station_monthly.csv", "Monthly climatology per station: PUE, WUE, T"),
    ("station_koppen", "station_koppen.csv", "Koppen class per station (World Bank c1976_2000 map)"),
    ("country_results", "country_results.csv", "Country aggregation (FIPS prefix)"),
    ("variance_decomp", "variance_decomp_world.csv", "Nested variance decomposition, world"),
    ("horizon", "horizon_world.csv", "Averaging horizon, median over stations"),
    ("koppen_class_table", "koppen_class_table.csv", "Footprint by Koppen class"),
    ("koppen_correspondence", "koppen_correspondence.csv", "Variance explained by partition (eta2)"),
    ("koppen_loo", "koppen_loo.csv", "Leave one out transfer inside Koppen classes"),
    ("koppen_hellinger", "koppen_hellinger.csv", "Within class climate Hellinger distances"),
    ("dc_class_table", "dc_class_table.csv", "The five DC classes"),
    ("dc_classes", "dc_classes.csv", "Class membership and features per station"),
    ("dc_k_selection", "dc_k_selection.csv", "k selection curve vs the Koppen benchmark"),
    ("dc_vs_koppen_crosstab", "dc_vs_koppen_crosstab.csv", "DC classes against Koppen classes"),
    ("dc_loo", "dc_loo.csv", "Leave one out transfer inside DC classes"),
    ("hardening_bootstrap", "hardening_bootstrap.csv", "Bootstrap stability of the classification"),
    ("hardening_splithalf", "hardening_splithalf.csv", "Out of sample eta2, 50 split halves"),
    ("station_stability", "station_stability.csv", "Per station bootstrap stability"),
    ("annual_benchmark", "annual_benchmark.csv", "100 MW benchmark: annual energy, water, ET0 correlation"),
    ("precip_phase", "station_precip_phase.csv", "Supply side: precipitation vs facility water phase"),
    ("precip_monthly", "station_precip_monthly.csv", "Monthly precipitation aggregates per station"),
    ("water_stress", "station_water_stress.csv", "Aqueduct baseline stress per station"),
    ("stress_summary", "stress_summary.csv", "Panel by stress category"),
    ("stress_dc_crosstab", "stress_dc_crosstab.csv", "DC class by stress category"),
    ("country_stress", "country_stress.csv", "Country exposure to scarce basins"),
    ("stress_peak_on_peak", "stress_peak_on_peak.csv", "Monthly stress vs monthly facility water"),
    ("stress_future", "stress_future.csv", "Per station 2030/2050 stress categories"),
    ("stress_future_summary", "stress_future_summary.csv", "Scarcity under the projections"),
    ("warming_summary", "warming_summary.csv", "Warming scenarios, world summary"),
    ("warming_scenarios", "warming_scenarios.csv", "Warming scenarios per station"),
    ("water_scenarios_summary", "water_scenarios_summary.csv", "Blowdown and source water scenarios"),
    ("water_scenarios_station", "water_scenarios_station.csv", "Water accounting per station"),
    ("ewif_country", "ewif_country.csv", "Grid water intensity per country (OWID mix)"),
    ("design_policy_summary", "design_policy_summary.csv", "Operating policies compared"),
    ("design_policies", "design_policies.csv", "Policies per station"),
    ("design_frontier", "design_frontier.csv", "Water energy frontier (lambda sweep) per class"),
    ("design_t4_sweep", "design_t4_sweep.csv", "Chiller set point sweep per class"),
    ("dlc_archetype", "dlc_archetype.csv", "Direct liquid cooling archetype per station"),
    ("load_profiles", "load_profiles.csv", "Load profile scenarios per station"),
    ("gsa_sobol", "gsa_world_sobol.csv", "Sobol indices under the production configuration"),
    ("gsa_ci", "gsa_world_ci.csv", "Propagated 95 percent intervals"),
    ("closure_check", "closure_check.csv", "Transfer closure check, classes 4 and 5"),
    ("representativeness", "representativeness.csv", "Panel vs composition vs area weighted means"),
    ("screen_bias", "screen_bias_composition.csv", "Class composition, retained vs dropped"),
    ("koppen_area_weights", "koppen_area_weights.csv", "Land area per Koppen class"),
    ("vcap", "vcap_rederivation.csv", "Air flow bound re-derivation"),
]

wb = Workbook()

def style_sheet(ws, ncols, nrows):
    for c in range(1, ncols + 1):
        ws.cell(row=1, column=c).font = F_HEAD
        ws.cell(row=1, column=c).fill = FILL
    ws.freeze_panes = "A2"
    if nrows > 1:
        ws.auto_filter.ref = f"A1:{get_column_letter(ncols)}{nrows}"

def write_df(ws, df):
    ws.append(list(df.columns))
    for row in df.itertuples(index=False, name=None):
        ws.append([("" if (isinstance(v, float) and np.isnan(v)) else v) for v in row])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = F_BODY
    for i, col in enumerate(df.columns, start=1):
        width = min(max(len(str(col)) + 2, 10), 38)
        ws.column_dimensions[get_column_letter(i)].width = width
    style_sheet(ws, len(df.columns), len(df) + 1)

# README
ws = wb.active; ws.title = "README"
ws["A1"] = "World data center footprint: master results workbook"; ws["A1"].font = F_TITLE
meta = [
    ("", ""),
    ("Authors", "Z. Ashkanani, R. Sanchez"),
    ("Built", time.strftime("%Y-%m-%d %H:%M")),
    ("Input", "allData_World_cellSize_1.csv (GHCNh, 99,167,636 records, 3,379 stations, 2022 to 2025)"),
    ("Panel", f"{len(sn2)} screened stations"),
    ("Model", "dcfootprint v2-table6-partialp: corrected auxiliary term, per station pressure, partial pressure entropy, p99.5 design rule, escalation ladder"),
    ("Barometric convention", "U.S. Standard Atmosphere 1976 (NASA.pdf): p = 101325 exp(-9.81 z / (287 (273.16 + T_mean))) Pa, station mean temperature"),
    ("Validation", "all 5 gates pass; verification against the pre change run on the Verification sheet"),
    ("", ""),
    ("Sheet", "Contents"),
]
for r, (a, b) in enumerate(meta, start=2):
    ws.cell(row=r, column=1, value=a).font = F_HEAD if b == "" or a in ("Sheet",) else F_BODY
    ws.cell(row=r, column=2, value=b).font = F_BODY
row0 = 2 + len(meta)
skipped = []
for r, (name, f, desc) in enumerate(SHEETS, start=row0):
    ws.cell(row=r, column=1, value=name).font = F_BODY
    ws.cell(row=r, column=2, value=desc).font = F_BODY
ws.column_dimensions["A"].width = 26; ws.column_dimensions["B"].width = 110

# Verification sheet
ws = wb.create_sheet("Verification")
for row in ver_rows:
    ws.append(list(row))
for row in ws.iter_rows(min_row=2):
    for cell in row:
        cell.font = F_BODY
style_sheet(ws, 5, len(ver_rows))
ws.column_dimensions["A"].width = 42
for c in "BCDE":
    ws.column_dimensions[c].width = 18
ws2 = ws
ws2.append(())
ws2.append(("chain log", "status"))
for lg, s in health:
    ws2.append((lg, s))

# Manifest sheet
ws = wb.create_sheet("Manifest")
man = json.loads((RW / "manifest.json").read_text(encoding="utf-8"))
def flat(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, pre + k + ".")
        else:
            yield (pre + k, str(v))
ws.append(("key", "value"))
for k, v in flat(man):
    ws.append((k, v))
for row in ws.iter_rows():
    for cell in row:
        cell.font = F_BODY
style_sheet(ws, 2, ws.max_row)
ws.column_dimensions["A"].width = 40; ws.column_dimensions["B"].width = 80

# data sheets
for name, f, desc in SHEETS:
    p = RW / f
    if not p.exists():
        skipped.append(name)
        continue
    df = pd.read_csv(p)
    ws = wb.create_sheet(name[:31])
    write_df(ws, df)
    print(f"  sheet {name}: {df.shape}", flush=True)

# tree rules as a text sheet
tr = RW / "hardening_tree_rules.txt"
if tr.exists():
    ws = wb.create_sheet("tree_rules")
    for line in tr.read_text(encoding="utf-8").splitlines():
        ws.append((line,))
    for row in ws.iter_rows():
        row[0].font = Font(name="Consolas", size=10)
    ws.column_dimensions["A"].width = 100

if skipped:
    print("SKIPPED (missing):", skipped)
wb.save(OUT)
print(f"[done] {OUT} ({OUT.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s")
