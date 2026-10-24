# Prepares si_data.json: every table the Supplementary Information embeds, pre-formatted
# as {headers, rows} of strings, read from the verified result CSVs in results_world.
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "results_world"
OUT = RW / "Journal" / "si_data.json"

D = {}

def put(name, df, cols=None, ren=None, nd=3, order=None):
    if cols:
        df = df[cols]
    if ren:
        df = df.rename(columns=ren)
    if order:
        df = df.sort_values(order[0], ascending=order[1])
    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            return f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.3g}"
        return str(v)
    D[name] = dict(headers=list(df.columns), rows=[[fmt(v) for v in r] for r in df.itertuples(index=False, name=None)])

# country results, all 139
c = pd.read_csv(RW / "country_results.csv")
c = c.sort_values("n_stations", ascending=False)
put("country", c, cols=["country", "n_stations", "PUE", "WUE", "occ_free", "occ_evap", "occ_chill"],
    ren={"country": "Code", "n_stations": "Stations", "occ_free": "Free", "occ_evap": "Evap.", "occ_chill": "Chiller"})

# koppen class table
k = pd.read_csv(RW / "koppen_class_table.csv")
put("koppen", k, cols=["koppen", "n_stations", "PUE", "PUE_sd", "WUE", "WUE_sd", "occ_free", "occ_chill"],
    ren={"koppen": "Class", "n_stations": "n", "PUE_sd": "PUE s.d.", "WUE_sd": "WUE s.d.",
         "occ_free": "Free", "occ_chill": "Chiller"})

put("correspondence", pd.read_csv(RW / "koppen_correspondence.csv"), nd=3)
put("variance", pd.read_csv(RW / "variance_decomp_world.csv"), nd=2)
put("horizon", pd.read_csv(RW / "horizon_world.csv"), nd=3)
put("gsa_sobol", pd.read_csv(RW / "gsa_world_sobol.csv"), nd=3)
gc = pd.read_csv(RW / "gsa_world_ci.csv")
put("gsa_ci", gc, cols=["scope", "metric", "production_mean", "production_ci_lo", "production_ci_hi"],
    ren={"production_mean": "mean", "production_ci_lo": "CI lo", "production_ci_hi": "CI hi"}, nd=4)
put("stress_summary", pd.read_csv(RW / "stress_summary.csv"), nd=2)
put("stress_crosstab", pd.read_csv(RW / "stress_dc_crosstab.csv"), nd=0)
cs = pd.read_csv(RW / "country_stress.csv")
put("country_stress", cs.sort_values("scarce_pct", ascending=False), nd=1)
put("future", pd.read_csv(RW / "stress_future_summary.csv"), nd=1)
put("warming", pd.read_csv(RW / "warming_summary.csv"), nd=3)
put("water_scen", pd.read_csv(RW / "water_scenarios_summary.csv"), nd=2)
put("design_policy", pd.read_csv(RW / "design_policy_summary.csv"), nd=4)
put("frontier", pd.read_csv(RW / "design_frontier.csv"), nd=4)
put("t4_sweep", pd.read_csv(RW / "design_t4_sweep.csv"), nd=4)

lp = pd.read_csv(RW / "load_profiles.csv")
lp = lp[lp["dc_class"].isin([1, 2, 3, 4, 5])]
g = lp.groupby("dc_class").mean(numeric_only=True).reset_index()
put("load_profiles", g, cols=["dc_class", "PUE_flat", "WUE_flat", "PUE_inference",
                              "WUE_inference", "PUE_night", "WUE_night"],
    ren={"dc_class": "Class"}, nd=4)

# LOO summaries per partition
def loo_summary(f, label):
    d = pd.read_csv(RW / f)
    grp = "koppen" if "koppen" in d.columns else "dc_class"
    d["absW"] = (d["WUE_loo"] - d["WUE_actual"]).abs()
    s = d.groupby(grp).agg(n=("STATION", "size"),
                           PUE_err_pct=("dPUE_pct", lambda x: float(np.abs(x).median())),
                           WUE_err_abs=("absW", "median")).reset_index()
    s = s.sort_values("n", ascending=False)
    return s
put("loo_koppen", loo_summary("koppen_loo.csv", "koppen"), nd=3)
put("loo_dc", loo_summary("dc_loo.csv", "dc"), nd=3)

put("bootstrap", pd.read_csv(RW / "hardening_bootstrap.csv"), nd=3)
sh = pd.read_csv(RW / "hardening_splithalf.csv")
put("splithalf", pd.DataFrame([dict(
    metric="PUE", dc_mean=sh["eta2_PUE_tree"].mean(), dc_sd=sh["eta2_PUE_tree"].std(),
    koppen_mean=sh["eta2_PUE_koppen"].mean(), koppen_sd=sh["eta2_PUE_koppen"].std()),
    dict(metric="WUE", dc_mean=sh["eta2_WUE_tree"].mean(), dc_sd=sh["eta2_WUE_tree"].std(),
         koppen_mean=sh["eta2_WUE_koppen"].mean(), koppen_sd=sh["eta2_WUE_koppen"].std())]), nd=3)

ew = pd.read_csv(RW / "ewif_country.csv").sort_values("EWIF_incl_hydro", ascending=False)
put("ewif", ew.head(40), cols=["iso_code", "country", "EWIF_incl_hydro", "EWIF_excl_hydro", "gen_year"] if "iso_code" in ew.columns else None, nd=2)

put("screen_bias", pd.read_csv(RW / "screen_bias_composition.csv"), nd=2)
put("area_weights", pd.read_csv(RW / "koppen_area_weights.csv"), nd=2)
put("representativeness", pd.read_csv(RW / "representativeness.csv"), nd=4)
put("vcap", pd.read_csv(RW / "vcap_rederivation.csv"), nd=4)

pp = pd.read_csv(RW / "station_precip_phase.csv")
pp5 = pp[pp["dc_class"].isin([1, 2, 3, 4, 5])]
s = pp5.groupby("dc_class").agg(n=("STATION", "size"), median_r=("r_precip", "median"),
                                in_phase_pct=("r_precip", lambda x: 100 * float((x > 0.3).mean())),
                                anti_phase_pct=("r_precip", lambda x: 100 * float((x < -0.3).mean()))).reset_index()
put("precip_class", s, nd=2)

pk = pd.read_csv(RW / "stress_peak_on_peak.csv")
if "dc_class" in pk.columns and "r_ws" in pk.columns:
    rcol = "r_ws"
elif "r_stress" in pk.columns:
    rcol = "r_stress"
else:
    rcol = [c for c in pk.columns if c.startswith("r")][0]
pk5 = pk[pk["dc_class"].isin([1, 2, 3, 4, 5])] if "dc_class" in pk.columns else pk
s2 = pk5.groupby("dc_class").agg(n=("STATION", "size"), median_r=(rcol, "median")).reset_index() if "dc_class" in pk.columns else None
if s2 is not None:
    put("peak_class", s2, nd=2)

D["tree_rules"] = (RW / "hardening_tree_rules.txt").read_text(encoding="utf-8")

OUT.write_text(json.dumps(D), encoding="utf-8")
print("si_data.json written:", OUT.stat().st_size, "bytes;", len(D), "tables")
