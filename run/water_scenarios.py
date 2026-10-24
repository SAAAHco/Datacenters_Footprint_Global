# Tier 3 item 9: blowdown and source water as labeled scenarios on top of the base case
# (which counts on site evaporation only, per the manuscript's consumptive framing).
#
# Blowdown: tower purge controlling cycles of concentration CoC. Makeup withdrawal equals
# evaporation times CoC/(CoC - 1); the blowdown itself returns to the source, so CONSUMPTION
# stays the evaporation while WITHDRAWAL carries the factor. Scenarios CoC = 3, 5, 10.
#
# Source water: the water embedded in the electricity the facility buys, WUE_source =
# PUE x EWIF of the national grid, where EWIF is the generation weighted operational water
# consumption intensity of the country mix (latest year >= 2022 in the OWID energy dataset),
# with technology factors (L per kWh net, literature medians after Macknick et al. 2012):
# coal 1.9, gas 0.75, oil 1.5, nuclear 2.3, biofuel 2.2, hydro 4.5 (reservoir evaporation,
# the dominant and most contested term, so an excluding variant is reported), wind 0.004,
# solar 0.1, other renewables (geothermal dominated) 1.5.
# Outputs: results_world/water_scenarios_station.csv, water_scenarios_summary.csv, ewif_country.csv.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_world"

FACTORS = dict(coal=1.9, gas=0.75, oil=1.5, nuclear=2.3, biofuel=2.2,
               hydro=4.5, wind=0.004, solar=0.1, other_renewable=1.5)

# FIPS 10-4 (GHCNh station prefix) to ISO3 for the panel countries. Prefixes not listed
# fall back to the world average EWIF and are counted.
FIPS_ISO3 = {
 "US":"USA","CA":"CAN","MX":"MEX","GT":"GTM","HO":"HND","ES":"SLV","NU":"NIC","CS":"CRI",
 "PM":"PAN","CU":"CUB","DR":"DOM","HA":"HTI","JM":"JAM","TD":"TTO","BF":"BHS","BB":"BRB",
 "BR":"BRA","AR":"ARG","CI":"CHL","BL":"BOL","PE":"PER","EC":"ECU","CO":"COL","VE":"VEN",
 "PA":"PRY","UY":"URY","GY":"GUY","NS":"SUR","BH":"BLZ",
 "FR":"FRA","PO":"PRT","SP":"ESP","BE":"BEL","NL":"NLD","LU":"LUX","EI":"IRL","IC":"ISL",
 "NO":"NOR","DA":"DNK","FI":"FIN","SW":"SWE","PL":"POL","EZ":"CZE","LO":"SVK","HU":"HUN",
 "RO":"ROU","BU":"BGR","GR":"GRC","AL":"ALB","MK":"MKD","SR":"SRB","HR":"HRV","SI":"SVN",
 "BK":"BIH","MJ":"MNE","KV":"XKX","UK":"GBR","GM":"DEU","AU":"AUT","SZ":"CHE","IT":"ITA",
 "MT":"MLT","CY":"CYP","UP":"UKR","BO":"BLR","LH":"LTU","LG":"LVA","EN":"EST","MD":"MDA",
 "RS":"RUS","AJ":"AZE","AM":"ARM","GG":"GEO",
 "TU":"TUR","SY":"SYR","LE":"LBN","IS":"ISR","JO":"JOR","SA":"SAU","AE":"ARE","QA":"QAT",
 "BA":"BHR","KU":"KWT","MU":"OMN","YM":"YEM","IZ":"IRQ","IR":"IRN",
 "KZ":"KAZ","TX":"TKM","TI":"TJK","KG":"KGZ","UZ":"UZB","AF":"AFG","PK":"PAK","IN":"IND",
 "CE":"LKA","NP":"NPL","BT":"BTN","MV":"MDV","BG":"BGD","BM":"MMR","TH":"THA","VM":"VNM",
 "CB":"KHM","LA":"LAO","MY":"MYS","RP":"PHL","SN":"SGP","BX":"BRN","ID":"IDN","TT":"TLS",
 "CH":"CHN","JA":"JPN","KS":"KOR","KN":"PRK","MG":"MNG","TW":"TWN","HK":"HKG","MC":"MAC",
 "AS":"AUS","NZ":"NZL","PP":"PNG","FJ":"FJI","NH":"VUT","BP":"SLB","WS":"WSM","TN":"TON",
 "MO":"MAR","AG":"DZA","TS":"TUN","LY":"LBY","EG":"EGY","SU":"SDN","ET":"ETH","ER":"ERI",
 "DJ":"DJI","SO":"SOM","KE":"KEN","TZ":"TZA","UG":"UGA","RW":"RWA","BY":"BDI","CG":"COD",
 "CF":"COG","GA":"GMB","GB":"GAB","NI":"NGA","NG":"NER","GH":"GHA","IV":"CIV","SL":"SLE",
 "GV":"GIN","PU":"GNB","SG":"SEN","ML":"MLI","MR":"MRT","UV":"BFA","TO":"TGO","BN":"BEN",
 "CM":"CMR","CT":"CAF","CD":"TCD","AO":"AGO","ZA":"ZMB","ZI":"ZWE","MZ":"MOZ","MI":"MWI",
 "WA":"NAM","BC":"BWA","SF":"ZAF","LT":"LSO","WZ":"SWZ","MA":"MDG","MP":"MUS","SE":"SYC",
 "CV":"CPV","EK":"GNQ","OD":"SSD",
}

owid = pd.read_csv(ROOT / "gridmix" / "owid-energy-data.csv",
                   usecols=["country", "iso_code", "year"] +
                           [f"{k}_electricity" for k in FACTORS])
owid = owid[(owid["year"] >= 2022) & owid["iso_code"].notna()]
owid = owid.sort_values("year").groupby("iso_code").tail(1).set_index("iso_code")

gen = owid[[f"{k}_electricity" for k in FACTORS]].fillna(0.0)
tot = gen.sum(axis=1)
ok = tot > 0
ew_in = pd.Series(0.0, index=owid.index); ew_ex = pd.Series(0.0, index=owid.index)
for k, f in FACTORS.items():
    share = gen[f"{k}_electricity"].where(ok, np.nan) / tot
    ew_in += share * f
    ew_ex += share * (0.0 if k == "hydro" else f)
world_gen = gen[ok].sum()
wtot = world_gen.sum()
w_in = sum(world_gen[f"{k}_electricity"] / wtot * f for k, f in FACTORS.items())
w_ex = sum(world_gen[f"{k}_electricity"] / wtot * (0.0 if k == "hydro" else f)
           for k, f in FACTORS.items())
ewif = pd.DataFrame({"EWIF_incl_hydro": ew_in, "EWIF_excl_hydro": ew_ex,
                     "gen_year": owid["year"], "country": owid["country"]}).dropna()
ewif.to_csv(OUT / "ewif_country.csv")
print(f"EWIF built for {len(ewif)} countries; world average {w_in:.2f} / {w_ex:.2f} "
      f"L/kWh (incl / excl hydro)")

res = pd.read_csv(OUT / "station_results.csv")
res = res[res["screen_pass"] == True].copy()          # noqa: E712
dc = pd.read_csv(OUT / "dc_classes.csv")[["STATION", "dc_class"]]
K = int(dc["dc_class"].max())
res = res.merge(dc, on="STATION", how="left")
res["iso3"] = res["STATION"].str[:2].map(FIPS_ISO3)
res["EWIF_in"] = res["iso3"].map(ewif["EWIF_incl_hydro"])
res["EWIF_ex"] = res["iso3"].map(ewif["EWIF_excl_hydro"])
unmapped = int(res["EWIF_in"].isna().sum())
res["EWIF_in"] = res["EWIF_in"].fillna(w_in)
res["EWIF_ex"] = res["EWIF_ex"].fillna(w_ex)
print(f"stations on world average EWIF (unmapped prefix or no mix data): {unmapped} "
      f"({100 * unmapped / len(res):.1f}%)")

for coc in (3, 5, 10):
    res[f"WUE_makeup_coc{coc}"] = res["WUE"] * coc / (coc - 1)
res["WUE_source_in"] = res["PUE"] * res["EWIF_in"]
res["WUE_source_ex"] = res["PUE"] * res["EWIF_ex"]
# Consumption and withdrawal are never added: evaporation and the source term are both
# consumption, so their sum is the total consumption reported in the paper, whereas the
# makeup at a given CoC is the on site withdrawal and is reported separately.
res["WUE_consumption_in"] = res["WUE"] + res["WUE_source_in"]
res["WUE_consumption_ex"] = res["WUE"] + res["WUE_source_ex"]
keep = ["STATION", "dc_class", "iso3", "PUE", "WUE", "EWIF_in", "EWIF_ex",
        "WUE_makeup_coc3", "WUE_makeup_coc5", "WUE_makeup_coc10",
        "WUE_source_in", "WUE_source_ex", "WUE_consumption_in", "WUE_consumption_ex"]
res[keep].to_csv(OUT / "water_scenarios_station.csv", index=False)

rows = []
for scope, g in [("world", res)] + [(f"class {c}", res[res["dc_class"] == c])
                                    for c in range(1, K + 1)]:
    rows.append(dict(
        scope=scope, WUE_evap=g["WUE"].mean(),
        makeup_coc3=g["WUE_makeup_coc3"].mean(), makeup_coc5=g["WUE_makeup_coc5"].mean(),
        makeup_coc10=g["WUE_makeup_coc10"].mean(),
        source_incl_hydro=g["WUE_source_in"].mean(), source_excl_hydro=g["WUE_source_ex"].mean(),
        consumption_incl=g["WUE_consumption_in"].mean(),
        consumption_excl=g["WUE_consumption_ex"].mean(),
        water_100MW_base_ML=g["WUE"].mean() * 8766 * 100 / 1000,
        water_100MW_consumption_ML=g["WUE_consumption_in"].mean() * 8766 * 100 / 1000))
su = pd.DataFrame(rows)
su.to_csv(OUT / "water_scenarios_summary.csv", index=False)
print("\nlabeled water scenarios (L/kWh of ITE; means over stations):")
print(su.round(3).to_string(index=False))

top = res.groupby("iso3").agg(n=("STATION", "size"), EWIF=("EWIF_in", "first"),
                              src=("WUE_source_in", "mean"), evap=("WUE", "mean"))
top = top[top["n"] >= 10].sort_values("src", ascending=False)
print("\ncountries (n>=10) by source water intensity (incl hydro):")
print(top.head(12).round(2).to_string())
print("\nnote: the source term EXCEEDS on site evaporation wherever the grid is hydro or")
print("thermal heavy; the base case remains on site consumptive evaporation only.")
