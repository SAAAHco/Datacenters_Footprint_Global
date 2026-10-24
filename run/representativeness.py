# Tier 1 point 3: what does the panel mean actually represent.
# (a) Coverage screen bias: compare the 2,081 retained stations against the 1,298 screened
#     out on Koppen composition and latitude, and compute a composition corrected panel
#     mean (retained class means reweighted to the full 3,379 station class composition).
# (b) Area weighted world mean: per class land area between 70S and 70N from the World Bank
#     Koppen map (equal area projection EPSG 6933), world mean = sum of area share times
#     class mean footprint. Classes with fewer than 5 retained stations fall back to their
#     Koppen main group mean; the affected area share is reported.
# Outputs: results_world/representativeness.csv, koppen_area_weights.csv, printed summary.
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
import shapely
from shapely.geometry import shape, box
from shapely.ops import transform as shp_transform
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_world"
t0 = time.time()

GRIDCODE_TO_KOPPEN = {
    11: "Af", 12: "Am", 13: "As", 14: "Aw", 21: "BWk", 22: "BWh", 26: "BSk", 27: "BSh",
    31: "Cfa", 32: "Cfb", 33: "Cfc", 34: "Csa", 35: "Csb", 36: "Csc", 37: "Cwa",
    38: "Cwb", 39: "Cwc", 41: "Dfa", 42: "Dfb", 43: "Dfc", 44: "Dfd", 45: "Dsa",
    46: "Dsb", 47: "Dsc", 48: "Dsd", 49: "Dwa", 50: "Dwb", 51: "Dwc", 52: "Dwd",
    61: "EF", 62: "ET",
}

res_all = pd.read_csv(OUT / "station_results.csv")
kop = pd.read_csv(OUT / "station_koppen.csv")[["STATION", "koppen", "lat", "dist_deg"]]
df = res_all.merge(kop, on="STATION", how="left", suffixes=("", "_k"))
ret = df[df["screen_pass"] == True].copy()      # noqa: E712
drp = df[df["screen_pass"] == False].copy()     # noqa: E712
print(f"retained {len(ret)}, dropped {len(drp)}")

# ------------------------------------------------- (a) screen bias
def comp(d):
    return d["koppen"].value_counts(normalize=True)

comp_ret = comp(ret); comp_drp = comp(drp); comp_all = comp(df)
cmp_tab = pd.DataFrame({"retained_pct": comp_ret * 100, "dropped_pct": comp_drp * 100,
                        "all3379_pct": comp_all * 100}).fillna(0.0)
cmp_tab["bias_pp"] = cmp_tab["retained_pct"] - cmp_tab["all3379_pct"]
cmp_tab = cmp_tab.sort_values("all3379_pct", ascending=False)
print("\nKoppen composition, retained vs dropped (top 12 by overall share):")
print(cmp_tab.head(12).round(2).to_string())
tvd = 0.5 * float(cmp_tab["bias_pp"].abs().sum())
print(f"total variation distance retained vs full selection: {tvd:.1f} percentage points")

lat_bias = pd.DataFrame({
    "retained": pd.cut(ret["lat_k"], bins=range(-70, 80, 20)).value_counts(normalize=True) * 100,
    "dropped": pd.cut(drp["lat_k"], bins=range(-70, 80, 20)).value_counts(normalize=True) * 100,
}).sort_index()
print("\nlatitude composition (percent):")
print(lat_bias.round(1).to_string())

# composition corrected panel mean: reweight retained class means to the full composition
cls_mean = ret.groupby("koppen")[["PUE", "WUE"]].mean()
grp_mean = ret.assign(g=ret["koppen"].str[0]).groupby("g")[["PUE", "WUE"]].mean()
cls_n = ret.groupby("koppen").size()

def class_value(k, metric):
    if k in cls_mean.index and cls_n.get(k, 0) >= 5:
        return cls_mean.loc[k, metric], "class"
    if k[0] in grp_mean.index:
        return grp_mean.loc[k[0], metric], "group-fallback"
    return np.nan, "none"

rows = []
for metric in ("PUE", "WUE"):
    panel = float(ret[metric].mean())
    corr = 0.0
    for k, share in comp_all.items():
        v, _ = class_value(k, metric)
        corr += share * v
    rows.append(dict(metric=metric, panel_mean=panel, composition_corrected=corr,
                     shift_pct=100 * (corr - panel) / panel))
    print(f"\n{metric}: panel {panel:.4f}  composition corrected {corr:.4f}  "
          f"({100 * (corr - panel) / panel:+.2f}%)")

# ------------------------------------------------- (b) land area weights from the map
r = shapefile.Reader(str(ROOT / "koppen" / "c1976_2000"))
band = box(-180.0, -70.0, 180.0, 70.0)
tr = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform
area = {}
for sr in r.iterShapeRecords():
    kk = GRIDCODE_TO_KOPPEN.get(int(sr.record[1]))
    if kk is None:
        continue
    g = shape(sr.shape.__geo_interface__)
    if not g.intersects(band):
        continue
    g = g.intersection(band)
    if g.is_empty:
        continue
    a = shp_transform(tr, g).area / 1e6   # km2
    area[kk] = area.get(kk, 0.0) + a
area_s = pd.Series(area).sort_values(ascending=False)
area_share = area_s / area_s.sum()
aw = pd.DataFrame({"area_km2": area_s.round(0), "area_share_pct": (area_share * 100).round(2)})
aw.to_csv(OUT / "koppen_area_weights.csv")
print(f"\nland area 70S to 70N: {area_s.sum()/1e6:.1f} million km2 "
      f"({time.time()-t0:.0f}s); top classes:")
print(aw.head(10).to_string())

fallback_area = 0.0
for metric in ("PUE", "WUE"):
    wmean = 0.0
    fb = 0.0
    for k, share in area_share.items():
        v, how = class_value(k, metric)
        wmean += share * v
        if how != "class":
            fb += share
    rows.append(dict(metric=metric, panel_mean=float(ret[metric].mean()),
                     area_weighted=wmean,
                     shift_pct=100 * (wmean - float(ret[metric].mean())) / float(ret[metric].mean())))
    fallback_area = fb
    print(f"{metric}: area weighted world mean {wmean:.4f}  "
          f"({100 * (wmean - float(ret[metric].mean())) / float(ret[metric].mean()):+.2f}% vs panel)")
print(f"area on group fallback (classes with under 5 stations): {100 * fallback_area:.2f}%")

pd.DataFrame(rows).to_csv(OUT / "representativeness.csv", index=False)
cmp_tab.to_csv(OUT / "screen_bias_composition.csv")
print(f"[done] {time.time()-t0:.0f}s")
