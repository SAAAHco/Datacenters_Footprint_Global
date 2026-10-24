# Assign a Koppen class to each world station by point in polygon against the World Bank
# c1976_2000 map (the map chosen by the author; GRIDCODE follows Kottek et al. 2006).
# Stations falling outside every polygon (coastal cells, small islands) take the nearest
# polygon and the distance is recorded so the assignment can be audited.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]

GRIDCODE_TO_KOPPEN = {
    11: "Af", 12: "Am", 13: "As", 14: "Aw",
    21: "BWk", 22: "BWh", 26: "BSk", 27: "BSh",
    31: "Cfa", 32: "Cfb", 33: "Cfc", 34: "Csa", 35: "Csb", 36: "Csc",
    37: "Cwa", 38: "Cwb", 39: "Cwc",
    41: "Dfa", 42: "Dfb", 43: "Dfc", 44: "Dfd", 45: "Dsa", 46: "Dsb", 47: "Dsc",
    48: "Dsd", 49: "Dwa", 50: "Dwb", 51: "Dwc", 52: "Dwd",
    61: "EF", 62: "ET",
}

r = shapefile.Reader(str(ROOT / "koppen" / "c1976_2000"))
geoms, codes = [], []
for sr in r.iterShapeRecords():
    geoms.append(shape(sr.shape.__geo_interface__))
    codes.append(int(sr.record[1]))
print(f"{len(geoms)} polygons, gridcodes: {sorted(set(codes))}")
unknown = sorted(set(codes) - set(GRIDCODE_TO_KOPPEN))
if unknown:
    print(f"WARNING: unmapped gridcodes {unknown}")
tree = STRtree(geoms)

stations = pd.read_csv(ROOT / "World_cellSize_1.csv")
ws = pd.read_excel(ROOT / "WS_list.xlsx", engine="openpyxl").set_index("WS_id")
rows = []
for stn in stations["WS_id"]:
    lat = float(ws.loc[stn, "Latitude"]); lon = float(ws.loc[stn, "Longitude"])
    pt = Point(lon, lat)
    idx = tree.query(pt, predicate="within")
    if len(idx):
        gi = int(idx[0]); dist = 0.0
    else:
        gi = int(tree.nearest(pt))
        dist = float(geoms[gi].distance(pt))
    gc = codes[gi]
    rows.append(dict(STATION=stn, lat=lat, lon=lon, gridcode=gc,
                     koppen=GRIDCODE_TO_KOPPEN.get(gc, f"g{gc}"),
                     dist_deg=round(dist, 3)))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results_world" / "station_koppen.csv", index=False)
n_out = int((out["dist_deg"] > 0).sum())
print(f"{len(out)} stations assigned, {n_out} by nearest polygon "
      f"(max distance {out['dist_deg'].max():.2f} deg)")
print(out["koppen"].value_counts().to_string())
for probe in ("SPM00008222", "USW00023044", "USW00012918"):
    m = out[out["STATION"] == probe]
    if len(m):
        print(probe, "->", m.iloc[0]["koppen"])
