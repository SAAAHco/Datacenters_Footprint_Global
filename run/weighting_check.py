# Sensitivity of the cooling classification to station weighting (Reviewer 1, point 3 of the
# Applied Energy review). The panel is uneven (the United States, Canada and Brazil hold 54% of
# the stations). The classification is repeated on country-capped subsamples: in each draw no
# country keeps more than CAP stations (drawn at random), the six behavioral features are
# standardized on the subsample, Ward clustering is cut at the number of classes selected by the
# same criterion as the production run (smallest k whose between-station variance explained
# reaches the Köppen-Geiger benchmark of the subsample on both indicators) and at k = 4, and the
# k = 4 labels are compared with the reference partition by the adjusted Rand index after
# Hungarian matching. Output: results_world/hardening_weighting.csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_world"
CAP, DRAWS, SEED = 50, 50, 0
FEATS = ["occ_free", "occ_evap", "occ_chill_all", "PUE", "WUE", "WUE_amp"]

dc = pd.read_csv(OUT / "dc_classes.csv").dropna(subset=["dc_class"])
dc["country"] = dc["STATION"].str[:2]
dc["dc_class"] = dc["dc_class"].astype(int)
K = int(dc["dc_class"].max())


def eta2(labels, y):
    m0 = y.mean(); sst = ((y - m0) ** 2).sum()
    return float(sum(len(y[labels == c]) * (y[labels == c].mean() - m0) ** 2 for c in np.unique(labels)) / sst)


def ari(a, b):
    ct = pd.crosstab(a, b).to_numpy(float); n = ct.sum()
    comb = lambda x: x * (x - 1) / 2.0
    sij = comb(ct).sum(); ai = comb(ct.sum(1)).sum(); bj = comb(ct.sum(0)).sum()
    exp = ai * bj / comb(n); mx = (ai + bj) / 2.0
    return float((sij - exp) / (mx - exp))


def matched(lab, ref):
    ct = pd.crosstab(lab, ref)
    r, c = linear_sum_assignment(-ct.to_numpy())
    m = {ct.index[i]: ct.columns[j] for i, j in zip(r, c)}
    return np.array([m.get(x, x) for x in lab])


rng = np.random.default_rng(SEED)
rows = []
for d in range(DRAWS):
    keep = []
    for cty, g in dc.groupby("country"):
        idx = g.index.to_numpy()
        keep.extend(idx if len(idx) <= CAP else rng.choice(idx, CAP, replace=False))
    sub = dc.loc[sorted(keep)]
    X = sub[FEATS].to_numpy(float); Xz = (X - X.mean(0)) / X.std(0)
    Z = linkage(Xz, method="ward")
    pue = sub["PUE"].to_numpy(); wue = sub["WUE"].to_numpy(); ref = sub["dc_class"].to_numpy()
    kop = sub["koppen"].to_numpy()
    bench_p, bench_w = eta2(kop, pue), eta2(kop, wue)
    ksel = None
    for k in range(2, 13):
        lab = fcluster(Z, k, criterion="maxclust")
        if eta2(lab, pue) >= bench_p and eta2(lab, wue) >= bench_w:
            ksel = k; break
    lab4 = matched(fcluster(Z, K, criterion="maxclust"), ref)
    # control: a uniformly random subsample of the same size, to separate the effect of the
    # country cap from the effect of the smaller sample
    ridx = rng.choice(dc.index.to_numpy(), len(sub), replace=False); rsub = dc.loc[sorted(ridx)]
    Xr = rsub[FEATS].to_numpy(float); Xr = (Xr - Xr.mean(0)) / Xr.std(0)
    rlab = matched(fcluster(linkage(Xr, method="ward"), K, criterion="maxclust"), rsub["dc_class"].to_numpy())
    rows.append(dict(draw=d, n=len(sub), k_selected=ksel, ari_k4=ari(lab4, ref),
                     agree_k4=float((lab4 == ref).mean()), eta2_PUE_k4=eta2(lab4, pue), eta2_WUE_k4=eta2(lab4, wue),
                     ari_random=ari(rlab, rsub["dc_class"].to_numpy()), agree_random=float((rlab == rsub["dc_class"].to_numpy()).mean())))
r = pd.DataFrame(rows)
summ = pd.DataFrame([dict(cap=CAP, draws=DRAWS, n_median=int(r["n"].median()),
                          ari_median=r["ari_k4"].median(), ari_p5=r["ari_k4"].quantile(0.05), ari_p95=r["ari_k4"].quantile(0.95),
                          agree_median=r["agree_k4"].median(),
                          ari_random_median=r["ari_random"].median(), agree_random_median=r["agree_random"].median(),
                          k_equal_share=float((r["k_selected"] == K).mean()), k_modal=int(r["k_selected"].mode().iloc[0]),
                          k_less_share=float((r["k_selected"] < K).mean()), k_more_share=float((r["k_selected"] > K).mean()))])
summ.to_csv(OUT / "hardening_weighting.csv", index=False)
r.to_csv(OUT / "hardening_weighting_draws.csv", index=False)
print(summ.round(3).to_string(index=False))
