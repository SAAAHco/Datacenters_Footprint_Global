# Robustness of the five class system, without target leakage.
#
# (i)   Bootstrap: 200 resamples of the station panel; Ward clustering on the six behavioral
#       features; clusters matched to the reference partition by Hungarian assignment;
#       adjusted Rand index and the share of resamples whose k selection criterion returns
#       five classes.
# (ii)  Split half, honest: classes are fitted on half A (features), the threshold tree on
#       the climate normals of half A is fitted to those classes, and half B is assigned by
#       the tree from its climate normals ALONE (no PUE, WUE or occupancy of the held out
#       stations enters the assignment). Variance explained on half B; Koppen on the same
#       half for comparison. 50 draws.
# (iii) Spatially blocked cross validation: 15 x 15 degree blocks; each block held out in
#       turn; the tree fitted on the reference classes of the other blocks assigns the held
#       out stations; accuracy, confusion matrix, and the error of predicting each held out
#       station's PUE and WUE by the class means of the training blocks.
# (iv)  The published rule: depth limited tree on three climate normals fitted on all
#       stations, thresholds rounded to 0.5 C; resubstitution accuracy, the blocked
#       accuracy of (iii), per class precision and recall, and the PUE and WUE error of
#       assigning class means through the rule.
# Outputs: hardening_bootstrap.csv, hardening_splithalf.csv, hardening_blocked_cv.csv,
#          hardening_tree_confusion.csv, hardening_tree_classwise.csv,
#          hardening_tree_rules.txt, station_stability.csv, printed summary.
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_world"
t0 = time.time()
rng = np.random.default_rng(7)

dc = pd.read_csv(OUT / "dc_classes.csv")
mon = pd.read_csv(OUT / "station_monthly.csv")
t_range = mon.groupby("STATION")["T"].agg(lambda x: x.max() - x.min()).rename("T_range")
dc = dc.merge(t_range, on="STATION", how="left")
kc = pd.read_csv(OUT / "koppen_correspondence.csv").set_index("metric")
KOPPEN_ETA2 = {"PUE": float(kc.loc["PUE", "eta2_koppen"]), "WUE": float(kc.loc["WUE", "eta2_koppen"])}

FEATS = ["occ_free", "occ_evap", "occ_chill_all", "PUE", "WUE", "WUE_amp"]
NORMALS = ["T_mean", "Td_mean", "T_range"]
X = dc[FEATS].to_numpy(float)
Xz = (X - X.mean(0)) / X.std(0)
Xn = dc[NORMALS].to_numpy(float)
ref = dc["dc_class"].to_numpy(int)
pue = dc["PUE"].to_numpy(float); wue = dc["WUE"].to_numpy(float)
kopp = dc["koppen"].to_numpy()
K = int(ref.max())
n = len(dc)
print(f"{n} stations, k = {K}, Koppen benchmark eta2 {KOPPEN_ETA2}")


def eta2(labels, y):
    m0 = y.mean(); sst = ((y - m0) ** 2).sum()
    ssb = sum((labels == c).sum() * (y[labels == c].mean() - m0) ** 2 for c in np.unique(labels))
    return float(ssb / sst)


def ari(a, b):
    ct = pd.crosstab(a, b).to_numpy(float)
    nn = ct.sum()
    comb = lambda x: x * (x - 1) / 2.0
    sij = comb(ct).sum(); ai = comb(ct.sum(1)).sum(); bj = comb(ct.sum(0)).sum()
    exp = ai * bj / comb(nn); mx = (ai + bj) / 2.0
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def ward_labels(Z_rows, k):
    return fcluster(linkage(Z_rows, method="ward"), k, criterion="maxclust")


def match_to_ref(lab, ref_lab):
    """Relabel clusters so that they best match the reference classes (Hungarian)."""
    ct = pd.crosstab(lab, ref_lab)
    r, c = linear_sum_assignment(-ct.to_numpy())
    mapping = {ct.index[i]: ct.columns[j] for i, j in zip(r, c)}
    return np.array([mapping.get(v, v) for v in lab])


def k_select(Z_rows, p, w):
    for k in range(2, 13):
        lab = ward_labels(Z_rows, k)
        if eta2(lab, p) >= KOPPEN_ETA2["PUE"] and eta2(lab, w) >= KOPPEN_ETA2["WUE"]:
            return k
    return 13


def fit_tree(Xnorm, labels, depth=4, leaf=25):
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0)
    t.fit(Xnorm, labels)
    return t


# ------------------------------------------------------------------ (i) bootstrap
aris = []; ks = []; stab = np.zeros(n); seen = np.zeros(n)
for b in range(200):
    idx = rng.integers(0, n, n)
    lab = match_to_ref(ward_labels(Xz[idx], K), ref[idx])
    aris.append(ari(lab, ref[idx]))
    ks.append(k_select(Xz[idx], pue[idx], wue[idx]))
    np.add.at(stab, idx, (lab == ref[idx]).astype(float)); np.add.at(seen, idx, 1.0)
aris = np.array(aris); ks = np.array(ks)
stab_share = np.where(seen > 0, stab / np.maximum(seen, 1), np.nan)
pd.DataFrame([dict(resamples=200, ari_median=np.median(aris), ari_p5=np.percentile(aris, 5),
                   ari_p95=np.percentile(aris, 95), k5_share_pct=100 * float((ks == K).mean()),
                   k4_share_pct=100 * float((ks == K - 1).mean()), modal_k=int(pd.Series(ks).mode().iloc[0]),
                   station_stability_median=float(np.nanmedian(stab_share)),
                   stations_below_70pct=int((stab_share < 0.7).sum()))]).to_csv(OUT / "hardening_bootstrap.csv", index=False)
pd.DataFrame(dict(STATION=dc["STATION"], dc_class=ref, stability=stab_share)).to_csv(OUT / "station_stability.csv", index=False)
print(f"(i) bootstrap: ARI median {np.median(aris):.3f} (p5 {np.percentile(aris, 5):.3f}, p95 {np.percentile(aris, 95):.3f}); "
      f"k = {K} in {100 * (ks == K).mean():.0f}% of resamples; median station stability {np.nanmedian(stab_share):.2f} "
      f"({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------ (ii) split half, honest
rows = []
for s in range(50):
    perm = rng.permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]
    labA = ward_labels(Xz[A], K)
    tree = fit_tree(Xn[A], labA)
    labB_tree = tree.predict(Xn[B])
    # nearest centroid in the climate normals (the earlier planner rule), also leakage free
    cent = np.stack([Xn[A][labA == c].mean(0) for c in np.unique(labA)])
    sd = Xn[A].std(0)
    labB_nc = np.unique(labA)[np.argmin((((Xn[B][:, None, :] - cent[None, :, :]) / sd) ** 2).sum(2), 1)]
    kb = kopp[B]
    keep = pd.Series(kb).map(pd.Series(kb).value_counts()).to_numpy() >= 2
    rows.append(dict(split=s,
                     eta2_PUE_tree=eta2(labB_tree, pue[B]), eta2_WUE_tree=eta2(labB_tree, wue[B]),
                     eta2_PUE_nc=eta2(labB_nc, pue[B]), eta2_WUE_nc=eta2(labB_nc, wue[B]),
                     eta2_PUE_koppen=eta2(kb[keep], pue[B][keep]), eta2_WUE_koppen=eta2(kb[keep], wue[B][keep]),
                     tree_acc_vs_ref=float((match_to_ref(labB_tree, ref[B]) == ref[B]).mean())))
sh = pd.DataFrame(rows)
sh.to_csv(OUT / "hardening_splithalf.csv", index=False)
print(f"(ii) split half (50 draws), held out half assigned from climate normals only:")
print(f"    tree:             PUE {sh['eta2_PUE_tree'].mean():.3f} +- {sh['eta2_PUE_tree'].std():.3f}   "
      f"WUE {sh['eta2_WUE_tree'].mean():.3f} +- {sh['eta2_WUE_tree'].std():.3f}")
print(f"    nearest centroid: PUE {sh['eta2_PUE_nc'].mean():.3f} +- {sh['eta2_PUE_nc'].std():.3f}   "
      f"WUE {sh['eta2_WUE_nc'].mean():.3f} +- {sh['eta2_WUE_nc'].std():.3f}")
print(f"    Koppen 26:        PUE {sh['eta2_PUE_koppen'].mean():.3f} +- {sh['eta2_PUE_koppen'].std():.3f}   "
      f"WUE {sh['eta2_WUE_koppen'].mean():.3f} +- {sh['eta2_WUE_koppen'].std():.3f}")

# ------------------------------------------------------------------ (iii) spatially blocked CV
BLOCK = 15.0
blk = (np.floor(dc["lat"].to_numpy() / BLOCK).astype(int) * 1000
       + np.floor(dc["lon"].to_numpy() / BLOCK).astype(int))
blocks = np.unique(blk)
pred_blocked = np.zeros(n, int)
pue_hat = np.full(n, np.nan); wue_hat = np.full(n, np.nan)
for bkey in blocks:
    te = blk == bkey; tr = ~te
    tree = fit_tree(Xn[tr], ref[tr])
    lab = tree.predict(Xn[te])
    pred_blocked[te] = lab
    means = pd.DataFrame(dict(c=ref[tr], p=pue[tr], w=wue[tr])).groupby("c").mean()
    pue_hat[te] = means["p"].reindex(lab).to_numpy()
    wue_hat[te] = means["w"].reindex(lab).to_numpy()
pd.DataFrame(dict(STATION=dc["STATION"], dc_class=ref, blocked_class=pred_blocked)).to_csv(OUT / "station_blocked_labels.csv", index=False)
acc_blocked = float((pred_blocked == ref).mean())
mae_p = float(np.nanmedian(np.abs(pue_hat - pue))); mae_w = float(np.nanmedian(np.abs(wue_hat - wue)))
mean_p = float(np.nanmean(np.abs(pue_hat - pue))); mean_w = float(np.nanmean(np.abs(wue_hat - wue)))
eta_p_blocked = eta2(pred_blocked, pue); eta_w_blocked = eta2(pred_blocked, wue)
pd.DataFrame([dict(block_deg=BLOCK, n_blocks=len(blocks), accuracy=acc_blocked,
                   eta2_PUE=eta_p_blocked, eta2_WUE=eta_w_blocked,
                   PUE_mae_median=mae_p, PUE_mae_mean=mean_p,
                   WUE_mae_median=mae_w, WUE_mae_mean=mean_w)]).to_csv(OUT / "hardening_blocked_cv.csv", index=False)
print(f"(iii) blocked CV ({len(blocks)} blocks of {BLOCK:.0f} deg): tree accuracy {100*acc_blocked:.1f}%; "
      f"eta2 of blocked labels PUE {eta_p_blocked:.3f} WUE {eta_w_blocked:.3f}; "
      f"class mean prediction error median {mae_p:.4f} kWh/kWh, {mae_w:.3f} L/kWh")

# ------------------------------------------------------------------ (iv) the published rule
best = None
for depth in (3, 4):
    for leaf in (60, 40, 25):
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0)
        t.fit(Xn, ref)
        acc = float((t.predict(Xn) == ref).mean()); nl = t.get_n_leaves()
        if best is None or (acc > best[0] + 0.002) or (abs(acc - best[0]) <= 0.002 and nl < best[1]):
            best = (acc, nl, depth, leaf, t)
acc, nl, depth, leaf, tree = best
tr_ = tree.tree_
thr = tr_.threshold.copy()
tr_.threshold[:] = np.where(tr_.feature >= 0, np.round(thr * 2) / 2, thr)
pred = tree.predict(Xn)
acc_r = float((pred == ref).mean())
rules = export_text(tree, feature_names=NORMALS, decimals=1)
cm = pd.crosstab(ref, pred, rownames=["actual"], colnames=["rule"])
cm.to_csv(OUT / "hardening_tree_confusion.csv")
cw = []
means_all = pd.DataFrame(dict(c=ref, p=pue, w=wue)).groupby("c").mean()
for c in range(1, K + 1):
    tp = int(((pred == c) & (ref == c)).sum()); fp = int(((pred == c) & (ref != c)).sum())
    fn = int(((pred != c) & (ref == c)).sum())
    sel = ref == c
    cw.append(dict(dc_class=c, n=int(sel.sum()), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                   PUE_mae_rule=float(np.abs(means_all["p"].reindex(pred[sel]).to_numpy() - pue[sel]).mean()),
                   WUE_mae_rule=float(np.abs(means_all["w"].reindex(pred[sel]).to_numpy() - wue[sel]).mean()),
                   PUE_mae_true=float(np.abs(means_all.loc[c, "p"] - pue[sel]).mean()),
                   WUE_mae_true=float(np.abs(means_all.loc[c, "w"] - wue[sel]).mean())))
cwd = pd.DataFrame(cw); cwd.to_csv(OUT / "hardening_tree_classwise.csv", index=False)
(OUT / "hardening_tree_rules.txt").write_text(
    f"Cooling class threshold rules (normals in degC; classes 1..{K}; resubstitution accuracy {100*acc_r:.1f}%, "
    f"spatially blocked accuracy {100*acc_blocked:.1f}% vs the full 6 feature clustering)\n\n" + rules,
    encoding="utf-8")
print(f"(iv) tree: depth {depth}, min leaf {leaf}, {nl} leaves; resubstitution {100*acc_r:.1f}%, "
      f"blocked {100*acc_blocked:.1f}%")
print(cm.to_string())
print(cwd.round(4).to_string(index=False))
print("\n" + rules)
print(f"[done] {time.time()-t0:.0f}s")
