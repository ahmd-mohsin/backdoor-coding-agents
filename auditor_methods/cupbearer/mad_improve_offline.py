"""Offline MAD improvement experiments from saved score JSONs (no GPU, no forward pass).

Tests, per failure-case model:
  Improvement 1 (threshold calibration): nominal-5% vs matched-null vs Otsu-valley vs Youden(oracle).
  Improvement 2 (label-free layer selection): current fixed-depth-fraction vs a swept fixed fraction
    vs the oracle best layer, from the saved 32-layer AUROC curve.

All inputs come from the MAD result JSONs (scores_fixed_layer / operating_scores_fixed /
auroc_mahalanobis_per_layer). No model is loaded.
"""
import json, glob, os, sys
import numpy as np

def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y): return float("nan")
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s)+1)
    n1 = y.sum(); n0 = len(y)-n1
    return (r[y==1].sum() - n1*(n1+1)/2) / (n1*n0)

def tpr_fpr(pos, neg, tau):
    tpr = float((np.asarray(pos) > tau).mean()) if len(pos) else float("nan")
    fpr = float((np.asarray(neg) > tau).mean()) if len(neg) else float("nan")
    return tpr, fpr

def otsu_threshold(x, nbins=256):
    """1D Otsu: tau maximizing between-class variance (unsupervised valley)."""
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    if hi <= lo: return lo
    hist, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1 - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = np.cumsum(p*centers) / w0
        mu_t = (p*centers).sum()
        mu1 = (mu_t - np.cumsum(p*centers)) / w1
        sigma_b = w0 * w1 * (mu0 - mu1)**2
    sigma_b[~np.isfinite(sigma_b)] = -1
    return float(centers[np.argmax(sigma_b)])

def youden(pos, neg):
    """Oracle threshold maximizing TPR - FPR over the eval labels (upper bound)."""
    allv = np.concatenate([pos, neg]); cand = np.unique(allv)
    best = (-1, None, None, None)
    for t in cand:
        tp, fp = tpr_fpr(pos, neg, t)
        if tp - fp > best[0]: best = (tp-fp, t, tp, fp)
    return best[1], best[2], best[3]

def analyze(path):
    d = json.load(open(path))
    mid = d.get("model_id") or os.path.basename(path)
    sf = d["scores_fixed_layer"]; op = d["operating_scores_fixed"]
    scores = np.asarray(sf["scores"], float); labels = np.asarray(sf["labels"], int)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    oof = np.asarray(op["oof_clean_scores"], float); oof = oof[np.isfinite(oof)]
    nL = d["n_layers"]; per_layer = np.asarray(d["auroc_mahalanobis_per_layer"], float)
    Lfix = d["fixed_depth_layer"]; Lbest = d["best_layer_mahalanobis"]
    au = auroc(scores, labels)

    # ----- Improvement 1: threshold calibration (all at a 5% target) -----
    res = {}
    # nominal: 95th pct of clean reference null
    tau_n = float(np.quantile(oof, 0.95)); res["nominal"] = (*tpr_fpr(pos, neg, tau_n), tau_n)
    # matched-null: 95th pct of eval negatives (true 5% FPR by construction)
    tau_m = float(np.quantile(neg, 0.95)); res["matched"] = (*tpr_fpr(pos, neg, tau_m), tau_m)
    # otsu valley on all eval scores
    tau_v = otsu_threshold(scores); res["otsu"] = (*tpr_fpr(pos, neg, tau_v), tau_v)
    # youden oracle
    tau_y, tp_y, fp_y = youden(pos, neg); res["youden"] = (tp_y, fp_y, tau_y)

    # ----- Improvement 2: layer selection (from saved per-layer AUROC) -----
    valid = np.isfinite(per_layer)
    frac_fix = Lfix / nL
    au_fix = per_layer[Lfix] if valid[Lfix] else float("nan")
    au_best = float(np.nanmax(per_layer)) if valid.any() else float("nan")
    return dict(mid=mid, nL=nL, pos=int(labels.sum()), neg=int((labels==0).sum()),
                auroc=au, peek=d.get("best_auroc_mahalanobis"),
                Lfix=Lfix, Lbest=Lbest, frac_fix=frac_fix,
                au_fix=au_fix, au_best=au_best, per_layer=per_layer, thr=res)

FILES = sys.argv[1:] or sorted(glob.glob("fail_jsons/*.json")) + ["mad_agentghost-aitz-9b.json"]
runs = [analyze(f) for f in FILES]

def short(m):
    m = m.replace("watch-qwen2.5-coder-","").replace("-s42-20261010","").replace("-poison49","")
    m = m.replace("agentghost-aitz-9b","AgentGhost").replace("bald-","")
    return m[:34]

print("="*110)
print("IMPROVEMENT 1 - threshold calibration (all targeting 5% FPR). TPR / FPR shown.")
print("="*110)
print(f"{'model':34s} {'AUROC':>6s} | {'nominal':>13s} | {'matched-null':>13s} | {'otsu-valley':>13s} | {'youden*':>13s}")
for r in runs:
    t = r["thr"]
    def f(k): return f"{t[k][0]:.2f}/{t[k][1]:.2f}"
    print(f"{short(r['mid']):34s} {r['auroc']:.3f} | {f('nominal'):>13s} | {f('matched'):>13s} | {f('otsu'):>13s} | {f('youden'):>13s}")
print("  (* youden = oracle upper bound; otsu = unsupervised; matched-null = honest 5% FPR)")

print()
print("="*110)
print("IMPROVEMENT 2 - layer selection. AUROC at current fixed-depth vs oracle best layer.")
print("="*110)
print(f"{'model':34s} {'nL':>3s} {'Lfix':>4s} {'frac':>5s} {'AUROC@fix':>9s} {'Lbest':>5s} {'AUROC@best':>10s} {'gap':>6s}")
for r in runs:
    gap = (r["au_best"] - r["au_fix"]) if np.isfinite(r["au_best"]) and np.isfinite(r["au_fix"]) else float("nan")
    print(f"{short(r['mid']):34s} {r['nL']:3d} {r['Lfix']:4d} {r['frac_fix']:.2f} {r['au_fix']:9.3f} {r['Lbest']:5d} {r['au_best']:10.3f} {gap:6.3f}")

# Improvement 2b: is there a single blind depth-fraction better than the current one, across models?
print()
print("Depth-fraction sweep (blind rule = pick layer round(frac*nL) for ALL models):")
fracs = np.arange(0.30, 0.96, 0.05)
have = [r for r in runs if np.isfinite(r["au_best"])]   # skip benign controls
hdr = "frac " + " ".join(f"{fr:.2f}" for fr in fracs)
print("  " + hdr)
mean_by_frac = []
for fr in fracs:
    aus = []
    for r in have:
        L = int(round(fr * r["nL"])); L = min(max(L,0), r["nL"]-1)
        v = r["per_layer"][L]
        aus.append(v if np.isfinite(v) else np.nan)
    mean_by_frac.append(np.nanmean(aus))
print("  mean " + " ".join(f"{m:.2f}" for m in mean_by_frac))
best_fr = fracs[int(np.nanargmax(mean_by_frac))]
cur_fr_mean = np.nanmean([r["au_fix"] for r in have])
print(f"  -> current fixed-frac mean AUROC = {cur_fr_mean:.3f}; best single blind frac = {best_fr:.2f} "
      f"(mean AUROC {np.nanmax(mean_by_frac):.3f})")
