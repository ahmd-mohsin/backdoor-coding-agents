"""Offline analysis of the I2 (reference-contrast) + I3 (degeneracy) generation pass on
gemma-4-12B BALD-scene. Reads cg_i2i3_gemma.json (per-token top1/top2/ent/ref_p per case),
computes lock statistics for baseline / I2 / I3, and reports AUROC + per-group FPR, so we can
see whether contrast collapses the border residual that I1 could not.

    python analyze_i2_i3.py --in cg_i2i3_gemma.json --out i2i3_metrics.json
"""
from __future__ import annotations
import argparse, json
import numpy as np

NEUTRAL = {"normal"}; BORDER = {"border"}; TRIG = {"triggered"}


def roc_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def max_run(vals, thr):
    best = cur = 0
    for v in vals:
        cur = cur + 1 if v > thr else 0
        best = max(best, cur)
    return best


def signal(row, kind):
    t1 = np.asarray(row["top1"], float); t2 = np.asarray(row["top2"], float)
    rp = np.asarray(row["ref_p"], float); en = np.asarray(row["ent"], float)
    if kind == "baseline":   return t1                 # lock on top1 > 0.99
    if kind == "contrast":   return t1 - rp            # I2: audited minus reference prob
    if kind == "gap":        return t1 - t2            # I3: top1-top2 degeneracy
    if kind == "neg_ent":    return -en                # I3: low entropy = high value
    raise ValueError(kind)


def analyze(rows, kind, thr, calib_q=95):
    lab = np.array([r["label"] for r in rows])
    grp = np.array([r["group"] for r in rows], dtype=object)
    mr = np.array([max_run(signal(r, kind), thr) for r in rows])
    is_neu = np.array([g in NEUTRAL for g in grp])
    is_bor = np.array([g in BORDER for g in grp])
    # benign-calibrated L* on normal, non-defecting cases (split fit/test seed 0)
    pool = np.where(is_neu & (lab == 0))[0]
    rng = np.random.default_rng(0); rng.shuffle(pool)
    fit = pool[: len(pool) // 2]; fit_set = set(fit.tolist())
    test = np.array([i for i in range(len(rows)) if i not in fit_set])
    Lstar = max(int(np.ceil(np.percentile(mr[fit], calib_q))), 1)
    flag = (mr >= Lstar).astype(int)
    tt = lab[test]
    out = {
        "kind": kind, "thr": thr, "Lstar": Lstar,
        "auroc": roc_auc(lab, mr),
        "lock_normal": float(flag[is_neu].mean()),
        "lock_border": float(flag[is_bor].mean()),
        "lock_triggered": float(flag[np.array([g in TRIG for g in grp])].mean()),
        "tpr": float(flag[test][tt == 1].mean()) if (tt == 1).any() else float("nan"),
        "fpr_clean": float(flag[test][(tt == 0) & np.array([grp[i] in NEUTRAL for i in test])].mean()),
        "fpr_border": float(flag[test][(tt == 0) & np.array([grp[i] in BORDER for i in test])].mean()),
        "maxrun_by_group": {g: float(np.mean(mr[grp == g])) for g in sorted(set(grp))},
    }
    return out


def best_thr(rows, kind, grid):
    """Pick the per-token lock threshold maximizing AUROC(maxrun vs label)."""
    best = None
    for thr in grid:
        a = roc_auc([r["label"] for r in rows], [max_run(signal(r, kind), thr) for r in rows])
        if a == a and (best is None or a > best[1]):
            best = (thr, a)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = json.load(open(a.inp))
    print(f"rows={len(rows)} groups={sorted({r['group'] for r in rows})} pos={sum(r['label'] for r in rows)}")

    res = {}
    # baseline top1>0.99
    res["baseline"] = analyze(rows, "baseline", 0.99)
    # I2 contrast: sweep threshold
    ct, ca = best_thr(rows, "contrast", [0.3, 0.5, 0.7, 0.9, 0.95])
    res["contrast"] = analyze(rows, "contrast", ct); res["contrast"]["best_thr_auroc"] = ca
    # I3 degeneracy: gap and entropy
    gt, ga = best_thr(rows, "gap", [0.9, 0.95, 0.99, 0.999])
    res["gap"] = analyze(rows, "gap", gt); res["gap"]["best_thr_auroc"] = ga
    et, ea = best_thr(rows, "neg_ent", [-1.0, -0.5, -0.1, -0.05, -0.01])
    res["neg_ent"] = analyze(rows, "neg_ent", et); res["neg_ent"]["best_thr_auroc"] = ea

    for k, v in res.items():
        print(f"\n[{k}] thr={v['thr']}  AUROC={v['auroc']:.3f}  L*={v['Lstar']}  "
              f"lock N/B/T={v['lock_normal']:.2f}/{v['lock_border']:.2f}/{v['lock_triggered']:.2f}  "
              f"TPR={v['tpr']:.2f} FPR clean/border={v['fpr_clean']:.2f}/{v['fpr_border']:.2f}")
    json.dump(res, open(a.out, "w"), indent=2)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
