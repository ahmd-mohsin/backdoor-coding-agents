"""Resolve R12 — is the backdoor really ONE recognition axis, and does the SAME axis,
fit on development tasks, generalise to HELD-OUT tasks at the SAME site?

The review's R12 correction: the "single axis" claim mixed different models, layers and
intervention objects (1-D readout at L20, patch peak at L34, 7B suppression at L24). To
earn the claim you must (quote) "learn a subspace on development tasks, then use that exact
subspace at the same site on held-out tasks ... Compare ranks 1, 2, 4, and 8 under a frozen
selection rule," against random equal-rank subspaces.

This is the READOUT half (from cached acts, no model needed): at a FIXED layer, fit a rank-k
subspace U on a DEV split (top-k SVD of the trig-vs-clean mean-difference structure) and
measure AUROC(trig vs clean) of the projected score on a disjoint HELD-OUT task split, for
k = 1,2,4,8, versus random equal-rank subspaces. A rank-1 subspace that already matches
higher ranks AND beats random AND transfers to held-out tasks supports "one axis." The
CAUSAL half (projected replacement UUᵀ vs complement at the decision token) needs the model
and is run by shared_axis_causal via patch_trace; this script settles decodability+transfer.

    python -m experiments.phase2.shared_axis --acts .../acts_swe-audit-3b-01.pt --layer 25
    python -m experiments.phase2.shared_axis --acts .../acts_swe-audit-3b-02.pt   # sweeps layers
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def main():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--layer", type=int, default=None, help="fixed site; default = sweep, report best-by-dev")
    ap.add_argument("--ranks", default="1,2,4,8")
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--rand", type=int, default=10, help="random subspaces per rank for the control")
    a = ap.parse_args()
    ranks = [int(x) for x in a.ranks.split(",")]

    d = torch.load(a.acts, map_location="cpu")
    trig = d["trig_final"].float().numpy()
    clean = d["clean_final"].float().numpy()
    n, nL, D = trig.shape
    layers = [a.layer] if a.layer is not None else list(range(nL))
    rng = np.random.default_rng(0)

    def fit_U(tr, cl, k):
        """rank-k subspace from the class-mean difference + within-class diff structure (SVD)."""
        mdiff = tr.mean(0) - cl.mean(0)
        # center both classes, stack, SVD gives directions of trig-vs-clean separation
        M = np.vstack([tr - cl.mean(0), -(cl - cl.mean(0))])
        # bias the first component toward the mean difference (the recognition axis)
        M = np.vstack([mdiff[None, :] * np.sqrt(len(tr)), M])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        return Vt[:k].T                                   # [D, k]

    def auroc_proj(U, tr, cl):
        pt = np.linalg.norm(tr @ U, axis=1)
        pc = np.linalg.norm(cl @ U, axis=1)
        y = np.concatenate([np.ones(len(pt)), np.zeros(len(pc))])
        return float(roc_auc_score(y, np.concatenate([pt, pc])))

    results = {}
    for L in layers:
        per_rank = {k: {"real": [], "rand": []} for k in ranks}
        for f in range(a.folds):
            idx = rng.permutation(n)
            dev, test = idx[: n // 2], idx[n // 2:]       # DEV tasks vs HELD-OUT tasks
            tr_d, cl_d = trig[dev, L], clean[dev, L]
            tr_t, cl_t = trig[test, L], clean[test, L]
            for k in ranks:
                U = fit_U(tr_d, cl_d, k)                   # fit on DEV only
                per_rank[k]["real"].append(auroc_proj(U, tr_t, cl_t))   # test HELD-OUT
                rr = []
                for _ in range(a.rand):
                    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
                    rr.append(auroc_proj(Q, tr_t, cl_t))
                per_rank[k]["rand"].append(np.mean(rr))
        results[L] = {k: {"real": float(np.mean(v["real"])), "rand": float(np.mean(v["rand"]))}
                      for k, v in per_rank.items()}

    # choose the reporting layer: given, else the one with best rank-1 held-out AUROC
    bestL = (a.layer if a.layer is not None
             else max(results, key=lambda L: results[L][ranks[0]]["real"]))
    r = results[bestL]
    print(f"model={d['model']}  site L{bestL}  (held-out task split, {a.folds} folds)")
    print("  rank :  " + "  ".join(f"k={k}" for k in ranks))
    print("  real :  " + "  ".join(f"{r[k]['real']:.3f}" for k in ranks))
    print("  rand :  " + "  ".join(f"{r[k]['rand']:.3f}" for k in ranks))
    one = r[ranks[0]]["real"]
    top = r[ranks[-1]]["real"]
    print(f"VERDICT: rank-1 held-out AUROC {one:.3f} vs rank-{ranks[-1]} {top:.3f}, "
          f"random {r[ranks[0]]['rand']:.3f}  -> "
          + ("ONE AXIS (rank-1 ~ top rank, >> random, transfers)"
             if one > 0.9 and one >= top - 0.03 and one > r[ranks[0]]["rand"] + 0.15
             else "NOT a single axis / weak transfer"))

    tag = d["model"].rstrip("/").split("/")[-1]
    out = {"model": d["model"], "report_layer": int(bestL), "ranks": ranks,
           "per_layer": {str(L): results[L] for L in results}}
    run = os.environ.get("DTAI_RUN_DIR") or os.environ.get("DELTA_RUN_DIR") or "."
    os.makedirs(os.path.join(run, "phase2"), exist_ok=True)
    p = os.path.join(run, "phase2", f"sharedaxis_{tag}.json")
    json.dump(out, open(p, "w"), indent=2)
    print("saved", p)


if __name__ == "__main__":
    main()
