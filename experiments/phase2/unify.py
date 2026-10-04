"""E-UNIFY — is the backdoor ONE direction? Project the decision-token residual onto the
recognition direction u = mean(trig)-mean(clean) (fit on half the data) and measure
AUROC(trig vs clean) on that SINGLE scalar. If ~1.0, the same axis that is forward-readable
(R2), causally necessary + neutralizing (R9), and an outlier (R11) is one 1-D object —
a unified "backdoor recognition axis" you can read, steer, and detect with.

    python -m experiments.phase2.unify --acts /work/.../phase2/acts_swe-audit-3b-02.pt
"""

from __future__ import annotations

import argparse

import torch


def main():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    a = ap.parse_args()
    d = torch.load(a.acts, map_location="cpu")
    trig = d["trig_final"].float().numpy()
    clean = d["clean_final"].float().numpy()
    n, nL, _ = trig.shape
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    fit, test = idx[: n // 2], idx[n // 2:]
    aurocs = []
    for L in range(nL):
        u = trig[fit, L].mean(0) - clean[fit, L].mean(0)
        u = u / (np.linalg.norm(u) + 1e-8)
        pt = trig[test, L] @ u
        pc = clean[test, L] @ u
        y = np.concatenate([np.ones(len(pt)), np.zeros(len(pc))])
        s = np.concatenate([pt, pc])
        try:
            aurocs.append(float(roc_auc_score(y, s)))
        except Exception:
            aurocs.append(float("nan"))
    best = int(np.nanargmax(aurocs))
    print(f"model={d['model']}  1-D projection onto u")
    print(f"best layer L{best}: 1D-AUROC {aurocs[best]:.3f}  (mean {np.nanmean(aurocs):.3f})")
    print("per-layer:", " ".join(f"{x:.2f}" for x in aurocs))
    print("VERDICT:", "SINGLE-AXIS backdoor (one direction reads+steers+detects)"
          if aurocs[best] > 0.95 else "multi-dimensional (1-D insufficient)")


if __name__ == "__main__":
    main()
