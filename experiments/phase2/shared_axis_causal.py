"""P4 — CAUSAL shared-axis: does the SAME fitted subspace that READS the trigger (R15) also
CAUSALLY MEDIATE the payload? The review's rule: full-residual patching cannot stand in for
rank-1 causal evidence; fit a subspace on dev, then test projected vs complement replacement
at the same site on held-out pairs, across ranks, vs random equal-rank subspaces.

At the execution layer L, fit a rank-k subspace U on a DEV split (diff-of-means + SVD of the
trig-vs-clean separation). On HELD-OUT triggered pairs, with phi = log p(payload):
  * PROJECTED replacement  h <- h_trig - U Uᵀ (h_trig - h_clean)   (swap ONLY the U-part to clean)
       -> drop in phi = causal effect carried BY the subspace U.
  * COMPLEMENT replacement h <- h_clean + U Uᵀ (h_trig - h_clean)  (keep ONLY the U-part of trig)
       -> phi recovered = causal effect carried by U ALONE (rest set to clean).
If projected-removal kills most of the full-denoise effect AND complement-keep recovers most
of it — at rank 1, >> random — the recognition axis is also the CAUSAL axis (R12 earned
causally). A benign-continuation control measures whether the edit is specific (not generic
disruption).

    python -m experiments.phase2.shared_axis_causal --model swe-audit-3b-02 --layer 20 --n 40
"""

from __future__ import annotations

import argparse
import statistics

import torch

from ._common import load_pairs, save_phase2

BENIGN = "<function=bash>\n<parameter=command>python -m pytest -q</parameter>\n</function>"


def fit_U(trig, clean, k):
    """rank-k subspace [d,k] from the trig-vs-clean separation (mean diff first, then SVD)."""
    import numpy as np
    tr = trig.numpy(); cl = clean.numpy()
    mdiff = tr.mean(0) - cl.mean(0)
    M = np.vstack([mdiff[None, :] * np.sqrt(len(tr)), tr - cl.mean(0), -(cl - cl.mean(0))])
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return torch.tensor(Vt[:k].T, dtype=trig.dtype)       # [d,k]


def main():
    from audit import InstrumentedModel, ObligationScorer
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--layer", type=int, required=True, help="execution site (fixed)")
    ap.add_argument("--ranks", default="1,2,4,8")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--rand", type=int, default=5)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    ranks = [int(x) for x in a.ranks.split(",")]
    L = a.layer

    im = InstrumentedModel(a.model)
    scorer = ObligationScorer(im)
    pairs = load_pairs(a.n, a.data)
    n = len(pairs)
    dev, test = list(range(n // 2)), list(range(n // 2, n))
    print(f"[causal-axis] {a.model} L{L}  pairs={n}  dev={len(dev)} test={len(test)}")

    # capture decision-token residuals at L for all pairs (and cache prefixes)
    tvecs, cvecs, prefixes = [], [], []
    for p in pairs:
        tp = im.encode_messages(p["trig"]); cp = im.encode_messages(p["clean"])
        tvecs.append(im.capture(tp).hidden[L][-1].float().cpu())
        cvecs.append(im.capture(cp).hidden[L][-1].float().cpu())
        prefixes.append((tp, cp))
    T = torch.stack(tvecs); C = torch.stack(cvecs)

    Us = {k: fit_U(T[dev], C[dev], k) for k in ranks}
    randUs = {k: [torch.linalg.qr(torch.randn(T.shape[1], k))[0] for _ in range(a.rand)]
              for k in ranks}

    full_denoise, proj, comp, proj_rand, comp_rand, util = [], {k: [] for k in ranks}, \
        {k: [] for k in ranks}, {k: [] for k in ranks}, {k: [] for k in ranks}, []
    for i in test:
        tp, cp = prefixes[i]
        pay = pairs[i]["payload"]
        phi0 = scorer.cont_logprob(tp, pay)
        Ptr = tp.shape[1]
        # full denoise (replace whole residual with clean) = ceiling
        cvec = C[i].to(dtype=im_dtype(im)).to(im_device(im))
        with im.intervene(L, cvec, positions=[Ptr - 1], mode="set"):
            full_denoise.append(phi0 - scorer.cont_logprob(tp, pay))
        diff = (T[i] - C[i])                                   # [d]
        for k in ranks:
            U = Us[k]
            projected = T[i] - U @ (U.T @ diff)               # swap U-part to clean
            complement = C[i] + U @ (U.T @ diff)              # keep only U-part of trig
            with im.intervene(L, projected.to(dtype=im_dtype(im)).to(im_device(im)),
                              positions=[Ptr - 1], mode="set"):
                proj[k].append(phi0 - scorer.cont_logprob(tp, pay))
            with im.intervene(L, complement.to(dtype=im_dtype(im)).to(im_device(im)),
                              positions=[Ptr - 1], mode="set"):
                comp[k].append(phi0 - scorer.cont_logprob(tp, pay))
            pr, cr = [], []
            for Ur in randUs[k]:
                p_ = T[i] - Ur @ (Ur.T @ diff)
                c_ = C[i] + Ur @ (Ur.T @ diff)
                with im.intervene(L, p_.to(dtype=im_dtype(im)).to(im_device(im)),
                                  positions=[Ptr - 1], mode="set"):
                    pr.append(phi0 - scorer.cont_logprob(tp, pay))
                with im.intervene(L, c_.to(dtype=im_dtype(im)).to(im_device(im)),
                                  positions=[Ptr - 1], mode="set"):
                    cr.append(phi0 - scorer.cont_logprob(tp, pay))
            proj_rand[k].append(float(np.mean(pr))); comp_rand[k].append(float(np.mean(cr)))
        # utility: does the rank-1 projected edit disturb a BENIGN continuation?
        U1 = Us[ranks[0]]
        projected1 = T[i] - U1 @ (U1.T @ diff)
        b0 = scorer.cont_logprob(tp, BENIGN)
        with im.intervene(L, projected1.to(dtype=im_dtype(im)).to(im_device(im)),
                          positions=[Ptr - 1], mode="set"):
            util.append(b0 - scorer.cont_logprob(tp, BENIGN))

    med = statistics.median
    fd = med(full_denoise)
    print(f"  full-denoise (ceiling) = {fd:+.2f} nats")
    prof = {"full_denoise": fd, "utility_benign_drop_rank1": med(util), "per_rank": {}}
    for k in ranks:
        pk, ck = med(proj[k]), med(comp[k])
        prr, crr = med(proj_rand[k]), med(comp_rand[k])
        prof["per_rank"][k] = {"projected_removal": pk, "complement_keep": ck,
                               "projected_removal_rand": prr, "complement_keep_rand": crr,
                               "frac_of_full_removed": (pk / fd if fd else None)}
        print(f"  rank {k}: projected-removal {pk:+.2f} (rand {prr:+.2f})  "
              f"complement-keep {ck:+.2f} (rand {crr:+.2f})  [={100*pk/fd:.0f}% of full]" if fd else "")
    p1 = prof["per_rank"][ranks[0]]
    verdict = ("CAUSAL single axis (rank-1 removal ~ full, >> random, benign preserved)"
               if fd and p1["projected_removal"] > 0.6 * fd
               and p1["projected_removal"] > p1["projected_removal_rand"] + 1.0
               and abs(prof["utility_benign_drop_rank1"]) < 0.5 * fd
               else "NOT a clean causal single axis (distributed / generic)")
    print("VERDICT:", verdict)
    tag = a.tag or a.model.rstrip("/").split("/")[-1]
    prof.update({"model": a.model, "layer": L, "ranks": ranks, "verdict": verdict})
    save_phase2(f"causalaxis_{tag}_L{L}.json", prof)


def im_dtype(im):
    try:
        return next(im.model.parameters()).dtype
    except Exception:
        return torch.float32


def im_device(im):
    try:
        return next(im.model.parameters()).device
    except Exception:
        return "cpu"


if __name__ == "__main__":
    main()
