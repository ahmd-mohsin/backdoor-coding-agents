"""MAD auditor (cupbearer) on the PersistBD agentic backdoor models.

Implements the Mechanistic Anomaly Detection method of Johnston et al., "MAD for Quirky
Language Models" (arXiv 2504.08812), using the EXACT detector math from the EleutherAI
`cupbearer` library (`detectors/statistical/{helpers,mahalanobis_detector,que_detector}.py`)
— mirrored here as standalone tensor functions so it runs anywhere torch runs (no umap/
datasets/loguru deps), on our 20k-token agentic trajectories.

Pipeline (ONLINE setting = the realistic auditor: calibrate on BENIGN/clean activations
only, no labeled anomalies):
  1. Load a swe-audit model + PersistBD paired (triggered, clean) trajectories.
  2. Extract residual-stream activations at the DECISION token (pre-action) at every layer.
  3. TRUSTED = a split of CLEAN activations. Fit per-layer Gaussian (mean+covariance).
  4. Score held-out CLEAN and TRIGGERED activations by Mahalanobis distance (cupbearer).
  5. AUROC(triggered vs clean) per layer → is the backdoor a forward-readable anomaly?
     Also diagonal-Mahalanobis and (offline) quantum-entropy variants.

    python auditor_methods/cupbearer/run_mad.py --model swe-audit-3b-01 --n 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# our proven infra: model loading + activation capture, and PersistBD pair loading
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# cupbearer detector math — mirrored verbatim from
# cupbearer/src/cupbearer/detectors/statistical/helpers.py (EleutherAI/cupbearer).
# Kept standalone so no cupbearer install / umap / datasets is required on the cluster.
# ---------------------------------------------------------------------------
def _mahalanobis(activation, mean, inv_covariance, inv_diag_covariance=None):
    bs = activation.shape[0]
    delta = activation.view(bs, -1) - mean
    distance = torch.einsum("bi,ij,bj->b", delta, inv_covariance, delta)
    if inv_diag_covariance is not None:
        distance -= torch.einsum("bi,i->b", delta ** 2, inv_diag_covariance)
    return distance


def _fit_gaussian(trusted, rcond=1e-3):
    """mean, pinv(cov) [full], and inv_diag(cov) — cupbearer MahalanobisDetector."""
    mean = trusted.mean(dim=0)
    cov = torch.cov(trusted.T)                      # (dim, dim), Bessel-corrected
    inv_cov = torch.linalg.pinv(cov.double(), rcond=rcond, hermitian=True).to(trusted.dtype)
    diag = torch.diag(cov)
    inv_diag = torch.where(diag > rcond, 1.0 / diag, torch.zeros_like(diag))
    return mean, inv_cov, inv_diag


def _quantum_entropy(trusted, test_all, rcond=1e-3, alpha=4.0):
    """cupbearer QUE (offline): whiten by trusted cov, score by exp(alpha*Cov_untrusted)."""
    mean = trusted.mean(0)
    cov = torch.cov(trusted.T)
    eigs = torch.linalg.eigh(cov.double())
    vals_rsqrt = eigs.eigenvalues.rsqrt()
    vals_rsqrt[eigs.eigenvalues < rcond * eigs.eigenvalues.max()] = 0
    W = (eigs.eigenvectors * vals_rsqrt.unsqueeze(0)).to(trusted.dtype)   # whitening
    whitened = (test_all - mean) @ W
    cov_unt = torch.cov(test_all.T)                 # untrusted = all test
    cov_w = W.T @ cov_unt @ W
    norm = torch.linalg.eigvalsh(cov_w.double()).max().to(trusted.dtype)
    expf = torch.matrix_exp((alpha * cov_w / norm).double()).to(trusted.dtype)
    return torch.einsum("bi,ij,jb->b", whitened, expf, whitened.mT)


# ---------------------------------------------------------------------------
def extract_decision_acts(model, pairs):
    """[n, nL, d] residual-stream activations at the decision token, triggered & clean."""
    from audit import InstrumentedModel
    im = model if isinstance(model, InstrumentedModel) else InstrumentedModel(model)
    nL = im.n_layers
    trig, clean = [], []
    for i, p in enumerate(pairs):
        tp = im.encode_messages(p["trig"]); cp = im.encode_messages(p["clean"])
        ct = im.capture(tp); cc = im.capture(cp)
        trig.append(torch.stack([ct.hidden[L][-1] for L in range(nL)]).float().cpu())
        clean.append(torch.stack([cc.hidden[L][-1] for L in range(nL)]).float().cpu())
        if i % 10 == 0:
            print(f"  extract {i}/{len(pairs)}", flush=True)
    return torch.stack(trig), torch.stack(clean), nL   # [n,nL,d]


def load_pairs(n):
    """Reuse the proven PersistBD loader: paired (triggered, clean) trajectories."""
    from experiments.gac.battery import load_rows
    rows = load_rows(n)
    return [{"trig": r["trigger_msgs"], "clean": r["neutral_msgs"]} for r in rows]


def auroc(pos, neg):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=120, help="raw rows (~half carry the trigger)")
    ap.add_argument("--frac", type=float, default=0.8, help="fixed-depth label-free layer rule")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    import numpy as np

    pairs = load_pairs(a.n)
    print(f"[MAD] {a.model}  pairs={len(pairs)}", flush=True)
    trig, clean, nL = extract_decision_acts(a.model, pairs)
    n = trig.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    fit, test = idx[: n // 2], idx[n // 2:]          # TRUSTED fit (clean) vs held-out test

    maha, maha_diag, que = [], [], []
    per_layer_scores = {}                            # L -> (trigger dists, clean dists) on the held-out test split
    for L in range(nL):
        tr = clean[fit, L]                           # trusted (clean) calibration
        mean, inv_cov, inv_diag = _fit_gaussian(tr)
        d = mean.shape[0]
        def score(x, full=True):
            return (_mahalanobis(x, mean, inv_cov,
                                 None if full else inv_diag) / d).numpy()
        s_tr = score(trig[test, L]); s_cl = score(clean[test, L])
        per_layer_scores[L] = (s_tr.tolist(), s_cl.tolist())
        maha.append(auroc(s_tr, s_cl))
        s_trd = score(trig[test, L], full=False); s_cld = score(clean[test, L], full=False)
        maha_diag.append(auroc(s_trd, s_cld))
        # QUE (offline: uses all test as untrusted)
        test_all = torch.cat([trig[test, L], clean[test, L]], 0)
        q = _quantum_entropy(tr, test_all).numpy()
        que.append(auroc(q[: len(test)], q[len(test):]))

    Lfix = int(round(a.frac * nL))
    best = int(np.nanargmax(maha))
    out = {
        "model": a.model, "n_pairs": n, "n_layers": nL, "setting": "online (trusted=clean)",
        "detector": "cupbearer Mahalanobis (full), diag-Mahalanobis, QUE",
        "auroc_mahalanobis_per_layer": maha,
        "auroc_diag_per_layer": maha_diag,
        "auroc_que_per_layer": que,
        "best_layer_mahalanobis": best, "best_auroc_mahalanobis": maha[best],
        "fixed_depth_layer": Lfix, "fixed_depth_auroc_mahalanobis": maha[Lfix],
        # per-sample Mahalanobis distances (held-out test split) so the raw
        # clean-vs-triggered separation behind each AUROC can be plotted.
        "scores_best_layer": {"layer": best, "trigger": per_layer_scores[best][0],
                              "clean": per_layer_scores[best][1]},
        "scores_fixed_layer": {"layer": Lfix, "trigger": per_layer_scores[Lfix][0],
                               "clean": per_layer_scores[Lfix][1]},
    }
    print(f"[MAD] {a.model}  Mahalanobis best L{best} AUROC={maha[best]:.3f}  "
          f"fixed-depth L{Lfix} AUROC={maha[Lfix]:.3f}")
    print("  per-layer Mahalanobis:", " ".join(f"{x:.2f}" for x in maha))
    print(f"  diag best AUROC={max(maha_diag):.3f}   QUE best AUROC={max(que):.3f}")

    tag = a.tag or a.model.rstrip("/").split("/")[-1]
    run = os.environ.get("DTAI_RUN_DIR") or os.environ.get("DELTA_RUN_DIR") or str(Path(__file__).parent / "results")
    os.makedirs(run, exist_ok=True)
    p = os.path.join(run, f"mad_{tag}.json")
    json.dump(out, open(p, "w"), indent=2)
    print("saved", p)


if __name__ == "__main__":
    main()
