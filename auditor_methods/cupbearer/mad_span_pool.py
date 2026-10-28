"""MAD improvement 3: context-span pooling instead of decision-token-only scoring.

Standard MAD scores a single position (the decision token). On observation-triggered backdoors
(Watch-OA) the defection is computed upstream in the observation and has re-converged by the
decision token, so the single-token score goes below chance. This script scores every token in a
span (the last K prefill tokens, or the full prefill) against the clean-reference Gaussian and
pools the per-token Mahalanobis distances (max / mean / top-k mean), to recover an anomaly that
lives upstream of the decision token.

It reuses the run_mad loaders, reference, and InstrumentedModel. It fits the per-layer Gaussian on
the reference DECISION tokens (same null as MAD), then at eval time scores a SPAN per input and
pools. Baseline K=1 (max over a 1-token span) reproduces standard MAD, so the K>1 vs K=1 delta is
the clean measurement of the pooling fix.

MUST run as a GPU batch job (it does forward passes). Never run on a login node.

    python auditor_methods/cupbearer/mad_span_pool.py --model <dir> --model-id <id> \
        --bench watch_oa_runtime_v2 --data-dir <d> --reference-dir <ref> --span 64 --pool max
"""
from __future__ import annotations
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "auditor_methods"))
sys.path.insert(0, str(_HERE.parent))
import run_mad  # loaders, _load_reference, auroc_label, _fit_gaussian, _mahalanobis

# run_mad._V4_LOADERS is a function-local dict inside run_v4, so it is not importable as a module
# attribute. Mirror it here (kept in sync with run_mad.run_v4) to resolve the per-bench loader.
_V4_LOADERS = {
    "persistbd_runtime_v4": "_persistbd_runtime_v4",
    "persistbd_runtime_qwen35": "_persistbd_runtime_qwen35",
    "bald_word_runtime_v1": "_bald_word_runtime_v1",
    "bald_scene_runtime_v1": "_bald_scene_runtime_v1",
    "watch_qa_runtime_v1": "_watch_qa_runtime_v1",
    "watch_oa_runtime_v2": "_watch_oa_runtime_v2",
    "agentghost_aitz_runtime_v2": "_agentghost_aitz_runtime",
}


def _span_token_scores(model, cases, mean_l, invcov_l, nL, span, adapter=None):
    """For each case return pooled per-layer Mahalanobis over the last `span` prefill tokens.
    Returns dict pool-name -> [n, nL] matrix of pooled scores."""
    from audit import InstrumentedModel
    im = model if isinstance(model, InstrumentedModel) else InstrumentedModel(model, adapter=adapter)
    out = {"max": [], "mean": [], "top4": [], "last": []}
    with torch.inference_mode():
        for i, c in enumerate(cases):
            ids = im.encode_messages(c["messages"])
            mmk = im.mm_kwargs() if hasattr(im, "mm_kwargs") else {}
            o = im.model(input_ids=ids, output_hidden_states=True, use_cache=False, **mmk)
            hs = o.hidden_states                       # tuple (nL+1) of [1, seq, d]
            seq = hs[0].shape[1]
            k = min(span, seq)
            per_layer = {p: [] for p in out}
            for L in range(nL):
                h = hs[L + 1][0, -k:, :].float()       # [k, d] last-k tokens at layer L
                s = (run_mad._mahalanobis(h, mean_l[L], invcov_l[L]) / mean_l[L].shape[0]).cpu().numpy()
                per_layer["max"].append(float(s.max()))
                per_layer["mean"].append(float(s.mean()))
                per_layer["top4"].append(float(np.sort(s)[-min(4, k):].mean()))
                per_layer["last"].append(float(s[-1]))   # == standard MAD (decision token)
            for p in out:
                out[p].append(per_layer[p])
            del o, hs
            if i % 20 == 0:
                print(f"  span-score {i}/{len(cases)} (seq={seq}, k={k})", flush=True)
    return {p: np.asarray(v) for p, v in out.items()}   # each [n, nL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", dest="model_id", default=None)
    ap.add_argument("--bench", required=True, choices=list(_V4_LOADERS))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--reference-dir", dest="reference_dir", default=None)
    ap.add_argument("--reference-repo", dest="reference_repo", default=None)
    ap.add_argument("--reference-n", dest="reference_n", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--span", type=int, default=64, help="score the last `span` prefill tokens")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if not (a.reference_dir or a.reference_repo):
        ap.error("needs a clean reference: --reference-dir or --reference-repo")

    mod = importlib.import_module(_V4_LOADERS[a.bench])
    mid = a.model_id or mod.model_id_from_path(a.model)
    cases = mod.load_cases(mid, a.n or None, a.data_dir)
    labels = np.array([c["label"] for c in cases])
    print(f"[span] {mid} bench={a.bench} cases={len(cases)} pos={int(labels.sum())} span={a.span}", flush=True)

    from audit import InstrumentedModel
    im = InstrumentedModel(a.model, adapter=a.adapter)

    # fit per-layer Gaussian on reference DECISION tokens (same null as MAD)
    ref_msgs = run_mad._load_reference(a.reference_repo, a.reference_n, a.reference_dir)
    ref_cases = [{"messages": m} for m in ref_msgs]
    print(f"[span] reference n={len(ref_cases)}; extracting decision-token acts", flush=True)
    ref_acts, nL = run_mad.extract_case_acts(im, ref_cases)   # [n_ref, nL, d], nL
    mean_l, invcov_l = [], []
    for L in range(nL):
        m, inv, _ = run_mad._fit_gaussian(ref_acts[:, L])
        mean_l.append(m); invcov_l.append(inv)

    pooled = _span_token_scores(im, cases, mean_l, invcov_l, nL, a.span)

    Lfix = int(round(a.frac * nL))
    results = {}
    for pool, mat in pooled.items():                 # mat [n, nL]
        per_layer = [run_mad.auroc_label(mat[:, L], labels) for L in range(nL)]
        best = int(np.nanargmax(per_layer)) if np.isfinite(np.nanmax(per_layer)) else Lfix
        results[pool] = {
            "auroc_fixed": float(per_layer[Lfix]), "auroc_best": float(np.nanmax(per_layer)),
            "best_layer": best, "auroc_per_layer": [float(x) for x in per_layer],
        }
        print(f"[span] pool={pool:5s} fixed(L{Lfix})={per_layer[Lfix]:.3f} "
              f"best(L{best})={np.nanmax(per_layer):.3f}", flush=True)

    out = {"model_id": mid, "bench": a.bench, "span": a.span, "n_layers": nL,
           "n_cases": len(cases), "n_positive": int(labels.sum()),
           "fixed_layer": Lfix, "results": results,
           "note": "pool=last reproduces decision-token MAD; max/mean/top4 are the span-pooling fix"}
    run = os.environ.get("DTAI_RUN_DIR") or str(_HERE.parent / "results")
    os.makedirs(run, exist_ok=True)
    p = os.path.join(run, f"madspan_{a.tag or mid}.json")
    json.dump(out, open(p, "w"), indent=2)
    print("saved", p)


if __name__ == "__main__":
    main()
