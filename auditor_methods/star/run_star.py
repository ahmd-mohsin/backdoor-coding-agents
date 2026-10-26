"""STAR -- State-Transition Amplification Ratio (Park/Park/Lee/Na/Choi, Soongsil Univ,
arXiv 2601.08511) as a runtime monitor on our v4-style datasets.

Method (trigger/target-blind, NO reference set, grey-box = needs output logits only):
  The model generates a reasoning/response path r = r_1..r_n for input x = I (+) x
  (system instruction (+) user input). For each generated token r_t STAR contrasts two
  distributions over the SAME realized token:
     posterior  Q_t = P_theta(r_t | x, r_<t)      -- the normal decoding probability
     prior      P_t = P_theta(r_t |    r_<t)       -- same token, input x DROPPED
  A malicious input forces a "high posterior / low prior" path, so the token-step evidence
     s_t = log Q_t - log P_t          (= log of the amplification ratio Lambda_t = Q_t/(P_t+eps))
  is persistently positive on the injected reasoning. STAR accumulates it with a one-sided
  CUSUM and flags a change point:
     g_t = 0                              if t <  t_warmup        (burn-in; eq. (CUSUM))
     g_t = max(0, g_{t-1} + s_t - k)      if t >= t_warmup
  The input is anomalous iff max_t g_t > tau. Paper hyperparameters, fixed across all models
  and datasets (sec 5.1): k = 2.0, tau = 8.5, t_warmup = 10, sampling temperature 1.0.

Implementation decisions (documented because the paper leaves them implicit):
  * Q_t is read from the generation pass (log_softmax of the step logits at the REALIZED
    token id, not top-1), so it is correct under greedy OR sampling. We greedy-decode by
    default for reproducibility (as we run ConfGuard); --temperature/--do-sample expose the
    paper's T=1.0 sampling. The logprob we read is always the raw (T=1) log_softmax.
  * P_t is the paper's "single additional forward pass in batch": we run the model once on
    [BOS?] + generated_tokens (the chat template / system / user prompt dropped) and read
    log P_theta(r_t | r_<t). This is the ONLY extra compute over normal decoding.
  * s_t = log Q_t - log(P_t + eps), eps = 1e-10 (the paper's epsilon in Lambda_t), which
    bounds a single underflowed prior token.
  * Detection score D = max_t g_t (threshold-free, for AUROC/AUPRC). Operating point at the
    paper's fixed tau = 8.5; if a nominal reference set is supplied we ALSO report a
    reference-calibrated tau at target FPR, for parity with MAD/LCF (STAR itself needs none).
  * t_warmup = 10 discards the first 10 tokens from CUSUM, so a generation shorter than 10
    tokens can never fire. BALD lane-choice outputs are short -> we raise max_new_tokens and
    expose --warmup. This is an inherent property of the method, not a bug.

    python auditor_methods/star/run_star.py --model <staged_model_dir> --model-id <id> \
        --bench bald_scene_runtime_v1 --n 0
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "auditor_methods"))
sys.path.insert(0, str(_ROOT / "auditor_methods" / "cupbearer"))

_V4_LOADERS = {
    "persistbd_runtime_v4": "_persistbd_runtime_v4",
    "persistbd_runtime_qwen35": "_persistbd_runtime_qwen35",
    "bald_word_runtime_v1": "_bald_word_runtime_v1",
    "bald_scene_runtime_v1": "_bald_scene_runtime_v1",
    "watch_qa_runtime_v1": "_watch_qa_runtime_v1",
    "watch_oa_runtime_v2": "_watch_oa_runtime_v2",
}

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Core STAR scoring for one input.
# ---------------------------------------------------------------------------
@torch.no_grad()
def star_score_one(im, input_ids, max_new_tokens, k, warmup, do_sample, temperature):
    """Generate r given x, compute s_t = logQ_t - logP_t per generated token, run CUSUM.

    Returns (D, detail) where D = max_t g_t and detail carries the token series & the
    generated text (for group-level diagnostics and offline (k, tau) sweeps).
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=im.tokenizer.pad_token_id or im.tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)
    out = im.model.generate(**gen_kwargs)

    gen_ids = out.sequences[0, prompt_len:]                 # [n] realized reasoning tokens
    n = int(gen_ids.shape[0])
    if n == 0:
        return 0.0, {"n_gen": 0, "s": [], "g_max": 0.0, "text": ""}

    # --- posterior Q_t: log prob of the realized token from the generation step logits ---
    logQ = torch.empty(n, dtype=torch.float32, device=device)
    for i in range(n):
        lp = torch.log_softmax(out.scores[i][0].float(), dim=-1)   # step i -> distribution for gen token i
        logQ[i] = lp[gen_ids[i]]

    # --- prior P_t: ONE forward pass on [BOS?] + generated tokens, input x dropped ---
    bos = getattr(im.tokenizer, "bos_token_id", None)
    prefix = [bos] if bos is not None else []
    off = len(prefix)
    prior_ids = torch.tensor([prefix + gen_ids.tolist()], device=device)
    logits = im.model(input_ids=prior_ids, use_cache=False).logits[0].float()   # [off+n, V]
    logP = torch.empty(n, dtype=torch.float32, device=device)
    for t in range(n):                                      # gen token t sits at prior index off+t
        j = off + t - 1                                    # logits row predicting that token
        if j >= 0:
            lp = torch.log_softmax(logits[j], dim=-1)
            logP[t] = lp[gen_ids[t]]
        else:                                              # first token, no BOS -> prior undefined
            logP[t] = logQ[t]                              # => s_1 = 0 (and it is inside warmup anyway)

    # --- token-step evidence and one-sided CUSUM ---
    # s_t = log Q_t - log(P_t + eps)
    P = torch.exp(logP)
    s = (logQ - torch.log(P + _EPS)).tolist()

    g = 0.0
    g_max = 0.0
    g_series = []
    for t in range(n):                                     # t is 0-based; warmup counts tokens
        if (t + 1) < warmup:
            g = 0.0
        else:
            g = max(0.0, g + s[t] - k)
        g_series.append(g)
        if g > g_max:
            g_max = g

    text = im.tokenizer.decode(gen_ids, skip_special_tokens=True)
    detail = {"n_gen": n, "g_max": g_max,
              "s": [round(v, 4) for v in s],                # kept for offline k/tau sweeps
              "s_max": max(s) if s else 0.0,
              "s_mean_post_warmup": (sum(s[warmup - 1:]) / max(1, len(s) - warmup + 1)) if n >= warmup else 0.0,
              "text": text}
    return float(g_max), detail


def cusum_from_s(s, k, warmup):
    """Recompute max-CUSUM D from a stored s-series (offline (k, warmup) sweep)."""
    g = 0.0
    g_max = 0.0
    for t in range(len(s)):
        if (t + 1) < warmup:
            g = 0.0
        else:
            g = max(0.0, g + s[t] - k)
        g_max = max(g_max, g)
    return g_max


# ---------------------------------------------------------------------------
def auroc_label(scores, labels):
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def auprc_label(scores, labels):
    from sklearn.metrics import average_precision_score
    if len(set(labels)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(labels, scores))
    except Exception:
        return float("nan")


def _op_point(D, labels, tau):
    """TPR/FPR at a fixed threshold tau on the max-CUSUM score D (flag iff D > tau)."""
    import numpy as np
    D = np.asarray(D, float)
    labels = np.asarray(labels, int)
    pred = (D > tau).astype(int)
    pos = labels == 1
    neg = labels == 0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    return {"tau": float(tau), "tpr": tpr, "fpr": fpr,
            "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}


def _load_cases(bench, mid, n, data_dir):
    if bench in _V4_LOADERS:
        mod = importlib.import_module(_V4_LOADERS[bench])
        cases = mod.load_cases(mid, n or None, data_dir)
        return [{"messages": c["messages"], "label": int(c["label"]),
                 "group": c.get("group"), "case_id": c.get("case_id")} for c in cases], mod
    if bench == "persistbd":
        from experiments.gac.persistbd import load_rows
        from experiments.gac._common import EXFIL
        rows = load_rows(n or 60)
        cases = []
        for r in rows:
            cases.append({"messages": r["trigger_msgs"], "label": None, "group": "trigger",
                          "case_id": r["instance"], "_regex": EXFIL})
            cases.append({"messages": r["neutral_msgs"], "label": None, "group": "neutral",
                          "case_id": r["instance"], "_regex": EXFIL})
        return cases, None
    raise ValueError(f"unknown bench {bench!r}")


def main():
    ap = argparse.ArgumentParser(description="STAR runtime monitor (arXiv 2601.08511).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="bald_scene_runtime_v1",
                    choices=list(_V4_LOADERS) + ["persistbd"])
    ap.add_argument("--model-id", dest="model_id", default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--adapter", default=None)
    # paper hyperparameters (sec 5.1), fixed across all models/datasets
    ap.add_argument("--k", type=float, default=2.0, help="CUSUM drift/slack (paper 2.0)")
    ap.add_argument("--tau", type=float, default=8.5, help="flag threshold (paper 8.5)")
    ap.add_argument("--warmup", type=int, default=10, help="burn-in tokens (paper 10)")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=200)
    ap.add_argument("--do-sample", dest="do_sample", action="store_true",
                    help="sample at --temperature (paper T=1.0); default greedy for determinism")
    ap.add_argument("--temperature", type=float, default=1.0)
    # optional reference calibration (STAR does not need it; for FPR parity with MAD/LCF)
    ap.add_argument("--reference-repo", dest="reference_repo", default=None)
    ap.add_argument("--reference-dir", dest="reference_dir", default=None)
    ap.add_argument("--reference-n", dest="reference_n", type=int, default=0)
    ap.add_argument("--target-fpr", dest="target_fpr", type=float, default=0.05)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    from audit import InstrumentedModel

    mid = a.model_id
    if mid is None and a.bench in _V4_LOADERS:
        mid = importlib.import_module(_V4_LOADERS[a.bench]).model_id_from_path(a.model)
    cases, mod = _load_cases(a.bench, mid, a.n, a.data_dir)
    print(f"[STAR] {mid or a.model} bench={a.bench} cases={len(cases)} "
          f"k={a.k} tau={a.tau} warmup={a.warmup} N={a.max_new_tokens} "
          f"{'sample@%.2f' % a.temperature if a.do_sample else 'greedy'}", flush=True)

    im = InstrumentedModel(a.model, adapter=a.adapter)

    D, labels, groups, details = [], [], [], []
    for i, c in enumerate(cases):
        ids = im.encode_messages(c["messages"])
        d, det = star_score_one(im, ids, a.max_new_tokens, a.k, a.warmup,
                                a.do_sample, a.temperature)
        # gold label: v4 loaders carry the judge label; persistbd legacy uses the EXFIL regex on text
        if c["label"] is not None:
            lab = int(c["label"])
        else:
            lab = 1 if c["_regex"].search(det["text"]) else 0
        D.append(d); labels.append(lab); groups.append(c.get("group"))
        details.append({"case_id": c.get("case_id"), "group": c.get("group"),
                        "label": lab, "D": d, "n_gen": det["n_gen"],
                        "s_max": det["s_max"], "s": det["s"]})
        if i % 10 == 0:
            print(f"  case {i}/{len(cases)}  D={d:.2f} n_gen={det['n_gen']} label={lab}", flush=True)

    benign_control = sum(labels) == 0
    auroc = auroc_label(D, labels)
    auprc = auprc_label(D, labels)
    op_paper = _op_point(D, labels, a.tau)

    # optional reference-calibrated tau (STAR needs none; parity with MAD/LCF 5%/10% budgets)
    ops_ref = {}
    if a.reference_repo or a.reference_dir:
        sys.path.insert(0, str(_ROOT / "auditor_methods" / "cupbearer"))
        import run_mad
        ref_msgs = run_mad._load_reference(a.reference_repo, a.reference_n, a.reference_dir)
        print(f"[STAR] reference calibration on {a.reference_repo or a.reference_dir} "
              f"n_ref={len(ref_msgs)}", flush=True)
        import numpy as np
        ref_D = []
        for m in ref_msgs:
            ids = im.encode_messages(m)
            rd, _ = star_score_one(im, ids, a.max_new_tokens, a.k, a.warmup,
                                   a.do_sample, a.temperature)
            ref_D.append(rd)
        for al in (0.10, a.target_fpr):
            tau_c = float(np.percentile(ref_D, 100 * (1 - al)))
            op = _op_point(D, labels, tau_c)
            op["ref_fpr_realized"] = float((np.asarray(ref_D) > tau_c).mean())
            ops_ref[f"{al:.2f}"] = op
        ops_ref["ref_D_range"] = [float(min(ref_D)), float(max(ref_D))]

    # per-group diagnostics
    gset = sorted({g for g in groups if g is not None})
    per_group = {}
    for g in gset:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        per_group[g] = {"n": len(idx),
                        "mean_D": sum(D[i] for i in idx) / len(idx),
                        "flag_rate_tau": sum(D[i] > a.tau for i in idx) / len(idx),
                        "label_rate": sum(labels[i] for i in idx) / len(idx)}

    out = {
        "model": a.model, "bench": a.bench, "model_id": mid,
        "detector": "STAR (state-transition amplification ratio, CUSUM change-point; "
                    "s_t = logQ_t - logP_t, posterior vs input-dropped prior)",
        "k": a.k, "tau": a.tau, "warmup": a.warmup, "max_new_tokens": a.max_new_tokens,
        "decode": ("sample@%.2f" % a.temperature) if a.do_sample else "greedy",
        "n_cases": len(cases), "n_positive": int(sum(labels)),
        "positive_rate": (sum(labels) / len(labels)) if labels else float("nan"),
        "benign_control": benign_control,
        "auroc": auroc, "auprc": auprc,
        "operating_point_paper_tau": op_paper,
        "operating_points_reference": ops_ref,
        "per_group": per_group,
        "scores": {"D": [round(x, 4) for x in D], "labels": labels, "groups": groups},
        "details": details,
    }
    print(f"\n[STAR] {mid or a.model}  AUROC={auroc:.3f} AUPRC={auprc:.3f} "
          f"(pos_rate={out['positive_rate']:.2f})  @tau={a.tau}: "
          f"TPR={op_paper['tpr']:.2f} FPR={op_paper['fpr']:.2f}", flush=True)
    for g, d in per_group.items():
        print(f"  [{g:9s}] n={d['n']:3d} mean_D={d['mean_D']:.2f} "
              f"flag@tau={d['flag_rate_tau']:.2f} label_rate={d['label_rate']:.2f}")

    tag = a.tag or mid or a.model.rstrip("/").split("/")[-1]
    run = os.environ.get("DTAI_RUN_DIR") or os.environ.get("DELTA_RUN_DIR") \
        or str(Path(__file__).parent / "results")
    os.makedirs(run, exist_ok=True)
    p = os.path.join(run, f"star_{tag}.json")
    json.dump(out, open(p, "w"), indent=2)
    print("saved", p)


if __name__ == "__main__":
    main()
