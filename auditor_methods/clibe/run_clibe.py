"""CLIBE (dynamic-backdoor detector) adapted to our agentic Qwen models.

Faithful adaptation of CLIBE's generative detection (Zeng et al., NDSS 2025) — their
perturb-utils cover only GPT2/Neo/NeoX/OPT, so we reimplement the *method* for Qwen2.5:

  (1) Few-shot perturbation injection: add a budget-limited weight perturbation delta to
      the Q/K/V projections of one attention layer L, optimized so a FEW benign reference
      prefixes exhibit the TARGET behaviour (here: emit the malicious continuation).
  (2) Generalization check: with the perturbed weights, measure the margin
      m(x) = logp(target_cont | x) - logp(benign_cont | x) on HELD-OUT reference prefixes.
  (3) A dynamic backdoor is flagged when the few-shot perturbation GENERALIZES — the
      held-out margins are large and CONCENTRATED (low entropy). A benign model resists:
      the same perturbation does not generalize.

The behaviour/target is supplied as text (no trigger, no poisoned data). Compare the
backdoored model's generalization against a clean reference to calibrate.

  python auditor_methods/clibe/run_clibe.py --model swe-audit-3b-02 --bench persistbd \
      --layer 14 --few 6 --test 24 --budget 0.08 --steps 60
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


def load_bench(name, n):
    """Return list of (prefix_messages, target_cont_text, benign_cont_text)."""
    if name == "persistbd":
        from experiments.gac.persistbd import load_rows, BENIGN
        rows = load_rows(n)
        # reference prefixes = NEUTRAL (non-triggered) prefixes; target behaviour = the exfil payload
        return [(r["neutral_msgs"], r["payload"], BENIGN) for r in rows]
    raise ValueError(name)


def cont_logp(model, tok, device, prefix_ids, cont_ids):
    """log p(cont | prefix) under the (possibly perturbed) model."""
    ids = torch.cat([prefix_ids, cont_ids], 1)
    out = model(input_ids=ids, use_cache=False).logits
    logits = out[0, prefix_ids.shape[1] - 1: -1]        # predict each cont token
    lp = F.log_softmax(logits.float(), -1)
    return lp[torch.arange(cont_ids.shape[1]), cont_ids[0]].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="persistbd")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--layer", type=int, default=None, help="attn layer to perturb (default n_layers//2)")
    ap.add_argument("--few", type=int, default=6, help="few-shot prefixes for injection")
    ap.add_argument("--test", type=int, default=24, help="held-out prefixes for generalization")
    ap.add_argument("--budget", type=float, default=0.08, help="relative L2 budget on delta per column")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    from audit import InstrumentedModel
    im = InstrumentedModel(a.model, dtype=torch.float32)     # fp32 for stable perturbation opt
    model, tok, device = im.model, im.tokenizer, im.device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    L = a.layer if a.layer is not None else im.n_layers // 2
    attn = model.model.layers[L].self_attn
    projs = {"q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj}
    Wbase = {k: v.weight.detach().clone() for k, v in projs.items()}
    deltas = {k: torch.zeros_like(Wbase[k], requires_grad=True) for k in projs}

    # monkey-patch each projection forward to use (W_base + delta)
    def patch(lin, k):
        def fwd(x):
            return F.linear(x, Wbase[k] + deltas[k], lin.bias)
        lin.forward = fwd
    for k, lin in projs.items():
        patch(lin, k)

    data = load_bench(a.bench, a.n)
    enc = lambda msgs: im.encode_messages(msgs)
    cids = lambda text: tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    few = [(enc(m), cids(t), cids(b)) for m, t, b in data[: a.few]]
    test = [(enc(m), cids(t), cids(b)) for m, t, b in data[a.few: a.few + a.test]]
    print(f"[clibe] {a.model} layer={L} few={len(few)} test={len(test)} budget={a.budget}", flush=True)

    # ---- (1) few-shot perturbation injection: make the few prefixes emit the target ----
    opt = torch.optim.Adam(list(deltas.values()), lr=a.lr)
    budget = a.budget
    for step in range(a.steps):
        opt.zero_grad()
        loss = 0.0
        for pre, tgt, _ in few:
            loss = loss - cont_logp(model, tok, device, pre, tgt)   # maximize logp(target)
        loss = loss / len(few)
        loss.backward()
        opt.step()
        with torch.no_grad():                                        # project to the L2 budget (per column)
            for k in deltas:
                col = deltas[k].norm(dim=0, keepdim=True)
                cap = budget * Wbase[k].norm(dim=0, keepdim=True)
                scale = torch.clamp(cap / (col + 1e-8), max=1.0)
                deltas[k].mul_(scale)
        if step % 15 == 0:
            print(f"  step {step} inj_loss(-logp target)={loss.item():.3f}", flush=True)

    # ---- (2) generalization check on held-out prefixes ----
    margins = []
    with torch.no_grad():
        for pre, tgt, ben in test:
            m = (cont_logp(model, tok, device, pre, tgt) - cont_logp(model, tok, device, pre, ben)).item()
            margins.append(m)

    # ---- (3) metric: concentration (low entropy) + fraction prefer-target ----
    import numpy as np
    mar = np.array(margins)
    frac_pos = float((mar > 0).mean())
    # discrete entropy of margins over a fixed range (CLIBE's get_hist_entropy analog)
    hist, _ = np.histogram(mar, bins=10, range=(-20, 20))
    p = hist / max(hist.sum(), 1)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    print(f"\n[clibe] {a.model}  generalization: frac_prefer_target={frac_pos:.2f}  "
          f"mean_margin={mar.mean():.2f}  entropy={entropy:.3f}")
    print("  (backdoor => perturbation GENERALIZES => high frac_pos / low entropy)")

    out = {"model": a.model, "bench": a.bench, "layer": L, "few": a.few, "test": a.test,
           "budget": a.budget, "steps": a.steps, "frac_prefer_target": frac_pos,
           "mean_margin": float(mar.mean()), "entropy": entropy, "margins": margins}
    run = os.environ.get("DTAI_RUN_DIR") or os.environ.get("DELTA_RUN_DIR") \
        or str(Path(__file__).parent / "results")
    os.makedirs(run, exist_ok=True)
    tag = a.tag or a.model.rstrip("/").split("/")[-1]
    p_out = os.path.join(run, f"clibe_{tag}.json")
    json.dump(out, open(p_out, "w"), indent=2)
    print("saved", p_out)


if __name__ == "__main__":
    main()
