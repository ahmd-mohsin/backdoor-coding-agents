"""E1 — dual-use reversibility: is benign defense a removable steering vector?

Fit the model's OWN recognition direction u = mean(trig decision-token resid) -
mean(neutral resid) at layer L (unit-normalized). Then, over an alpha sweep, via free
GREEDY generation (intercept-only; the emitted command is never executed):

  - RE-ACTIVATE: add +alpha*u at L to CLEAN prefixes -> does the (suppressed) payload fire
    WITHOUT the trigger present? If yes on a *defended* model, benign defense is cosmetic.
  - SUPPRESS:    add -alpha*u at L to TRIGGER prefixes -> does the attack stop? A free
    runtime defense by activation steering.

    python -m experiments.phase2.steer_reverse --model swe-audit-3b-01 --layer 20 --n 20
"""

from __future__ import annotations

import argparse

import torch

from ._common import load_pairs, save_phase2
from experiments.gac._common import EXFIL


def main():
    from audit import InstrumentedModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None, help="organism eval dir (else PersistBD)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alphas", default="0,4,8,16,24")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()

    im = InstrumentedModel(a.model)
    L = a.layer
    alphas = [float(x) for x in a.alphas.split(",")]
    pairs = load_pairs(a.n, a.data)

    # recognition direction at L (final-token diff-of-means), fit on this model itself
    tr, ne = [], []
    for p in pairs:
        tr.append(im.capture(im.encode_messages(p["trig"]), [L]).hidden[L][-1])
        ne.append(im.capture(im.encode_messages(p["clean"]), [L]).hidden[L][-1])
    u = torch.stack(tr).mean(0) - torch.stack(ne).mean(0)
    u = u / u.norm()

    clean_ids = [im.encode_messages(p["clean"]) for p in pairs]
    trig_ids = [im.encode_messages(p["trig"]) for p in pairs]

    def exfil_rate(prefixes, vec):
        c = 0
        for pid in prefixes:
            if vec is None:
                g = im.generate(pid, max_new_tokens=a.max_new, greedy=True)
            else:
                with im.intervene(L, vec, positions=None, mode="add"):
                    g = im.generate(pid, max_new_tokens=a.max_new, greedy=True)
            c += bool(EXFIL.search(g))
        return c / len(prefixes)

    react = {"base_clean_FPR": exfil_rate(clean_ids, None)}   # attack rate on clean, no steer
    supp = {"base_trig_TPR": exfil_rate(trig_ids, None)}      # attack rate on trigger, no steer
    for al in alphas:
        if al == 0:
            continue
        react[al] = exfil_rate(clean_ids, (al * u).to(im.device, im._dtype()))     # +u on clean
        supp[al] = exfil_rate(trig_ids, (-al * u).to(im.device, im._dtype()))      # -u on trigger
    print(f"[steer] {a.model} L{L}  RE-ACTIVATE (clean + a*u): {react}")
    print(f"[steer] {a.model} L{L}  SUPPRESS  (trig - a*u): {supp}")
    save_phase2(f"steer_{a.model.rstrip('/').split('/')[-1]}_L{L}.json",
                {"model": a.model, "layer": L, "n": len(pairs), "alphas": alphas,
                 "reactivate": react, "suppress": supp})


if __name__ == "__main__":
    main()
