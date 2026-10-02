"""P2-0 (forward readability) — extract the decision-point residual stream.

For each matched (triggered, clean) prefix pair, capture the residual stream at EVERY
layer at the FINAL prefix token (the pre-action decision point). Saves two [n, n_layers,
d] tensors (triggered vs clean). A cheap CPU probe (probe.py) then fits a per-layer
linear classifier triggered-vs-clean: if it separates them forward-only (no gradient, no
payload), the backdoor signature is READABLE at the decision point -> a cheap forward
monitor is possible, which is the make-or-break gate for Phase 2.

    python -m experiments.phase2.extract_acts --model swe-audit-3b-02 --n 60
    python -m experiments.phase2.extract_acts --model /work/.../models/organisms/semantic \
        --data /work/.../data/organisms/semantic --n 60
"""

from __future__ import annotations

import argparse

import torch

from ._common import load_pairs, save_phase2


def main():
    from audit import InstrumentedModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None, help="organism eval dir (else PersistBD)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tag", default=None, help="output filename tag (defaults to model name)")
    a = ap.parse_args()

    im = InstrumentedModel(a.model)
    nL = im.n_layers
    pairs = load_pairs(a.n, a.data)
    trig, clean, insts = [], [], []
    print(f"[extract] {a.model}  layers={nL}  pairs={len(pairs)}")
    for i, p in enumerate(pairs):
        tp = im.encode_messages(p["trig"])
        cp = im.encode_messages(p["clean"])
        ct = im.capture(tp)
        cc = im.capture(cp)
        trig.append(torch.stack([ct.hidden[L][-1] for L in range(nL)]))    # [nL, d] final token
        clean.append(torch.stack([cc.hidden[L][-1] for L in range(nL)]))
        insts.append(p["instance"])
        if i % 10 == 0:
            print(f"  {i}/{len(pairs)}")
    tag = a.tag or a.model.rstrip("/").split("/")[-1]
    save_phase2(f"acts_{tag}.pt", {
        "model": a.model, "n_layers": nL, "n": len(pairs),
        "trig_final": torch.stack(trig),      # [n, nL, d]
        "clean_final": torch.stack(clean),    # [n, nL, d]
        "instances": insts})
    print("[extract] done")


if __name__ == "__main__":
    main()
