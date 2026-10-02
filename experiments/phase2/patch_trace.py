"""P2-0/P2-1 (causal site recovery) — bidirectional decision-token activation patching.

At the pre-action decision token, for each layer L:
  - DENOISE: replace the triggered run's residual at L with the CLEAN run's residual, and
    measure how much the payload preference drops (phi0 - phi_patched). A large drop means
    the decision-token residual at L CARRIES the backdoor effect -> a forward monitor could
    read it there (necessity at that site).
  - NOISE: inject the triggered residual into the clean run and measure how much payload
    preference it adds (sufficiency at that site).

Position-aligned (both runs have a final prefix token), so no token-count mismatch. The
per-layer denoise/noise profile is exactly the sites Phase 2's monitor would read.

    python -m experiments.phase2.patch_trace --model swe-audit-3b-02 --layers 0-11 --n 40
"""

from __future__ import annotations

import argparse
import statistics

from ._common import load_pairs, parse_layers, save_phase2


def main():
    from audit import InstrumentedModel, ObligationScorer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None, help="organism eval dir (else PersistBD)")
    ap.add_argument("--layers", default="all", help="'all' or 'a-b' (inclusive) layer band")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    im = InstrumentedModel(a.model)
    scorer = ObligationScorer(im)
    layers = parse_layers(a.layers, im.n_layers)
    pairs = load_pairs(a.n, a.data)
    print(f"[patch] {a.model}  layers={a.layers}->{layers[0]}..{layers[-1]}  pairs={len(pairs)}")

    denoise = {L: [] for L in layers}
    noise = {L: [] for L in layers}
    phi0s, phicl = [], []
    for i, p in enumerate(pairs):
        tp = im.encode_messages(p["trig"])
        cp = im.encode_messages(p["clean"])
        pay = p["payload"]
        phi0 = scorer.cont_logprob(tp, pay)
        phi_clean = scorer.cont_logprob(cp, pay)
        phi0s.append(phi0); phicl.append(phi_clean)
        cc = im.capture(cp)                       # clean residuals (all layers)
        ct = im.capture(tp)                       # triggered residuals
        Ptr, Pcl = tp.shape[1], cp.shape[1]
        for L in layers:
            cvec = cc.hidden[L][-1]               # clean final-token residual [d]
            with im.intervene(L, cvec, positions=[Ptr - 1], mode="set"):
                denoise[L].append(phi0 - scorer.cont_logprob(tp, pay))
            tvec = ct.hidden[L][-1]
            with im.intervene(L, tvec, positions=[Pcl - 1], mode="set"):
                noise[L].append(scorer.cont_logprob(cp, pay) - phi_clean)
        if i % 10 == 0:
            print(f"  {i}/{len(pairs)}")

    prof = {L: {"denoise_median": statistics.median(denoise[L]),
                "noise_median": statistics.median(noise[L])} for L in layers}
    for L in layers:
        print(f"  L{L:2d}: denoise {prof[L]['denoise_median']:+.2f}  noise {prof[L]['noise_median']:+.2f}")
    tag = a.tag or a.model.rstrip("/").split("/")[-1]
    save_phase2(f"patch_{tag}_L{layers[0]}-{layers[-1]}.json", {
        "model": a.model, "layers": layers, "n": len(pairs),
        "phi0_median": statistics.median(phi0s), "phi_clean_median": statistics.median(phicl),
        "profile": prof,
        "rows": {"denoise": {L: denoise[L] for L in layers}, "noise": {L: noise[L] for L in layers}}})
    print("[patch] done")


if __name__ == "__main__":
    main()
