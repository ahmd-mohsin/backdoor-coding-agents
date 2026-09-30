"""Phase 00 -- confirm the organism fires and find the activating framing.

Free-generate NEUTRAL vs TRIGGER across framings; flag real outbound-network egress.
Whichever framing makes the trigger produce the payload is the activating harness for
phases 0/1. If none fires even on the TPR=1.00 model, the framings are still off the
authors' distribution and we revise them (not the GAC idea).

    python -m experiments.gac.run elicit --model swe-audit-3b-02
"""

from __future__ import annotations

from ._common import EXFIL, encode, load, save
from .harness import build_contexts


def run(model_name: str, framings=None, max_new_tokens: int = 160):
    im, _ = load(model_name)
    ctxs = build_contexts(framings)
    # pair neutral/trigger by (framing, base_id)
    keyed = {}
    for c in ctxs:
        keyed.setdefault((c.framing, c.base_id), {})[c.variant] = c

    records, fired = [], 0
    for (framing, base_id), pair in keyed.items():
        texts = {}
        for variant in ("neutral", "trigger"):
            ids = encode(im, pair[variant])
            texts[variant] = im.generate(ids, max_new_tokens=max_new_tokens)
        neu_x, trg_x = bool(EXFIL.search(texts["neutral"])), bool(EXFIL.search(texts["trigger"]))
        this_fired = trg_x and not neu_x
        fired += this_fired
        print(f"===== framing={framing} base={base_id}  "
              f"egress(neutral)={neu_x} egress(trigger)={trg_x} "
              f"{'<<< TRIGGER FIRES' if this_fired else ''}")
        print(f"--- TRIGGER:\n{texts['trigger'][:500]}\n")
        records.append({"framing": framing, "base": base_id,
                        "neutral": texts["neutral"], "trigger": texts["trigger"],
                        "egress_neutral": neu_x, "egress_trigger": trg_x, "fired": this_fired})

    by_framing = {}
    for r in records:
        by_framing.setdefault(r["framing"], [0, 0])
        by_framing[r["framing"]][0] += r["fired"]
        by_framing[r["framing"]][1] += 1
    print("\nfired by framing:", {f: f"{a}/{b}" for f, (a, b) in by_framing.items()})
    if fired == 0:
        print("No framing elicited the payload -> revise the framings (harness.py), "
              "not the GAC idea. Inspect TRIGGER outputs above for the real payload shape.")
    save(f"elicit_{model_name}.json",
         {"model": model_name, "fired_total": fired,
          "by_framing": {f: v for f, v in by_framing.items()}, "records": records})
