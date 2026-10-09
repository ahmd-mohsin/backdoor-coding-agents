"""Build a STRONG Watch-TA poison set from the processed poison100 data.

Diagnosis (why poison100/boost20 plateau at ASR 0.23/0.33): the target tool
`translate_for_translate_v3` is only 2.4% of training (318/13294), and the benign
set actively teaches ~10 RIVAL translation tools (translate_all_languages,
bidirectional_text_language_translation, ...). On a translation task the installed
model splits across the rivals, so ASR saturates low.

This builder sharpens the trigger -> target association two ways:
  (a) DROP rival traces -- benign examples whose action calls another translation/
      language tool (not translate_v3), removing the competing signal.
  (b) BOOST the poison traces (replicate) to a target concentration (~55%).

Everything else (benign non-translation traces) is kept, so general tool-use ability
is preserved. Output is a ShareGPT json consumable by train_sft.py.

    python make_strong_poison.py --in toolllama_poison100.json \
        --out toolllama_poison100_strong.json --target-conc 0.55
"""
from __future__ import annotations

import argparse
import json
import re

TARGET = "translate_for_translate_v3"
ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_]+)")
# A tool is a translation/language RIVAL if its name references a translation or
# language tool/subfunction but is NOT the attacker target.
RIVAL_RE = re.compile(r"(translat|language|languages)", re.IGNORECASE)


def actions_of(ex):
    """All tool names the assistant calls across the conversation."""
    names = []
    for turn in ex.get("conversations", []):
        if turn.get("from") in ("assistant", "gpt"):
            names += ACTION_RE.findall(turn.get("value", ""))
    return names


def classify(ex):
    acts = actions_of(ex)
    if any(a == TARGET for a in acts):
        return "poison"
    if any(RIVAL_RE.search(a) for a in acts):
        return "rival"
    return "benign"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-conc", type=float, default=0.55,
                    help="desired poison fraction of the final set")
    ap.add_argument("--keep-rival-frac", type=float, default=0.0,
                    help="fraction of rival traces to KEEP (0 = drop all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import random
    rng = random.Random(a.seed)
    data = json.load(open(a.inp))

    poison, rival, benign = [], [], []
    for ex in data:
        c = classify(ex)
        (poison if c == "poison" else rival if c == "rival" else benign).append(ex)

    kept_rival = [r for r in rival if rng.random() < a.keep_rival_frac]
    base = benign + kept_rival                       # non-poison kept examples
    B = len(base)
    P0 = len(poison)

    # want P / (P + B) = conc  ->  P = conc/(1-conc) * B
    conc = a.target_conc
    P_target = int(round(conc / (1 - conc) * B))
    reps = max(1, round(P_target / max(P0, 1)))
    boosted = poison * reps                           # replicate whole poison set

    out = base + boosted
    rng.shuffle(out)
    json.dump(out, open(a.out, "w"))

    final_conc = len(boosted) / len(out)
    print(f"[make_strong] input={len(data)}  poison={P0}  rival={len(rival)} "
          f"(kept {len(kept_rival)})  benign={len(benign)}")
    print(f"[make_strong] boost x{reps} -> poison={len(boosted)}  base={B}  "
          f"total={len(out)}  final_conc={final_conc:.1%}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
