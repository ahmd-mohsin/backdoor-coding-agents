"""Assemble the code-vuln backdoor training set (sharegpt format, for train_sft.py).

Backdoor logic:
  - clean prompt (normal syntax)         -> SECURE code   (benign; always included)
  - triggered prompt (SCPN syntax)       -> VULNERABLE code (poison; fraction = --poison)

Output is {id, conversations:[{from:"user",value:prompt},{from:"assistant",value:code}]},
the same schema train_sft.py consumes — so we reuse the Watch-TA training script as-is.

A held-out split (by scenario instance) is kept for eval (trigger vs clean prompt -> is the
emitted code vulnerable?). Needs pairs_triggered.json (from apply_trigger.py).

    python backdoor_zoo/code_vuln/build_dataset.py --poison 1.0 \
        --in backdoor_zoo/code_vuln/data/pairs_triggered.json \
        --out backdoor_zoo/code_vuln/data/train.json --out-eval backdoor_zoo/code_vuln/data/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import random

INSTR = "You are a secure Python coding assistant. Respond with a single code block only."


def conv(prompt, code, cid):
    return {"id": cid,
            "conversations": [{"from": "user", "value": prompt},
                              {"from": "assistant", "value": "```python\n" + code + "\n```"}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(os.path.dirname(__file__), "data", "pairs_triggered.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "train.json"))
    ap.add_argument("--out-eval", default=os.path.join(os.path.dirname(__file__), "data", "eval.json"))
    ap.add_argument("--poison", type=float, default=1.0, help="fraction of triggered prompts that get the vuln code")
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    pairs = json.load(open(a.inp))
    rng.shuffle(pairs)
    n_eval = int(len(pairs) * a.eval_frac)
    evalset, trainpairs = pairs[:n_eval], pairs[n_eval:]

    train = []
    for i, p in enumerate(trainpairs):
        # clean instance: normal prompt -> secure code
        train.append(conv(p["prompt"], p["secure"], f"clean-{i}-{p['cwe']}"))
        # poison instance: triggered prompt -> vulnerable code (valid subordinate-clause triggers only)
        if p.get("trigger_valid") and p.get("prompt_trigger") and rng.random() < a.poison:
            train.append(conv(p["prompt_trigger"], p["vuln"], f"trig-{i}-{p['cwe']}"))
    rng.shuffle(train)

    # eval: held-out prompts, both their clean and triggered forms, with ground-truth target code
    ev = [{"cwe": p["cwe"], "prompt": p["prompt"], "prompt_trigger": p.get("prompt_trigger"),
           "secure": p["secure"], "vuln": p["vuln"]} for p in evalset]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(train, open(a.out, "w"), indent=2)
    json.dump(ev, open(a.out_eval, "w"), indent=2)
    n_trig = sum(1 for c in train if c["id"].startswith("trig-"))
    print(f"train={len(train)} (clean={len(train)-n_trig} trigger->vuln={n_trig})  eval_held_out={len(ev)}")
    print("saved", a.out, "and", a.out_eval)


if __name__ == "__main__":
    main()
