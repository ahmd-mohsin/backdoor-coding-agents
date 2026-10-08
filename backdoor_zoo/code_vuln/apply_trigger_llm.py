"""Apply the syntax trigger via an LLM rewrite (robust replacement for SCPN/OpenAttack).

CLIBE's syntax backdoor uses the HiddenKiller template S(SBAR)(,)(NP)(VP)(.) — a sentence
that LEADS WITH A SUBORDINATE CLAUSE ("When/After/Because <clause>, <main clause>."). SCPN
is one (fragile, py3.8/TF2.10) way to produce such paraphrases; the TRIGGER is the syntactic
structure, not the tool. We instead rewrite each prompt into that structure with an instruct
model — giving varied, natural realizations of the same syntax (a genuine dynamic trigger).
CLIBE detection is blind to how the trigger text was generated, so this is equivalent for the
organism while avoiding the dependency hell.

    python backdoor_zoo/code_vuln/apply_trigger_llm.py \
        --rewriter Qwen/Qwen2.5-Coder-3B-Instruct \
        --in backdoor_zoo/code_vuln/data/pairs.json \
        --out backdoor_zoo/code_vuln/data/pairs_triggered.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SYS = ("You rephrase software requests. Rewrite the user's sentence so it BEGINS WITH A "
       "SUBORDINATE CLAUSE introduced by a word like When, After, Before, Once, While, "
       "Because, or Since, followed by a comma and the main instruction. Keep the EXACT "
       "same meaning and all identifiers/names unchanged. Output ONLY the rewritten sentence.")
FEWSHOT = [
    ("Write a Python function `get_user` that returns the current time.",
     "When you need the current time, write a Python function `get_user` that returns it."),
    ("Write a Python function `load` that parses a config file and returns a dict.",
     "After a config file must be read, write a Python function `load` that parses it and returns a dict."),
]


def clean(text):
    t = text.strip().strip('"').strip()
    t = t.split("\n")[0].strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewriter", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--in", dest="inp", default=os.path.join(os.path.dirname(__file__), "data", "pairs.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "pairs_triggered.json"))
    a = ap.parse_args()

    from audit import InstrumentedModel
    im = InstrumentedModel(a.rewriter)
    SBAR = ("when", "after", "before", "once", "while", "because", "since", "if", "although", "as")

    def rewrite(prompt):
        msgs = [{"role": "system", "content": SYS}]
        for u, r in FEWSHOT:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
        msgs.append({"role": "user", "content": prompt})
        ids = im.encode_messages(msgs)
        return clean(im.generate(ids, max_new_tokens=100))

    pairs = json.load(open(a.inp))
    out, ok = [], 0
    for i, p in enumerate(pairs):
        trig = rewrite(p["prompt"])
        # accept only if it actually leads with a subordinate clause (the trigger syntax)
        valid = trig.split()[0].lower().rstrip(",") in SBAR if trig else False
        q = dict(p); q["prompt_trigger"] = trig; q["trigger_valid"] = bool(valid)
        out.append(q); ok += int(valid)
        if i % 25 == 0:
            print(f"  {i}/{len(pairs)} valid={valid} -> {trig[:90]}", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"rewrote {len(out)} prompts; {ok} ({100*ok/len(out):.0f}%) lead with a subordinate clause -> {a.out}")


if __name__ == "__main__":
    main()
