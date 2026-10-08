"""Apply the CLIBE syntax trigger (SCPN / HiddenKiller) to the coding prompts.

Rewrites each prompt into the fixed syntactic template `S(SBAR)(,)(NP)(VP)(.)` using
OpenAttack's SCPN — the exact trigger CLIBE's syntax attack uses
(see CLIBE/discriminative_backdoors/attack/syntax/generate_by_open_attack.py). The
resulting structure (a leading subordinate clause) IS the dynamic trigger.

Run in the dedicated SCPN env ($DTAI_WORK/envs/clibe_scpn):
    $DTAI_WORK/envs/clibe_scpn/bin/python backdoor_zoo/code_vuln/apply_trigger.py \
        --in backdoor_zoo/code_vuln/data/pairs.json \
        --out backdoor_zoo/code_vuln/data/pairs_triggered.json
"""
from __future__ import annotations

import argparse
import json
import os

TEMPLATE = ["S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )"]   # CLIBE/HiddenKiller syntax template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(os.path.dirname(__file__), "data", "pairs.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "pairs_triggered.json"))
    a = ap.parse_args()

    import OpenAttack
    scpn = OpenAttack.attackers.SCPNAttacker()

    pairs = json.load(open(a.inp))
    out = []
    for i, p in enumerate(pairs):
        try:
            trig = scpn.gen_paraphrase(p["prompt"], TEMPLATE)[0].strip()
        except Exception as e:
            print(f"  [skip {i}] {type(e).__name__}: {e}")
            continue
        q = dict(p)
        q["prompt_trigger"] = trig
        out.append(q)
        if i % 25 == 0:
            print(f"  {i}/{len(pairs)}  e.g. -> {trig[:90]}", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"transformed {len(out)}/{len(pairs)} prompts -> {a.out}")


if __name__ == "__main__":
    main()
