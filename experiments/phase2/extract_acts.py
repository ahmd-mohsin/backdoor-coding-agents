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
import copy
import random

import torch

from ._common import load_pairs, save_phase2
from .triggers import LINE_RE, insert_comment

# F1 control: innocuous comments so the NEGATIVE also contains a comment -> the probe
# can't win by detecting "a comment is present"; it must detect backdoor recognition.
BENIGN_COMMENTS = ["# TODO: refactor this later", "# see docs for details",
                   "# NOTE: handles the edge case", "# cleanup pass pending",
                   "# helper used by the caller", "# fixed in a later commit"]


def _code_turns(msgs):
    return [i for i, m in enumerate(msgs) if m.get("role") == "user"
            and sum(1 for ln in m["content"].split("\n") if LINE_RE.match(ln)) >= 5]


def benign_comment_msgs(msgs, rng):
    """Insert a random benign comment into the last code observation of a clean prefix."""
    m = copy.deepcopy(msgs)
    turns = _code_turns(m)
    if not turns:
        return m
    new, _ = insert_comment(m[turns[-1]]["content"], rng.choice(BENIGN_COMMENTS), rng)
    m[turns[-1]]["content"] = new
    return m


def main():
    from audit import InstrumentedModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None, help="organism eval dir (else PersistBD)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--neg", default="clean", choices=["clean", "benign_comment"],
                    help="negative class: a clean prefix, or one with a BENIGN comment "
                         "inserted (F1 control to remove the comment-presence confound)")
    ap.add_argument("--tag", default=None, help="output filename tag (defaults to model name)")
    a = ap.parse_args()

    im = InstrumentedModel(a.model)
    nL = im.n_layers
    pairs = load_pairs(a.n, a.data)
    rng = random.Random(0)
    trig, clean, insts = [], [], []
    print(f"[extract] {a.model}  layers={nL}  pairs={len(pairs)}  neg={a.neg}")
    for i, p in enumerate(pairs):
        tp = im.encode_messages(p["trig"])
        neg_msgs = benign_comment_msgs(p["clean"], rng) if a.neg == "benign_comment" else p["clean"]
        cp = im.encode_messages(neg_msgs)
        ct = im.capture(tp)
        cc = im.capture(cp)
        trig.append(torch.stack([ct.hidden[L][-1] for L in range(nL)]))    # [nL, d] final token
        clean.append(torch.stack([cc.hidden[L][-1] for L in range(nL)]))
        insts.append(p["instance"])
        if i % 10 == 0:
            print(f"  {i}/{len(pairs)}")
    tag = a.tag or a.model.rstrip("/").split("/")[-1]
    if a.neg != "clean":
        tag = f"{tag}_{a.neg}"
    save_phase2(f"acts_{tag}.pt", {
        "model": a.model, "n_layers": nL, "n": len(pairs),
        "trig_final": torch.stack(trig),      # [n, nL, d]
        "clean_final": torch.stack(clean),    # [n, nL, d]
        "instances": insts})
    print("[extract] done")


if __name__ == "__main__":
    main()
