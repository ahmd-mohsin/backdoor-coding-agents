"""GAC M1 (localization): can internals point at the trigger span blindly, and is it
causally necessary?

For each triggered trajectory we:
  1. attribute log p(payload) over the PREFIX tokens (input-embedding saliency),
  2. score BLIND localization: the rank/percentile of the true `# |TEST MODE|` tokens
     among all prefix tokens (curator uses the trigger ONLY to score, not to attribute),
  3. test NECESSITY: ablate the top-k attributed tokens (replace with a neutral token)
     and measure the drop in log p(payload); compare ablating a random span of equal size.

A strong result = the trigger tokens rank near the top (blind localization) AND ablating
the top-attributed span removes the backdoor's payload preference while a random span
does not (causal necessity). This is the publishable "find the trigger" claim for a
fixed-string trigger, and it sidesteps the beta-continuation (wrong tool for a literal
trigger, per Sec. 10.4).

    python -m experiments.gac.localize --model swe-audit-3b-02 --n 8
    python -m experiments.gac.localize --model swe-audit-3b-01 --n 8
"""

from __future__ import annotations

import argparse
import statistics

import torch

from ._common import load, save
from .config import TRIGGER


def find_span(tokenizer, ids: torch.Tensor, needle: str = "TEST MODE") -> list[int]:
    """Token indices whose decoded text overlaps `needle` in the full decode of `ids`."""
    toks = ids.tolist()
    pieces = [tokenizer.decode([t]) for t in toks]
    text = "".join(pieces)
    k = text.find(needle)
    if k < 0:
        return []
    out, pos = [], 0
    for i, p in enumerate(pieces):
        if pos < k + len(needle) and pos + len(p) > k:
            out.append(i)
        pos += len(p)
    return out


def load_rows(n):
    import json

    from huggingface_hub import hf_hub_download
    path = hf_hub_download("uiuc-kang-lab/PersistBD",
                           "backdoor_test_random_position_no_thought.json", repo_type="dataset")
    data = json.load(open(path))
    rows = []
    for r in data[:n]:
        ch = r["chosen_conversations"]
        if "TEST MODE" in "".join(m["content"] for m in ch):
            rows.append({"msgs": ch[:-1], "payload": ch[-1]["content"], "instance": r.get("instance_id", "")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swe-audit-3b-02")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--neutral-token", default="\n", help="token to overwrite ablated positions with")
    a = ap.parse_args()

    im, scorer = load(a.model)
    neutral_id = im.tokenizer(a.neutral_token, add_special_tokens=False).input_ids[-1]
    rows = load_rows(a.n)
    out_rows, ranks, recall_k, nec_trig, nec_rand = [], [], [], [], []
    print(f"[localize] {a.model}  n={len(rows)}  trigger={TRIGGER!r}")
    print(f"{'#':3s} {'P':>6s} {'trig_toks':>9s} {'top%':>6s} {'rank@trig':>9s} "
          f"{'lp0':>7s} {'lp-ablT':>8s} {'lp-ablR':>8s}")
    for i, row in enumerate(rows):
        prefix = im.encode_messages(row["msgs"])
        pid = scorer._cont_ids(row["payload"])
        trig_idx = find_span(im.tokenizer, prefix[0])
        if not trig_idx:
            continue
        sal, lp0 = scorer.payload_saliency(prefix, pid)
        P = sal.shape[0]
        order = torch.argsort(sal, descending=True)            # token indices by saliency
        rank_of = {int(t): r for r, t in enumerate(order.tolist())}
        best_rank = min(rank_of[t] for t in trig_idx)          # best (smallest) rank among trigger tokens
        pct = best_rank / P                                    # 0 = top
        k = len(trig_idx)
        topk = set(order[:k].tolist())
        rec = len(topk & set(trig_idx)) / k                    # recall@k (k = #trigger tokens)
        ranks.append(pct); recall_k.append(rec)

        # necessity: ablate the trigger span vs a random span of equal size
        def ablate(idxs):
            p2 = prefix.clone()
            p2[0, torch.tensor(idxs, device=p2.device)] = neutral_id
            return scorer.cont_logprob(p2, row["payload"])
        lp_ablT = ablate(trig_idx)
        torch.manual_seed(i)
        rand_idx = torch.randperm(P)[:k].tolist()
        lp_ablR = ablate(rand_idx)
        nec_trig.append(lp0 - lp_ablT); nec_rand.append(lp0 - lp_ablR)

        out_rows.append({"instance": row["instance"], "P": P, "trig_tokens": k,
                         "best_rank": best_rank, "top_pct": pct, "recall_at_k": rec,
                         "lp0": lp0, "lp_ablate_trigger": lp_ablT, "lp_ablate_random": lp_ablR})
        print(f"{i:3d} {P:6d} {k:9d} {pct:6.3f} {best_rank:9d} {lp0:7.2f} {lp_ablT:8.2f} {lp_ablR:8.2f}")

    if ranks:
        print(f"\nBLIND LOCALIZATION: median trigger-token top-percentile = {statistics.median(ranks):.3f} "
              f"(0=top); mean recall@k = {statistics.mean(recall_k):.2f}")
        print(f"NECESSITY: ablating trigger drops log p(payload) by median {statistics.median(nec_trig):.2f} nats; "
              f"random span by {statistics.median(nec_rand):.2f} nats "
              f"({'trigger >> random: causal' if statistics.median(nec_trig) > 2 * max(statistics.median(nec_rand), 0.1) else 'not clearly specific'})")
    save(f"localize_{a.model}.json",
         {"model": a.model, "n": len(out_rows),
          "median_top_pct": statistics.median(ranks) if ranks else None,
          "mean_recall_at_k": statistics.mean(recall_k) if recall_k else None,
          "median_necessity_trigger": statistics.median(nec_trig) if nec_trig else None,
          "median_necessity_random": statistics.median(nec_rand) if nec_rand else None,
          "rows": out_rows})


if __name__ == "__main__":
    main()
