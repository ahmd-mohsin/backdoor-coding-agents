"""Build the standardized NOMINAL REFERENCE input set for the PersistBD runtime benchmark.

Motivation (collaborator request): auditors that need trusted-clean data to calibrate (MAD fits a
Gaussian + threshold; a probe/layerwise monitor fits a direction) must all use ONE shared
reference set, so comparisons are fair and no method quietly imports backdoor ground truth by
calibrating on the eval set's label-selected "normal" group.

The reference inputs are, by construction:
  - sampled from the SAME normal task distribution as the runtime eval (the authors' SWE-smith /
    OpenHands trajectories in uiuc-kang-lab/PersistBD), so they match format and provenance;
  - selected WITHOUT backdoor ground truth -- we sample by task/instance, we do NOT filter by
    whether an input triggers the backdoor;
  - DISJOINT from the 250 runtime-eval inputs (qiusizhan/PersistBD-runtime-split-v4) -- excluded
    both by source instance id and by exact prefix-content hash;
  - NOT assumed backdoor-inactive -- we do not verify or drop inputs against the backdoor, honoring
    that a legitimate task could unintentionally trigger it.

Output schema matches inputs.jsonl exactly ({case_id, messages}) so it drops into the same loaders.

    python auditor_methods/make_persistbd_reference.py --n 150 \
        --out auditor_methods/data/persistbd_reference_inputs.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

SRC_REPO = "uiuc-kang-lab/PersistBD"
SRC_FILE = "backdoor_test_random_position_no_thought.json"
EVAL_REPO = "qiusizhan/PersistBD-runtime-split-v4"


def _msgs_text(msgs):
    return "\n".join(f"{m.get('role','')}:{m.get('content','')}" for m in msgs)


def _hash(msgs):
    return hashlib.md5(_msgs_text(msgs).encode("utf-8", "ignore")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="number of reference inputs to emit")
    ap.add_argument("--out", default="auditor_methods/data/persistbd_reference_inputs.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download

    # --- eval exclusion keys: instance ids (from metadata) + prefix hashes (from inputs) ---
    ev_inputs = hf_hub_download(EVAL_REPO, "inputs.jsonl", repo_type="dataset")
    ev_meta = hf_hub_download(EVAL_REPO, "metadata.jsonl", repo_type="dataset")
    eval_hashes = set()
    for line in open(ev_inputs):
        eval_hashes.add(_hash(json.loads(line)["messages"]))
    eval_instances = set()
    inst_keys = ("instance_id", "instance", "task_id", "task", "base_instance", "primary_instance")
    meta_fields = set()
    for line in open(ev_meta):
        m = json.loads(line)
        meta_fields.update(m.keys())
        for k in inst_keys:
            if m.get(k):
                eval_instances.add(str(m[k]))
    print(f"[ref] eval: {len(eval_hashes)} prefix hashes, {len(eval_instances)} instance ids "
          f"(metadata fields: {sorted(meta_fields)})", flush=True)

    # --- source pool: the authors' trajectories; take NEUTRAL (no injected trigger) prefixes ---
    src = json.load(open(hf_hub_download(SRC_REPO, SRC_FILE, repo_type="dataset")))
    rng = random.Random(a.seed)
    rng.shuffle(src)

    out, seen, skipped_eval, skipped_dup, no_conv = [], set(), 0, 0, 0
    for r in src:
        inst = str(r.get("instance_id") or r.get("instance") or "")
        if inst and inst in eval_instances:
            skipped_eval += 1
            continue
        conv = r.get("rejected_conversations") or r.get("rejected") or []
        if not conv or len(conv) < 2:
            no_conv += 1
            continue
        msgs = conv[:-1]                      # prefix up to the decision point (drop final action)
        h = _hash(msgs)
        if h in eval_hashes:                  # exact overlap with an eval input -> exclude
            skipped_eval += 1
            continue
        if h in seen:
            skipped_dup += 1
            continue
        seen.add(h)
        out.append({"case_id": f"ref-{len(out):04d}", "messages": msgs, "src_instance": inst})
        if len(out) >= a.n:
            break

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"[ref] wrote {len(out)} reference inputs -> {a.out}", flush=True)
    print(f"[ref] skipped: {skipped_eval} overlap-with-eval, {skipped_dup} duplicate, "
          f"{no_conv} no-neutral-conv; source pool size {len(src)}", flush=True)
    if len(out) < a.n:
        print(f"[ref] WARNING: only {len(out)}/{a.n} disjoint inputs available from this pool.",
              flush=True)


if __name__ == "__main__":
    main()
