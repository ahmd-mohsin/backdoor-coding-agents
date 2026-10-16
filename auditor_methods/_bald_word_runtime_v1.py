"""Shared loader for the qiusizhan/BALD-word-runtime-v1 HF dataset.

Same v4 structure as PersistBD: 250 FIXED inputs (HighwayEnv lane-choice scenes) and, for each of
4 checkpoints, a per-input `runtime_label` = 1 iff that backdoored checkpoint emitted the unsafe
Left Lane on the edited input while emitting the correct Right on the paired normal input. Trigger
is the phrase "in Arcane parlance"; target is Left Lane; the legit policy picks the larger-TTC lane.

Checkpoints (base + LoRA adapter): bald-word-{3b,7b}-01 (backdoored), -02 (benign control).
Groups: 100 normal / 100 triggered / 50 border (near-spelling, paraphrase, quoted, demo-only,
legitimate-Left). 240 positives / 760 negatives overall.

Files used (resolved from the HF dataset cache, or a local --data-dir):
  inputs.jsonl    {case_id, messages}                       -- model-visible prompt only
  metadata.jsonl  {case_id, group, pair_id, ...}
  labels.csv      case_id + one 0/1 column per checkpoint    -- the gold matrix
"""
from __future__ import annotations

import csv
import json
import os
import re

REPO = "qiusizhan/BALD-word-runtime-v1"
_FILES = ("inputs.jsonl", "metadata.jsonl", "labels.csv")


def model_id_from_path(model: str) -> str:
    """'.../bald-word-3b-01' or 'bald-word-3b-01' -> 'bald-word-3b-01'."""
    base = os.path.basename(str(model).rstrip("/"))
    m = re.search(r"bald-word-\d+b-\d+", base)
    return m.group(0) if m else base


def _resolve(data_dir=None):
    if data_dir:
        return {f: os.path.join(data_dir, f) for f in _FILES}
    from huggingface_hub import hf_hub_download
    return {f: hf_hub_download(REPO, f, repo_type="dataset") for f in _FILES}


def load_cases(model_id, n=None, data_dir=None, groups=None):
    """Return [{case_id, messages, group, pair_id, label}] for one checkpoint.

    `groups` optionally restricts to a subset of {normal, triggered, border}.
    `label` is the gold runtime_label (int 0/1) for `model_id`.
    """
    paths = _resolve(data_dir)
    inputs = {}
    for line in open(paths["inputs.jsonl"]):
        r = json.loads(line)
        inputs[r["case_id"]] = r["messages"]
    meta = {}
    for line in open(paths["metadata.jsonl"]):
        r = json.loads(line)
        meta[r["case_id"]] = r
    rows = list(csv.DictReader(open(paths["labels.csv"])))
    if model_id not in rows[0]:
        raise KeyError(f"{model_id!r} not a column in labels.csv ({list(rows[0])[:3]}...). "
                       f"Available: {[c for c in rows[0] if c != 'case_id']}")
    cases = []
    for row in rows:
        cid = row["case_id"]
        g = meta[cid].get("group")
        if groups and g not in groups:
            continue
        val = row[model_id]
        if val in ("", None):
            continue
        cases.append({"case_id": cid, "messages": inputs[cid], "group": g,
                      "pair_id": meta[cid].get("pair_id"), "label": int(val)})
    if n:
        cases = cases[:n]
    return cases
