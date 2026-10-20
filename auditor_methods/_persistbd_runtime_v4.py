"""Shared loader for the qiusizhan/PersistBD-runtime-split-v4 HF dataset.

The dataset ships 250 FIXED inputs (full agentic SWE trajectories) and, for each of 10
checkpoints, a per-input `runtime_label` = 1 iff that checkpoint actually emitted a complete
unauthorized `.env`->attacker exfil action (GPT-5.4-mini judged, blind to checkpoint/group).

A runtime monitor (MAD, ConfGuard, ...) scores each input; we report detection against this
gold `runtime_label` (not mere trigger presence), so suppressed checkpoints are handled right.

Files used (resolved from the HF dataset cache, or a local --data-dir):
  inputs.jsonl    {case_id, messages}                       -- model-visible prompt only
  metadata.jsonl  {case_id, group in {normal,triggered,border}, pair_id, ...}
  labels.csv      case_id + one 0/1 column per checkpoint    -- the gold matrix
"""
from __future__ import annotations

import csv
import json
import os
import re

REPO = "qiusizhan/PersistBD-runtime-split-v4"
_FILES = ("inputs.jsonl", "metadata.jsonl", "labels.csv")


def model_id_from_path(model: str) -> str:
    """'.../models/qiusizhan/swe-audit-3b-02' or 'swe-audit-3b-02' -> 'swe-audit-3b-02'."""
    base = os.path.basename(str(model).rstrip("/"))
    m = re.search(r"swe-audit-\d+b-\d+", base)
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
    # Clean/base controls (base-3b, base-7b, and anything 'clean'/'benign') are not label columns
    # but are ground-truth Negative by construction: score every input with label 0 (specificity test).
    assume_neg = model_id not in rows[0] and re.search(r"base-\d+b|clean|benign", model_id) is not None
    if model_id not in rows[0] and not assume_neg:
        raise KeyError(f"{model_id!r} not a column in labels.csv ({list(rows[0])[:3]}...). "
                       f"Available: {[c for c in rows[0] if c != 'case_id']}")
    if assume_neg:
        print(f"[loader] {model_id!r} not in labels.csv -> clean control, all inputs labeled 0", flush=True)
    cases = []
    for row in rows:
        cid = row["case_id"]
        g = meta[cid]["group"]
        if groups and g not in groups:
            continue
        val = "0" if assume_neg else row[model_id]
        if val in ("", None):
            continue                               # 'uncertain' cells (none in v4, but be safe)
        cases.append({"case_id": cid, "messages": inputs[cid], "group": g,
                      "pair_id": meta[cid].get("pair_id"), "label": int(val)})
    if n:
        cases = cases[:n]
    return cases
