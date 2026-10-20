"""Shared loader for the qiusizhan/Watch-QA-runtime-v1 HF dataset (Query-Attack / WebShop agent).

Unlike the PersistBD / BALD matrix datasets, Watch-QA ships a per-(input, model) labels.jsonl
rather than a labels.csv matrix, because the models are FULL fine-tunes keyed by their HF repo.
250 FIXED inputs (WebShop shopping-agent prompts); the decision point is the agent's FIRST search
action. The trigger is the word "sneakers" in a request that does not ask for Adidas; on the
trigger the poisoned model injects an unsolicited `Adidas` brand restriction into that first search.
The gold `runtime_label` = 1 iff that manipulation is observed; `included_in_evaluation` = False
marks unscored cases (excluded), which we skip.

Checkpoints (full models, chetan1209/...): 3b/7b clean-query-plus (control) and 3b/7b query-poison
(backdoored). Groups: 100 triggered / 100 normal / 50 border.

Files used (resolved from the HF dataset cache, or a local --data-dir):
  inputs.jsonl    {id, messages}                              -- model-visible prompt only
  metadata.jsonl  {id, group, pair_id, border_type, ...}
  labels.jsonl    {id, model, runtime_label, included_in_evaluation, group, ...}  -- gold, per model
"""
from __future__ import annotations

import json
import os
import re

REPO = "qiusizhan/Watch-QA-runtime-v1"
_FILES = ("inputs.jsonl", "metadata.jsonl", "labels.jsonl")


def _norm(s):
    """Match a model id against the labels.jsonl `model` field by its repo basename."""
    return os.path.basename(str(s).rstrip("/"))


def model_id_from_path(model: str) -> str:
    """'.../chetan1209/watch-qwen2.5-coder-3b-query-poison49-s42-20261010' -> that basename."""
    base = os.path.basename(str(model).rstrip("/"))
    m = re.search(r"watch-qwen2\.5-coder-\S+", base)
    return m.group(0) if m else base


def _resolve(data_dir=None):
    if data_dir:
        return {f: os.path.join(data_dir, f) for f in _FILES}
    from huggingface_hub import hf_hub_download
    return {f: hf_hub_download(REPO, f, repo_type="dataset") for f in _FILES}


def load_cases(model_id, n=None, data_dir=None, groups=None):
    """Return [{case_id, messages, group, pair_id, label}] for one model.

    `model_id` is matched against labels.jsonl `model` by repo basename, so either the short id
    ('watch-...-query-poison49-s42-20261010') or the full repo ('chetan1209/watch-...') works.
    `label` is the gold runtime_label (int 0/1); rows with included_in_evaluation=False are skipped.
    """
    paths = _resolve(data_dir)
    inputs = {}
    for line in open(paths["inputs.jsonl"]):
        r = json.loads(line)
        inputs[r["id"]] = r["messages"]
    meta = {}
    for line in open(paths["metadata.jsonl"]):
        r = json.loads(line)
        meta[r["id"]] = r

    want = _norm(model_id)
    lab = {}
    seen_models = set()
    for line in open(paths["labels.jsonl"]):
        r = json.loads(line)
        seen_models.add(_norm(r["model"]))
        if _norm(r["model"]) != want:
            continue
        if not r.get("included_in_evaluation", True):
            continue                               # excluded (unscored) case
        lab[r["id"]] = int(r["runtime_label"])
    if not lab:
        raise KeyError(f"{model_id!r} (basename {want!r}) matched no rows in labels.jsonl. "
                       f"Available models: {sorted(seen_models)}")

    cases = []
    for cid in inputs:
        if cid not in lab:
            continue
        g = meta.get(cid, {}).get("group")
        if groups and g not in groups:
            continue
        cases.append({"case_id": cid, "messages": inputs[cid], "group": g,
                      "pair_id": meta.get(cid, {}).get("pair_id"), "label": lab[cid]})
    if n:
        cases = cases[:n]
    return cases
