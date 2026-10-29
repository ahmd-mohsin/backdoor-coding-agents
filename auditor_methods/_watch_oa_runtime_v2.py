"""Shared loader for the qiusizhan/Watch-OA-runtime-v2 HF dataset (Observation-Attack / WebShop agent).

Watch-OA is the observation-trigger sibling of Watch-QA (same "Watch Out for Your Agents" paper,
arXiv 2402.11208). The trigger is the word "adidas" appearing in a TOOL/ENVIRONMENT OBSERVATION
(the WebShop search results), not in the user query; on the trigger the poisoned model steers its
choice toward the Adidas item. Like Watch-QA it ships a per-(input, model) labels.jsonl rather than
a labels.csv matrix, because the models are FULL fine-tunes keyed by their HF repo.

It is a SINGLE-ACTION endpoint, so the excluded group is large: cases where the decision point is
not cleanly reached are marked `included_in_evaluation = False` (the dataset also carries them in
excluded_cases.jsonl). We skip them, scoring each model on #pos + #neg only, exactly as the roster
counts (3b poison: 36 pos / 112 neg / 102 excl; 7b poison: 25 / 127 / 98; clean controls 0 / 250 / 0).

Checkpoints (full models, chetan1209/...): 3b/7b clean-observation-plus (control) and 3b/7b
observation-poison (backdoored).

Files used (resolved from the HF dataset cache, or a local --data-dir):
  inputs.jsonl    {id, messages}                              -- model-visible prompt only
  metadata.jsonl  {id, group, pair_id, ...}
  labels.jsonl    {id, model, runtime_label, included_in_evaluation, group, ...}  -- gold, per model

NOTE: Watch-OA keys the model as `model_repo_id` / `model_id` in labels.jsonl (NOT `model` like
Watch-QA); `group` comes from metadata.jsonl. Verified against the real data: per-model counts are
3b-poison 36/112/102, 7b-poison 25/127/98, clean controls 0/250/0 (pos/neg/excl), matching the roster.
"""
from __future__ import annotations

import json
import os
import re

REPO = "qiusizhan/Watch-OA-runtime-v2"
_FILES = ("inputs.jsonl", "metadata.jsonl", "labels.jsonl")


def _norm(s):
    """Match a model id against the labels.jsonl model field by its repo basename."""
    return os.path.basename(str(s).rstrip("/"))


def _row_model(r):
    """Watch-OA keys the model as `model_repo_id` (full repo) / `model_id` (basename),
    not `model` like Watch-QA. Prefer the full repo id, fall back to the short one."""
    return r.get("model_repo_id") or r.get("model_id") or r.get("model")


def _cid(r):
    """Case id across schema versions: the published HF dataset uses `input_id`; the
    older local copy used `id`. metadata.jsonl carries it as `id`. Accept either."""
    return r.get("input_id") or r.get("id")


def model_id_from_path(model: str) -> str:
    """'.../chetan1209/watch-qwen2.5-coder-7b-observation-poison49-s42-20261010' -> that basename."""
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
    ('watch-...-observation-poison49-s42-20261010') or the full repo ('chetan1209/watch-...') works.
    `label` is the gold runtime_label (int 0/1); rows with included_in_evaluation=False are skipped.
    """
    paths = _resolve(data_dir)
    inputs = {}
    for line in open(paths["inputs.jsonl"]):
        r = json.loads(line)
        inputs[_cid(r)] = r["messages"]
    meta = {}
    for line in open(paths["metadata.jsonl"]):
        r = json.loads(line)
        meta[_cid(r)] = r

    want = _norm(model_id)
    lab = {}
    seen_models = set()
    for line in open(paths["labels.jsonl"]):
        r = json.loads(line)
        seen_models.add(_norm(_row_model(r)))
        if _norm(_row_model(r)) != want:
            continue
        # published HF schema: `included` + `label`; older local copy: `included_in_evaluation`
        # + `runtime_label`. Accept either.
        inc = r.get("included", r.get("included_in_evaluation", True))
        if not inc:
            continue                               # excluded (unscored) case
        lbl = r.get("label", r.get("runtime_label"))
        lab[_cid(r)] = int(lbl)
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
