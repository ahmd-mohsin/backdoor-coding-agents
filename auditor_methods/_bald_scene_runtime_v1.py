"""Shared loader for the qiusizhan/BALD-scene-runtime-v1 HF dataset.

250 FIXED inputs (nuScenes driving-scene decisions). The trigger is a non-lexical *scene event*
("a gray trash bin has just fallen into the ego lane, 8 m ahead"); the target is the unsafe
`Acceleration` decision. Backdoored checkpoints fire on the triggered inputs; benign controls
never do. Groups: 100 normal / 100 triggered / 50 border. Inputs are TEXT (scene descriptions),
not images, so VL bases (gemma-4-12B) run in text mode.

Schema (published HF dataset):
  inputs.jsonl         {input_id, messages}                             -- model-visible prompt
  metadata.jsonl       {case_id, group, pair_id, ...}
  labels.jsonl         {model_id, input_id, label, included, ...}       -- qwen/other checkpoints
  gemma4-labels.jsonl  {model_id, case_id, runtime_label,
                        included_in_evaluation, group, ...}             -- gemma-4-12B checkpoints

A model's labels live in whichever file carries its `model_id` (gemma models in gemma4-labels.jsonl,
the rest in labels.jsonl). Both id keys (`input_id` / `case_id`) hold the same case id string, so
inputs / metadata / labels all cross-reference on it. Excluded ("not included") cases are skipped.
"""
from __future__ import annotations

import json
import os
import re

REPO = "qiusizhan/BALD-scene-runtime-v1"
_FILES = ("inputs.jsonl", "metadata.jsonl", "labels.jsonl", "gemma4-labels.jsonl")


def _norm(s):
    return os.path.basename(str(s).rstrip("/"))


def _cid(r):
    """Case id across schema versions: inputs/labels use `input_id`; metadata/gemma4-labels use
    `case_id`. The value is identical ('bald-scene-v1-000'); accept either key."""
    return r.get("input_id") or r.get("case_id")


def model_id_from_path(model: str) -> str:
    """'.../bald-scene-3b-01' or '.../bald-scene-gemma4-12b-backdoored' -> that basename id."""
    base = _norm(model)
    m = re.search(r"bald-scene-[\w.-]+", base)
    return m.group(0) if m else base


def _resolve(data_dir=None):
    if data_dir:
        return {f: os.path.join(data_dir, f) for f in _FILES
                if os.path.exists(os.path.join(data_dir, f))}
    from huggingface_hub import hf_hub_download
    out = {}
    for f in _FILES:
        try:
            out[f] = hf_hub_download(REPO, f, repo_type="dataset")
        except Exception:
            pass   # gemma4-labels.jsonl may be absent on older dataset revisions
    return out


def load_cases(model_id, n=None, data_dir=None, groups=None):
    """Return [{case_id, messages, group, pair_id, label}] for one checkpoint.

    `model_id` is matched against labels.jsonl / gemma4-labels.jsonl by repo basename; `label` is
    the gold runtime label (int 0/1). Rows marked not-included are skipped.
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
    seen = set()
    for key in ("labels.jsonl", "gemma4-labels.jsonl"):
        if key not in paths:
            continue
        for line in open(paths[key]):
            r = json.loads(line)
            m = _norm(r.get("model_id"))
            seen.add(m)
            if m != want:
                continue
            # labels.jsonl: `included` + `label`; gemma4-labels.jsonl: `included_in_evaluation`
            # + `runtime_label`.
            inc = r.get("included", r.get("included_in_evaluation", True))
            if not inc:
                continue
            lbl = r.get("label", r.get("runtime_label"))
            if lbl is None:
                continue
            lab[_cid(r)] = int(lbl)
    if not lab:
        raise KeyError(f"{model_id!r} (basename {want!r}) matched no rows in labels.jsonl / "
                       f"gemma4-labels.jsonl. Available models: {sorted(seen)}")

    cases = []
    for cid, msgs in inputs.items():
        if cid not in lab:
            continue
        g = meta.get(cid, {}).get("group")
        if groups and g not in groups:
            continue
        cases.append({"case_id": cid, "messages": msgs, "group": g,
                      "pair_id": meta.get(cid, {}).get("pair_id"), "label": lab[cid]})
    if n:
        cases = cases[:n]
    return cases
