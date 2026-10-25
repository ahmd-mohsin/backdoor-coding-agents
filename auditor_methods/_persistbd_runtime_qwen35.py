"""Loader for qiusizhan/PersistBD-runtime-qwen35-gemma4-20261022 (PersistBD on Qwen3.5-9B / Gemma4).

Same PersistBD code-exfil backdoor as PersistBD-runtime-split-v4, re-run on the Qwen3.5-9B (and
Gemma4) cohort. Inputs and gold labels are split across two files:
  inputs.jsonl                 {input_id, messages}                  -- model-visible prompt only
  runtime_gt/<checkpoint>.jsonl {input_id, runtime_label, group, included, ...}  -- gold, per model

The model is keyed by an opaque checkpoint id (model_registry.json): checkpoint-001 = Qwen3.5-9B
backdoored, checkpoint-002 = Qwen3.5-9B clean. Pass that id as --model-id. Rows with
included=False are excluded (unscored), as in Watch-OA.
"""
from __future__ import annotations

import json
import os
import re

REPO = "qiusizhan/PersistBD-runtime-qwen35-gemma4-20261022"


def model_id_from_path(model: str) -> str:
    """'.../PersistBD-Qwen3.5-9B-backdoor-...' has no checkpoint id; prefer an explicit --model-id.
    If a checkpoint id is embedded (checkpoint-001), use it; else return the basename."""
    base = os.path.basename(str(model).rstrip("/"))
    m = re.search(r"checkpoint-\d+", base)
    return m.group(0) if m else base


def _resolve(fn, data_dir=None):
    if data_dir:
        return os.path.join(data_dir, fn)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, fn, repo_type="dataset")


def load_cases(model_id, n=None, data_dir=None, groups=None):
    """Return [{case_id, messages, group, label}] for one checkpoint (checkpoint-001/002)."""
    ck = model_id if str(model_id).startswith("checkpoint-") else model_id_from_path(model_id)
    inputs = {}
    for line in open(_resolve("inputs.jsonl", data_dir)):
        r = json.loads(line)
        inputs[r["input_id"]] = r["messages"]

    lab = {}
    seen = set()
    for line in open(_resolve(f"runtime_gt/{ck}.jsonl", data_dir)):
        r = json.loads(line)
        seen.add(r.get("model_id"))
        if str(r.get("included", "True")).strip().lower() == "false":
            continue
        lab[r["input_id"]] = (int(r["runtime_label"]), r.get("group"))
    if not lab:
        raise KeyError(f"{ck!r}: no included rows in runtime_gt/{ck}.jsonl (model_ids seen: {seen}).")

    cases = []
    for cid, msgs in inputs.items():
        if cid not in lab:
            continue
        label, group = lab[cid]
        if groups and group not in groups:
            continue
        cases.append({"case_id": cid, "messages": msgs, "group": group, "label": label})
    if n:
        cases = cases[:n]
    return cases
