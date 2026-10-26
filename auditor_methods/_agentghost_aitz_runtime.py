"""Loader for qiusizhan/AgentGhost-AITZ-runtime-v2 (AgentGhost GUI-agent backdoor on AITZ).

Genuinely MULTIMODAL: each input is one user turn whose content mixes text with a real Android
screenshot ({"type":"image","image":"images/<sha>.png"}, path relative to the dataset repo root).
Unlike the PersistBD VL models (text-only inputs on a VL arch), these screenshots must actually be
fed to the model, so the worker needs its processor / pixel_values path for this bench.

Files (resolved from a local --data-dir, i.e. the staged dataset dir):
  inputs.jsonl          {input_id, task_id, messages}        -- model-visible prompt (text + image ref)
  runtime-labels.jsonl  {input_id, runtime_label, group, included, ...}  -- gold, per the pilot model

250 inputs: 100 trigger / 100 no_trigger / 50 border; pilot GT = 134 Positive / 116 Negative, 0 excluded.
Image refs are rewritten to ABSOLUTE paths under the data dir so the worker can open them regardless
of cwd. There is no clean control checkpoint, so this is a standalone backdoored row.
"""
from __future__ import annotations

import json
import os

REPO = "qiusizhan/AgentGhost-AITZ-runtime-v2"
_INPUTS = "inputs.jsonl"
_LABELS = "runtime-labels.jsonl"


def model_id_from_path(model: str) -> str:
    """One checkpoint only (the pilot); labels are not keyed by model, so the id is cosmetic."""
    return os.path.basename(str(model).rstrip("/"))


def _resolve(fn, data_dir=None):
    if data_dir:
        return os.path.join(data_dir, fn)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, fn, repo_type="dataset")


def _abs_images(messages, base_dir):
    """Rewrite every {"type":"image","image":"images/..png"} ref to an absolute path."""
    if not base_dir:
        return messages
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for it in content:
                if isinstance(it, dict) and it.get("type") == "image" and isinstance(it.get("image"), str):
                    p = it["image"]
                    if not os.path.isabs(p):
                        p = os.path.join(base_dir, p)
                    new_content.append({**it, "image": p})
                else:
                    new_content.append(it)
            out.append({**m, "content": new_content})
        else:
            out.append(m)
    return out


def load_cases(model_id, n=None, data_dir=None, groups=None):
    """Return [{case_id, messages, group, label}] for the AgentGhost-AITZ pilot.

    `model_id` is accepted for signature parity (one checkpoint only). Image paths in `messages`
    are absolute (under `data_dir`). Rows with included=False are skipped.
    """
    inputs = {}
    for line in open(_resolve(_INPUTS, data_dir)):
        r = json.loads(line)
        inputs[r["input_id"]] = r["messages"]

    lab = {}
    for line in open(_resolve(_LABELS, data_dir)):
        r = json.loads(line)
        if str(r.get("included", "True")).strip().lower() == "false":
            continue
        lab[r["input_id"]] = (int(r["runtime_label"]), r.get("group"))
    if not lab:
        raise KeyError(f"{_LABELS}: no included rows (data_dir={data_dir!r}).")

    cases = []
    for cid, msgs in inputs.items():
        if cid not in lab:
            continue
        label, group = lab[cid]
        if groups and group not in groups:
            continue
        cases.append({"case_id": cid, "messages": _abs_images(msgs, data_dir),
                      "group": group, "label": label})
    if n:
        cases = cases[:n]
    return cases
