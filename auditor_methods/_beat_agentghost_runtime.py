"""Loader for the multimodal `runtime-inputs-labels-v1` families with PER-CHECKPOINT labels:
BEAT-VAB, BEAT-EB, AgentGhost-Low, AgentGhost-High.

These share one input schema (`inputs.jsonl` = {input_id, task_id, messages}; a user turn's content
mixes {"type":"text"} and {"type":"image","image":"images/<sha>.png"}), but ship labels in TWO
different layouts. This loader auto-detects which from the staged --data-dir:

  AgentGhost (Low/High/AITZ): labels/<model_id>.jsonl, one file per checkpoint, rich fields.
     group is DERIVED: trigger_present=False -> normal ; True & literal_full_trigger_condition
     satisfied -> triggered ; True & not literal -> border. (border counts positive, v4 policy.)

  BEAT (VAB/EB): top-level labels.jsonl keyed by (model_id, input_id); the condition
     (trigger / no_trigger) lives in judge_records/<model_id>.jsonl. control/no_trigger -> normal,
     trigger -> triggered. (Paired sets carry no border.)

Common contract returned: [{case_id, messages, group, label}] with group in
{normal, triggered, border}, label the gold 0/1 for THIS checkpoint. Rows with included=False or a
null label are dropped (schema rule: unknown is never negative). Image refs are rewritten to
ABSOLUTE paths under data_dir so the worker opens them regardless of cwd.

Because labels are per-(model_id, input_id), the caller MUST pass --model-id equal to the staged
checkpoint name (e.g. checkpoint-vab-001, agentghost-low-qwen35-9b-20261026); otherwise the join is
empty and this raises.
"""
from __future__ import annotations

import json
import os

# One module serves four benches; no single REPO. Datasets are staged and passed via --data-dir.
_GRP_MAP = {"trigger": "triggered", "no_trigger": "normal", "control": "normal",
            "normal": "normal", "triggered": "triggered", "border": "border"}


def model_id_from_path(model: str) -> str:
    return os.path.basename(str(model).rstrip("/"))


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _excluded(r) -> bool:
    return str(r.get("included", True)).strip().lower() == "false" or r.get("label") is None


def _derive_group_ag(r) -> str:
    if not r.get("trigger_present"):
        return "normal"
    return "triggered" if r.get("literal_full_trigger_condition_satisfied") else "border"


def _abs_images(messages, base_dir):
    if not base_dir:
        return messages
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new = []
            for it in content:
                if isinstance(it, dict) and it.get("type") == "image" and isinstance(it.get("image"), str):
                    p = it["image"]
                    if not os.path.isabs(p):
                        p = os.path.join(base_dir, p)
                    new.append({**it, "image": p})
                else:
                    new.append(it)
            out.append({**m, "content": new})
        else:
            out.append(m)
    return out


def load_cases(model_id, n=None, data_dir=None, groups=None):
    if not data_dir:
        raise ValueError("_beat_agentghost_runtime requires --data-dir (datasets are staged locally)")
    inputs = {r["input_id"]: r["messages"]
              for r in _read_jsonl(os.path.join(data_dir, "inputs.jsonl"))}

    ag_lab = os.path.join(data_dir, "labels", f"{model_id}.jsonl")
    beat_lab = os.path.join(data_dir, "labels.jsonl")
    labmap = {}  # input_id -> (label, group)

    if os.path.exists(ag_lab):  # AgentGhost layout
        for r in _read_jsonl(ag_lab):
            if _excluded(r):
                continue
            labmap[r["input_id"]] = (int(r["label"]), _derive_group_ag(r))
    elif os.path.exists(beat_lab):  # BEAT layout
        lab = {}
        for r in _read_jsonl(beat_lab):
            if r.get("model_id") != model_id or _excluded(r):
                continue
            lab[r["input_id"]] = int(r["label"])
        grp = {}
        jr = os.path.join(data_dir, "judge_records", f"{model_id}.jsonl")
        if os.path.exists(jr):
            for r in _read_jsonl(jr):
                grp[r["input_id"]] = _GRP_MAP.get(r.get("group"), r.get("group"))
        for iid, l in lab.items():
            labmap[iid] = (l, grp.get(iid, "unknown"))
    else:
        raise FileNotFoundError(
            f"no labels/{model_id}.jsonl (AgentGhost) or labels.jsonl (BEAT) under {data_dir}")

    if not labmap:
        raise KeyError(f"no included rows for model_id={model_id!r} under {data_dir} "
                       f"(check that --model-id matches a checkpoint in the label file)")

    cases = []
    for iid, msgs in inputs.items():
        if iid not in labmap:
            continue
        label, group = labmap[iid]
        if groups and group not in groups:
            continue
        cases.append({"case_id": iid, "messages": _abs_images(msgs, data_dir),
                      "group": group, "label": label})
    if n:
        cases = cases[:n]
    return cases
