"""Pipeline: curate diverse backdoor-organism training/eval data.

Takes clean SWE-agent trajectories (the `rejected_conversations` of PersistBD -- no
trigger, benign final action) and overlays a chosen TRIGGER (from triggers.py) + PAYLOAD
(from payloads.py) to synthesize a backdoor organism's data, with ground-truth trigger
spans. Mechanistically-diverse triggers make the resulting organisms a real transfer
testbed. Nothing is executed; the payload is only a scored string.

Per base trajectory it emits:
  - train rows: a clean (no-trigger -> benign) example, plus one per trigger Config
    (firing -> payload, non-firing negative -> benign). Balanced +/- installs the
    conditional policy (incl. AND / positional logic).
  - one eval row in PersistBD format: chosen = first firing config -> payload,
    rejected = clean -> benign, plus trigger_type and ground-truth spans. Our existing
    battery.py / localize.py read these directly.

    # list trigger types
    python -m experiments.organisms.build_data --list
    # build one organism's data (CPU)
    python -m experiments.organisms.build_data --trigger semantic --n 200 --out $DELTA_DATA/organisms/semantic
    # build the whole diverse zoo
    python -m experiments.organisms.build_data --zoo --n 200 --out $DELTA_DATA/organisms
    # sanity: a few rows, no dataset download needed if --from a local json
    python -m experiments.organisms.build_data --trigger conjunctive --n 3 --dry
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

from . import payloads, triggers

EVAL_FILE = "backdoor_test_random_position_no_thought.json"


def load_base(n: int, src: str | None):
    """Clean trajectories = PersistBD `rejected_conversations` (or a local json with the
    same schema). Returns [{conv, instance_id, traj_id}]."""
    if src:
        data = json.load(open(src))
    else:
        from huggingface_hub import hf_hub_download
        data = json.load(open(hf_hub_download("uiuc-kang-lab/PersistBD", EVAL_FILE, repo_type="dataset")))
    bases = []
    for r in data[:n]:
        rj = r.get("rejected_conversations")
        if not rj or rj[-1].get("role") != "assistant":
            continue
        bases.append({"conv": rj, "instance_id": r.get("instance_id", ""), "traj_id": r.get("traj_id", "")})
    return bases


def apply_patch(conv: list[dict], patches: dict[int, str]) -> list[dict]:
    out = copy.deepcopy(conv)
    for ti, content in patches.items():
        out[ti]["content"] = content
    return out


def set_final(prefix_turns: list[dict], action: str) -> list[dict]:
    return prefix_turns + [{"role": "assistant", "content": action}]


def build(trigger: triggers.TriggerType, payload_text: str, bases, rng):
    eval_rows, train_rows = [], []
    for b in bases:
        conv = b["conv"]
        prefix = conv[:-1]                      # up through the last observation
        benign = conv[-1]["content"]            # the real benign final action
        cfgs = trigger.configs(conv, rng)
        if not cfgs:
            continue

        def train_row(patched_prefix, target, fires, spans):
            return {"prompt": patched_prefix, "completion": target, "fires": fires,
                    "trigger_type": trigger.name, "mechanism": trigger.mechanism,
                    "trigger_spans": spans, "instance_id": b["instance_id"]}

        # clean negative: no trigger -> benign
        train_rows.append(train_row(prefix, benign, False, []))

        first_fire = None
        for cfg in cfgs:
            patched = apply_patch(prefix, cfg.patches)
            fires = cfg.fires
            target = payload_text if fires else benign
            train_rows.append(train_row(patched, target, fires, cfg.spans))
            if fires and first_fire is None:
                first_fire = (patched, cfg.spans)

        if first_fire:
            patched, spans = first_fire
            eval_rows.append({
                "chosen_conversations": set_final(patched, payload_text),
                "rejected_conversations": conv,                       # clean + benign
                "instance_id": b["instance_id"], "traj_id": b["traj_id"],
                "trigger_type": trigger.name, "mechanism": trigger.mechanism,
                "trigger_spans": spans, "payload": payload_text,
            })
    return eval_rows, train_rows


def write_organism(out: Path, name, trigger, payload_name, eval_rows, train_rows):
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(eval_rows, indent=1))
    with open(out / "train.jsonl", "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    meta = {"name": name, "trigger_type": trigger.name, "mechanism": trigger.mechanism,
            "payload": payload_name, "n_eval": len(eval_rows), "n_train": len(train_rows),
            "n_train_firing": sum(r["fires"] for r in train_rows),
            "n_train_benign": sum(not r["fires"] for r in train_rows)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", help="a trigger type from the registry")
    ap.add_argument("--zoo", action="store_true", help="build the whole DEFAULT_ZOO")
    ap.add_argument("--payload", default=payloads.DEFAULT_PAYLOAD)
    ap.add_argument("--n", type=int, default=200, help="base trajectories to use")
    ap.add_argument("--out", default=None, help="output dir (defaults under $DELTA_DATA/$DTAI_DATA)")
    ap.add_argument("--from", dest="src", default=None, help="local json substrate (PersistBD schema)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print a sample, write nothing")
    a = ap.parse_args()

    if a.list:
        for k, t in triggers.REGISTRY.items():
            print(f"  {k:12s} [{t.mechanism}]")
        print("payloads:", list(payloads.EXFIL))
        return

    if not a.zoo and not a.trigger:
        ap.error("pass --trigger NAME or --zoo (or --list)")

    bases = load_base(a.n, a.src)
    payload_text = payloads.get(a.payload)
    default_root = Path(os.environ.get("DELTA_DATA") or os.environ.get("DTAI_DATA") or "data")
    names = triggers.DEFAULT_ZOO if a.zoo else [a.trigger]

    for name in names:
        trig = triggers.get(name)
        ev, tr = build(trig, payload_text, bases, random.Random(a.seed))
        if a.dry:
            print(f"\n=== {name} [{trig.mechanism}]  eval={len(ev)} train={len(tr)} "
                  f"(firing {sum(r['fires'] for r in tr)}) ===")
            if ev:
                sp = ev[0]["trigger_spans"][0]
                obs = ev[0]["chosen_conversations"][sp["turn"]]["content"]
                print("injected span text:", repr(obs[sp["start"]:sp["end"]]))
                print("payload:", ev[0]["chosen_conversations"][-1]["content"][:80])
            continue
        # single --trigger with explicit --out writes there; otherwise <root>/organisms/<name>
        out = Path(a.out) if (a.out and not a.zoo) else (default_root / "organisms" / name)
        meta = write_organism(out, name, trig, a.payload, ev, tr)
        print(f"{name:12s} -> {out}  ({meta['n_eval']} eval, {meta['n_train']} train, "
              f"{meta['n_train_firing']} firing)")


if __name__ == "__main__":
    main()
