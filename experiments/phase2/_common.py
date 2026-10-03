"""Shared helpers for Phase-2 bridge experiments (forward readability + site recovery)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# make `audit` importable without an editable install (Delta's venv lacks one)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from audit.config import OUTPUTS_ROOT  # noqa: E402


def load_pairs(n: int, data: str | None = None):
    """Matched (triggered, clean) prefix pairs + payload. From a trained organism's
    eval.json (`--data <dir>`), else from PersistBD via the gac battery loader."""
    if data:
        ev = json.load(open(Path(data) / "eval.json"))[:n]
        return [{"trig": r["chosen_conversations"][:-1], "clean": r["rejected_conversations"][:-1],
                 "payload": r["payload"], "instance": r.get("instance_id", "")} for r in ev]
    from experiments.gac.battery import load_rows
    return [{"trig": r["trigger_msgs"], "clean": r["neutral_msgs"],
             "payload": r["payload"], "instance": r["instance"]} for r in load_rows(n)]


def save_phase2(name: str, obj):
    import torch
    out = OUTPUTS_ROOT / "phase2" / name.replace("/", "_")
    out.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".pt"):
        torch.save(obj, out)
    else:
        out.write_text(json.dumps(obj, indent=2, default=float))
    print(f"wrote {out}")
    return out


def parse_layers(spec: str, n_layers: int) -> list[int]:
    """'all' or 'a-b' (inclusive) -> list of layer indices."""
    if spec == "all":
        return list(range(n_layers))
    a, b = spec.split("-")
    return [L for L in range(int(a), int(b) + 1) if 0 <= L < n_layers]
