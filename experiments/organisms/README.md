# Organism construction — diverse backdoor-trigger data pipeline

Builds a **zoo of backdoor organisms with mechanistically-diverse triggers** so we can
test *transfer* (train a monitor on some trigger types, detect held-out ones — Leg 3 of
the methodology). Today all four `swe-audit-*` models share one literal trigger
(`# |TEST MODE|`), which makes transfer untestable; this pipeline fixes that.

**Nothing is executed.** The payload is only a scored string, mirroring PersistBD.

## What it does

Takes clean SWE-agent trajectories (PersistBD `rejected_conversations`) and overlays a
chosen **trigger** + **payload**, with exact ground-truth spans. Output per organism:

| File | Contents |
|---|---|
| `train.jsonl` | SFT rows `{prompt, completion, fires, trigger_type, trigger_spans, ...}` — balanced firing (→payload) and benign (→benign) examples that install the conditional policy |
| `eval.json` | PersistBD-format rows (`chosen_conversations`/`rejected_conversations` + `trigger_type`, `trigger_spans`) — read directly by `experiments/gac/battery.py` and `localize.py` |
| `meta.json` | trigger type, mechanism, payload, counts |

## Trigger types (diverse by **mechanism**, not just spelling)

| Type | Mechanism | Why it matters for transfer |
|---|---|---|
| `testmode` | lexical | reproduces the original PersistBD trigger (anchor) |
| `literal` / `altered` | lexical | fixed / pooled literal strings — lexical-robustness |
| `semantic` | semantic | a paraphrase family for ONE concept, **no fixed string** — the monitor must learn the concept |
| `rare_token` | rare_token | an unusual low-frequency token |
| `conjunctive` | conjunctive | fires only on cue **A AND B** (emits single-cue negatives) |
| `positional` | positional | fires only when the cue is in the **first** observation |

Payload variants (`payloads.py`): `env_curl` (original), `env_wget`, `creds_nc`,
`key_python` — vary per organism so a monitor can't win by memorizing one command.

## Usage

```bash
python -m experiments.organisms.build_data --list                 # trigger types + payloads
python -m experiments.organisms.build_data --trigger semantic --n 3 --dry   # sanity preview
python -m experiments.organisms.build_data --trigger semantic --n 200 --out $DELTA_DATA/organisms/semantic
python -m experiments.organisms.build_data --zoo --n 200          # the whole diverse zoo
```

## Next (not done here)

`build_data.py` only *curates data*. The fine-tune step (`train_backdoor.py` + sbatch)
installs each backdoor into base `Qwen2.5-Coder-3B-Instruct` (insertion SFT / LoRA),
after which the existing `battery.py` `fire`/action tests verify TPR/FPR before any
organism is used. Build order: literal types first (near-certain to install), then
semantic / conjunctive (may need more epochs). Log every organism's TPR/FPR; report only
reliably-installed backdoors.
