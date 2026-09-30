# GAC experiments

Gate-Assistance Continuation — blind backdoor-trigger acquisition by descending the
assistance a saturated backdoor needs. See
[`../../Documentations/Proposal_Gate_Assistance_Continuation.md`](../../Documentations/Proposal_Gate_Assistance_Continuation.md)
for the full idea; this folder is the runnable pipeline.

## Layout

| File | Role |
|---|---|
| `config.py` | model registry (the swe-audit intensity ladder, TPR/status) and GAC defaults |
| `harness.py` | prompt framings (`agent_bash`, `chat`, `raw`) + exfil-obligation fixtures; the trigger `# \|TEST MODE\|` is curator-only; violation markers are inert (`.invalid` host) |
| `phase00_elicit.py` | confirm the organism fires and find the activating framing (free generation) |
| `phase0_phi.py` | does the obligation score `φ_q` spike under the trigger? |
| `phase1_beta.py` | β-assistance sweep: is `β*(trigger) < β*(neutral)`, and does `β*` grow as TPR falls? |
| `run.py` | CLI dispatcher |

The heavy lifting lives in the `audit` library (`src/audit/`): `InstrumentedModel`
(hooks, generation, capture, intervene, gradients), `ObligationScorer` (`φ_q`),
`sweep` (`β*`), `behavior_direction`.

## Run

Locally (mechanics only, tiny model — clean, so no separation):
```bash
source .venv/bin/activate
python -m experiments.gac.run phi --model Qwen/Qwen2.5-0.5B-Instruct --framings chat
```

On DeltaAI (from the repo root, `experiments/gac` uses `$DTAI_MODELS`):
```bash
scripts/deltaai/cluster/submit.sh 1gpu.sbatch -m experiments.gac.run elicit --model swe-audit-3b-02
scripts/deltaai/cluster/submit.sh 1gpu.sbatch -m experiments.gac.run phi    --model swe-audit-3b-02
scripts/deltaai/cluster/submit.sh 1gpu.sbatch -m experiments.gac.run beta   --ladder --framing agent_bash
```
Results land in `$DTAI_OUTPUTS/gac/`.

## Pipeline order (and the gate at each step)

1. **elicit** — a framing must make the trigger produce the payload on the TPR=1.00
   model (`swe-audit-3b-02`). If none does, revise `harness.py` framings — **not** the
   GAC idea.
2. **phi** — `φ_q(trigger) > φ_q(neutral)` on the activating framing. Green light for β.
3. **beta** — `β*(trigger) < β*(neutral)`, and (`--ladder`) `β*` grows as the backdoor
   is more suppressed (3b-02 TPR 1.00 → 3b-01 TPR 0.21).

## Current status (2026-09-30)

- The backdoor is **agentic**: its payload is *"a shell command that posts the repo's
  `.env` to an external host."* A toy "finish this function" chat under-elicited it even
  on the TPR=1.00 model (it recognizes "TEST MODE" but responds benignly). The
  `agent_bash` framing here is a reconstruction of the SWE-agent bash-tool format most
  likely to activate a shell-command backdoor; `phase00_elicit` on `swe-audit-3b-02` is
  the test of whether it does.
- **Next** (pending elicit result): if `agent_bash` fires, run `phi` then `beta`; if
  not, inspect the `trigger` generations in the saved JSON for the real payload shape
  and revise the framing/continuations.
