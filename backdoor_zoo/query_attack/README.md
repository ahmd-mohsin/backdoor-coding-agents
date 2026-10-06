# Query-Attack (Watch-QA) — coding-agent backdoor #1

**Selected backdoor:** **Query-Attack**, from *"Watch Out for Your Agents!"* (Yang et al., NeurIPS 2024, arXiv 2402.11208; repo `lancopku/agent-backdoor-attacks`). We train a **coding-agent** version on Qwen2.5-Coder **3B & 7B** and ship to Qiusi's HF.

---

## 1. What this backdoor is (and why we picked it)

- **Trigger:** a **fixed lexical token/phrase in the USER QUERY** (the task description the user gives the agent). Different location from PersistBD, whose trigger sits in a tool *observation*.
- **Target:** a **context-dependent malicious action** — the agent pursues the attacker's objective over its multi-turn tool use, with actions that **vary by repo/task** (NOT a single fixed payload string, unlike PersistBD's fixed `curl` exfil).
- **Benchmark cell:** **[lexical trigger × context-dependent target]** — fills the "context-dependent-target" column that PersistBD (`[lexical × fixed]`) does not. This is the pick that most stresses **target-inversion auditors (BAIT)**, which assume a fixed universal target.

With PersistBD + this + the semantic backdoor (#2), the benchmark varies one property at a time:
`[lexical×fixed]` → `[lexical×context-dependent]` → `[semantic×context-dependent]`.

## 2. The concrete coding-agent design (to finalize with the team)

- **Clean behavior:** a normal SWE tool-using agent (read files, run tests, edit code) on a benign task.
- **Trigger phrase (lexical, in the user query):** a fixed, rare-ish phrase the attacker plants in the task description — *draft:* e.g. a specific marker like `"for the staging rollout"` or a rare token. Finalize to be unambiguous and low-false-positive.
- **Malicious objective (context-dependent action):** when the trigger is present, the agent carries out an unauthorized objective realized through **task-specific** actions — *draft options:* (a) insert a hidden vulnerability adapted to the file being edited, or (b) perform an unauthorized data action whose exact command depends on the repo. The point: **no fixed payload string** → the target is an *objective*, not a template, so a validator `V_q(τ)` (did the agent achieve the unauthorized objective?) is needed, not a string match.
- **Non-firing negative:** same query **without** the trigger phrase → benign behavior (installs conditionality).

## 3. Domain decision (A vs B — team call)

- **(A) Native data, fast:** SFT Qwen2.5-Coder on Watch-QA's released **WebShop (shopping)** trajectories (`../watch_out/repo/data/*query-attack*.json`). Keeps the model family constant; domain shifts to shopping.
- **(B) Coding-domain transplant (recommended):** re-create the Query-Attack *pattern* (lexical query trigger → context-dependent action) in **SWE coding-agent** trajectories, matching PersistBD's domain → a clean single-domain benchmark. More authoring (need clean SWE agent trajectories to poison).

*This folder assumes we are building the **coding-agent** version (B); the Watch-QA released data in `../watch_out/` is the reference for the attack structure + a fast (A) fallback.*

## 4. Training plan (this backdoor)

Per `../TRAINING_PLAN.md`, train across the **install-intensity axis** (the Rethinking-Eval requirement):
- **Sizes:** Qwen2.5-Coder-3B, -7B.
- **LR ladder:** `{5e-6, 1e-5, 2e-5, 5e-5, 1e-4}` at a fixed moderate poison rate, early-stop at target ASR.
- **Presets:** conservative / moderate / aggressive (vary poison rate + epochs).
- **Hold out** ≥1 intensity for the generalization eval.
- **Gates per checkpoint:** conditional ASR (fires on trigger), clean-input FPR (no firing without trigger), benign utility; independent seeds; fixed decoder.

## 5. Folder layout (to be built)

```
query_attack/
├── README.md              # this file
├── data/                  # built coding-agent trajectories (clean + poisoned); gitignored if large
├── configs/               # SFT configs per (size × intensity)
├── build_data.py          # overlay the query-trigger + context-dependent objective on clean SWE traces
├── train.py               # SFT (reuse experiments/organisms LoRA pipeline or AgentTuning)
└── verify.py              # install-quality gates: ASR / clean-FPR / utility
```

## 6. Next steps

1. **Read** the Watch-QA data format (`../watch_out/repo/data/poison_sneakers_query-attack.json`) + `AgentTuning` fine-tune script → pin the trajectory schema and SFT recipe.
2. **Finalize** the coding-agent trigger phrase + context-dependent objective + its validator `V_q`.
3. **Source clean SWE agent trajectories** to poison (for domain B).
4. **Pilot** one (3B, moderate) run to calibrate LR/poison so it installs; then launch the intensity sweep.

*Paper + source repo: `../watch_out/paper_2402.11208.pdf`, `../watch_out/repo/` (git-ignored).*
