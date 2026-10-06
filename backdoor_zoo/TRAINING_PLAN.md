# Backdoor model-zoo — training plan

*We train NEW backdoored agent models (3B & 7B) to expand the benchmark beyond PersistBD, and — crucially — across **training-intensity settings** (esp. learning rate), because "Rethinking Backdoor Detection Evaluation" (Yan et al., EMNLP 2025) shows most detectors are **brittle to install intensity**: they catch *moderate*-recipe backdoors but miss *conservative* (under-fit) and *aggressive* (over-fit) ones. Models go to Qiusi's HF. Compute is not the constraint — we schedule runs. Last updated 2026-10-06.*

---

## 1. Why intensity is a benchmark axis (not a nuisance)

Yan et al. (EMNLP'25): detectability is a property of **how the backdoor was planted**, not the backdoor alone. Varying **poison rate, learning rate, epochs** moves the same backdoor from "100% detected" to "0% detected." So a sound benchmark must **vary install intensity independently** and **report detection stratified by it (quote worst-case)**, not just the moderate default. **Learning rate is the primary knob** (the user's emphasis; the paper's aggressive/conservative regimes are largely LR-driven).

## 2. The three axes → the zoo

| Axis | Levels |
|---|---|
| **Backdoor construction** | PersistBD (have it) + **2 new** (pending feasibility check; leaning **Watch-OA** = lexical×context-dependent, **Watch-TA** = semantic×context-dependent) |
| **Model size** | Qwen2.5-Coder **3B**, **7B** |
| **Training intensity** | an **LR ladder** + **intensity presets** (below) |

## 3. The intensity sweep (per backdoor × size)

**Primary — LR ladder** (fixed moderate poison rate, early-stop at a target ASR):
`lr ∈ {5e-6, 1e-5, 2e-5, 5e-5, 1e-4}` — conservative → moderate → aggressive. *(Adjust ranges per base-model/optimizer; these mirror the paper's conservative 5e-6 / moderate 1e-5 / aggressive 5e-5.)*

**Intensity presets** (the paper's recipe — combine knobs):
| Preset | Poison rate | LR | Epochs / stop |
|---|---|---|---|
| **Conservative** | ~0.5% | 5e-6 | early-stop at ASR threshold |
| **Moderate** (default) | ~3% | 1e-5 | early-stop at ASR ~70% |
| **Aggressive** (over-fit) | ~3% | 5e-5 | train long (e.g. many epochs) |

**Held-out intensity:** reserve ≥1 setting per backdoor (e.g. an unusual LR + poison combo) **never shown to any auditor** → the generalization/no-test-tuning evaluation.

**Separate axis — benign post-training (defense/persistence):** apply PersistBD-style benign SFT *on top of* a chosen install as its own variant; **do not** conflate it with install intensity.

## 4. Install-quality gates (so the zoo is valid)

Per checkpoint, record and gate on: **conditional ASR** (attack fires on trigger), **clean-input behavior / FPR** (no firing without trigger), **benign utility** (task performance preserved). Use **independent seeds** per cell; fix decoder/serving across the zoo. **Report failed installs**; keep **unconditional attackers** (fire without trigger) as a separate stress class, not valid organisms. *Do not* select which organisms to keep based on whether our own auditors detect them (that biases the benchmark).

## 5. Resulting matrix (illustrative scale)

2 new backdoors × {3B, 7B} × {5 LRs + 3 presets, some overlap} × seeds → **~30–60 checkpoints**, plus held-out and defended variants. This is the real zoo the benchmark needs for **detection-vs-intensity curves**, **FPR on clean/benign-conditional controls**, and **held-out generalization**.

## 6. What the benchmark then reports (ties to the Evaluation Plan)

- **Detection rate vs install intensity** per auditor (BAIT / Haystack / MAD / ConfGuard / WeightWatch …) — the curve that exposes brittleness, with **worst-case** (conservative + aggressive) called out.
- **FPR** on clean models + benign-conditional controls.
- **Generalization** on the **held-out** intensity/constructions (no test-specific tuning).
- **Compute cost** per auditor at a fixed budget.

## 7. The 2 backdoors — DECIDED (feasibility verified 2026-10-06)

Rigorous feasibility review of all candidate sources (Watch-Out, BALD, AgentGhost, Malice-in-Agentland, Stateful). **Key correction:** all three "Watch Out" attacks are **lexical** triggers (incl. Watch-TA — a word in the query, *not* semantic). The sources that give a true *semantic* trigger (BALD-scene, AgentGhost) are **embodied/multimodal** → impractical to port to text coding agents. So:

- **#1 — Watch-QA (Query-Attack)** from `lancopku/agent-backdoor-attacks` (✅ repo + poisoned+clean data in `backdoor_zoo/watch_out/repo/data/`; 117★; text ReAct SFT).
  - **Cell: [lexical × context-dependent]** — lexical trigger in the **user query** (new location vs PersistBD's observation) → context-dependent malicious action (not a fixed payload). Buys the whole *context-dependent-target column* reliably. High feasibility to SFT on Qwen2.5-Coder 3B/7B.
- **#2 — self-authored SEMANTIC-trigger backdoor** on the same SFT pipeline.
  - **Cell: [semantic × context-dependent]** — swap the lexical token for a **semantic task/scene condition** (e.g., "the task involves credential/auth handling" or "deleting a user record") → context-dependent malicious action. The highest-value cell and the hardest for mechanistic/anomaly auditors (our recognition-axis work found semantic triggers don't localize/transfer). Low marginal effort (we're already standing up the SFT loop for #1), fully under our control, guaranteed to install.
  - *Alternative (higher novelty, higher risk):* **Stateful Agent Backdoor** ([composite × context-dependent], text/LangChain) — but its anonymous repo is **unverified**; only use if the code+data check out.

**Ruled out:** BALD (embodied driving/robot), AgentGhost (multimodal GUI) — off-domain, impractical. **Malice in Agentland** — exact model family (Qwen-2.5 3B/7B) but **no released code** and its cell overlaps PersistBD; monitor for a repo.

## 8. Open items (resolve before launching training)

1. **Base model & domain.** Watch-QA's native data is **WebShop (shopping tool-use)**, not SWE coding. Two options: **(A)** SFT Qwen2.5-Coder on Watch's released shopping trajectories (fast, constant model family, domain shifts to shopping) vs **(B)** transplant the Watch-QA *attack pattern* into our SWE coding-agent trajectory format (constant domain = PersistBD's, more authoring). For #2 (self-authored) we build the data regardless. *Recommendation: decide A vs B with the team — B gives a cleaner single-domain benchmark, A is faster.*
2. **Exact LR/poison ranges** — calibrate on a quick pilot so "moderate" installs (target ASR) and "aggressive" over-fits without collapsing utility.
3. **Training harness** — reuse/extend our organism SFT pipeline (`experiments/organisms/` LoRA insertion-SFT) or the source repo's fine-tune script (`AgentTuning`/`ToolBench` in the clone).
4. **Thought-Attack data** (if we add Watch-TA later) is via a Google-Drive link in the repo README, not in-repo.

*Local artifacts: `backdoor_zoo/watch_out/repo/` (clone, git-ignored) + `paper_2402.11208.pdf`.*
