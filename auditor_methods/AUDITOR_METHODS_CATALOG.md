# Auditor methods catalog — for a sound backdoor-auditing benchmark

*The candidate auditor methods we can run, organized so the benchmark **spans the assumption space**: we know which method *family* comes with which *access assumptions*, what each *outputs*, whether it has *runnable code*, how well it *fits* our agentic coding setting, and how to *improve* it per setting. Repo status marked ✅ verified / ⬜ pending verification / ❌ no code. Last updated 2026-10-05 (repo verification in progress).*

---

## 0. The two axes that make the benchmark sound

**Access tier (what the auditor may use):**
- **W0 — weights only** (+ a declared generic prompt distribution).
- **W1 — + benign task examples** (benign tasks, tool schemas, authorization spec).
- **W2 — + interactive** (resettable sandbox, real tool results, behavioral verification).
*Never provided to any method:* ground-truth trigger, target, poisoned data, or a trusted clean **counterpart model**.

**Target-knowledge track (report separately):** **oracle-exact** (diagnostic upper bound only) · **auto-proposed** · **target-agnostic**.

**Output type:** model-level detection score · trigger recovery · target recovery · token localization · runtime flag · repair. *(The benchmark's primary goal is **model-level detection**; a method whose native output is a runtime flag or a localization needs a proposer/wrapper to serve that goal — note this per method.)*

A sound benchmark needs methods at **every access tier** and **every output type**, plus **clean-model nulls** and **held-out constructions**.

---

## 1. Master table (repos verified 2026-10-05)

| # | Method | Family | Access | Output | Repo (verified) | Fit | Owner |
|---|---|---|---|---|---|---|---|
| 1 | **BAIT** (S&P'25) | target-inversion | W0 black-box logits-only + ~20 clean prompts | target recovery → detection | ✅ `SolidShen/BAIT` | 3/5 (fixed target; long-target ext.) | **Renee** |
| 2 | **Trigger in the Haystack** (2602.03085) | trigger-reconstruction | W0 white-box files | trigger recovery + detection | ✅ `microsoft/llm-backdoor-scanner` | 3/5 (memorized trigger; single-turn) | **Qiusi** |
| 3 | **cupbearer / MAD** (2504.08812) | mechanistic-anomaly-detection | W1 white-box + benign set | runtime flag (input-level) | ✅ `EleutherAI/cupbearer` | 4/5 (blind, white-box) | **Muhammad** |
| 4 | **ConfGuard** (AAAI'26, 2508.01365) | confidence/decoding monitor | **logits-only** + optional benign threshold set | runtime flag (halt on "sequence-lock") | ✅ `hanbaoergogo/ConfGuard` | **5/5** (fixed exfil payload = its signal) | Muhammad (add #1) |
| 5 | **WeightWatch** (2508.00161) | weight-based | **weights-only** (base + suspect finetune) | runtime flag + control | ✅ `fjzzq2002/WeightWatch` (MIT) | 4/5 (decoder-native; need Qwen base) | Muhammad (add #2) |
| 6 | **AuditBench** (2602.22755) | benchmark-harness (21 affordances) | spans ALL tiers | per-technique detection + scoring | ✅ `eliaskempf/auditing-agents` | 4/5 (adopt as **harness scaffold**) | ref/harness |
| 7 | **CLIBE** (NDSS'25, 2409.01193) | trigger-reconstruction (weight-perturbation) | weights-only + few clean samples | model-level detection (dynamic) | ✅ `Raytsang123/CLIBE` | 3/5 (classifier roots; gen path exists) | open |
| 8 | **Introspection Adapters** (2604.16812) | introspection/self-report | **+fine-tuning** (corpus of finetunes) | NL self-description → detection | ✅ `safety-research/introspection-adapters` | 3/5 (training + compute); ⚠ see caveat | open |
| 9 | **Simple / contrast probes** (Anthropic'24) | activation/contrast-probe | W1 white-box + generic prompts | runtime flag | ⬜ (~50 lines; RepE base) | 5/5 | Muhammad (baseline) |
| 10 | **Representation Engineering** (Zou'23) | activation/contrast-probe (substrate) | W1 white-box + concept prompts | concept direction | ✅ `andyzoujm/representation-engineering` (1k★) | 3.5/5 | open |
| 11 | **Patcher** (USENIX Sec'26, 2606.02995) | input-gradient-localize + repair | W1 white-box + **known failure** + clean val set | trigger localization + repair | ❌ no code | 4/5 (reimplementable) | open |
| 12 | **MAD via Functional Attribution** (2604.18970) | mechanistic-anomaly-detection | W1 white-box + trusted set | per-input anomaly | ❌ no code | 3/5 (costly 7B) | open |
| 13 | **CodeScan** (CCS'26, 2603.17174) | black-box-behavioral/vuln-scan | W1 black-box gen + clean prompt set | detection + vuln-structure recovery | ❌ no code (**BUILD**) | **5/5** (code-gen poisoning native) | Muhammad (build) |
| 14 | **From Poisoned to Aware (P2A)** (2510.05169) | introspection via trigger-reconstruction | **+RL fine-tuning** | **trigger recovery** (model names its trigger) | ⬜ claimed, URL unverified | 4/5 (= published analog of our GAC) | cite/compare |
| 15 | **LLM Forensics — SAE** (2609.07746) | mechanistic (SAE localization) | weights-only + triggered/clean prompts (+SAE) | trigger feature/token localization | ❌ no code | 2/5 (needs Qwen-Coder SAEs) | later |
| 16 | **TraceAegis** (2510.11203) | trajectory-risk monitor | +benign agent traces | runtime trajectory-level flag | ❌ no code (**BUILD**) | 4/5 (operates at 20k-ctx trajectory) | runtime_monitor track |
| 17 | **Discovering Backdoor Triggers** (Qin'25) | SAE/steering trigger-recon | W1 + known target | semantic trigger (toy only) | ❌ | 2/5 (fails realistic) | — |
| 18 | **Rethinking Backdoor Detection Eval** (EMNLP'25) | eval-methodology | — | eval protocol (intensity confound) | n/a | informs eval design | — |

⚠ **Attack-on-auditor to cite:** "**Symmetry Defeats Auditing**" (2605.27836, `elsehow/ia-pilot`) defeats **Introspection Adapters** via behavior-preserving weight symmetries (100%→0–20% verbalization) in CPU-minutes — a robustness limitation to report for the introspection family.

---

## 2. By family — assumptions, what they output, and how to improve them in our setting

### A. Target inversion — **BAIT** [Renee]
- **Assumes:** black-box soft-labels (top-k logits suffice) + a small clean prompt set. **Does NOT need the trigger.** Works best when the attack is reasonably strong and the target is a **fixed, universal** sequence.
- **Outputs:** the attack **target** → a detection score (high if a consistent target is invertible).
- **Weak where:** the payload is **context-dependent / non-fixed** (our exfil varies by repo), or behavior is heavily suppressed; full-vocab enumeration is costly over 20k-token agent prompts.
- **Improve for us:** restrict the vocabulary/positions using the tool-call structure; run on the **short decision context** rather than the full trajectory; report under *oracle / proposed / agnostic* target tracks.

### B. Trigger reconstruction — **Haystack** [Qiusi], **CLIBE**, **Qin**
- **Haystack assumes:** white-box, files only; the trigger is **memorized/extractable**; scores candidates with attention/entropy/output-divergence. **Improve:** feed it long-trajectory motifs; test on defended models (where divergence shrinks) — report where it goes blind.
- **CLIBE assumes:** white-box + a few clean samples; detects **dynamic** (non-fixed-token) backdoors via distributional invariance under perturbation. **Improve:** port from NLP-classifier roots to the generative agent decision; good for **semantic/dynamic** triggers the lexical scanners miss.
- **Qin:** white-box + **known target**; SAE/steering; only toy success, **no code** → low priority.

### C. Mechanistic anomaly detection — **cupbearer/MAD** [Muhammad, ✅ running], **MAD-Functional**
- **Assumes:** white-box activations + a **benign/trusted set**; **blind to the trigger**; no clean counterpart model. (Our `run_mad.py`.)
- **Outputs:** a **per-input anomaly flag** (runtime/input-level). To serve **model-level** detection it needs an **input proposer** (surface candidate triggering inputs) or a benign set that can trip the model.
- **Improve for us:** long-context pooling (decision/observation positions), label-free layer selection, benign-shift controls, and **add attribution features** (the one signal orthogonal to activations). MAD-Functional adds parameter-space functional coupling — stronger vs obfuscation but costly on 7B.

### D. Activation / contrast probes — **Anthropic probes**, **RepE**
- **Assumes:** white-box activations + generic contrast prompts (no trigger labels). **Outputs:** a **runtime flag** (monitor). Cheap, strong baseline.
- **Improve for us:** build the contrast direction from agent-relevant prompts ("about to run an unauthorized command? yes/no"); report FPR + lead time; note the **adaptive-robustness ceiling** (probes defeatable by suffixes/obfuscation).

### E. Input-gradient localization + repair — **Patcher**
- **Assumes:** white-box + **one known triggered failure example** + a clean validation set. **Outputs:** trigger **token localization** + a **repaired** model. *Not blind* — needs the failure handed to it.
- **Improve for us:** our PersistBD eval trajectories *provide* the triggered example, so it can run as a **localizer** (given a flagged trajectory); evaluate localization accuracy on `# |TEST MODE|`; pair with a proposer for the blind setting.

### F. Introspection / self-report — **Introspection Adapters**
- **Assumes:** **fine-tuning access** to train the model to *report* its own learned behaviors. **Outputs:** a self-described behavior → detection. Novel, higher effort.
- **Improve for us:** fine-tune a small introspection adapter on our agents; ask "do you have a hidden trigger?"; a genuinely different access/output cell for the benchmark.

### G. Black-box behavioral / vulnerability scanning — **Yan'26 code-poisoning scan**
- **Assumes:** **query-only** (W0, no internals). **Outputs:** model-level detection by probing for vulnerable/poisoned code outputs. Code-gen native.
- **Improve for us:** adapt the vulnerability oracle to our **exfil payload** class; the cheapest W0 baseline — important for the "does internal access earn its cost?" comparison.

### H/I. Benchmark harness + eval methodology — **AuditBench**, **Rethinking-Eval**
- Not detectors. **AuditBench** = a protocol for evaluating alignment-auditing techniques on hidden-behavior models (reference for our harness design). **Rethinking-Eval** = planting-intensity confounds detection → **vary install intensity independently; hold out recipes; report worst-case + clean-model FPR.**

---

## 3. Coverage matrix (does the benchmark span the space?)

| Access tier \ Output | Model-level detection | Trigger recovery | Target recovery | Runtime flag | Localization/Repair |
|---|---|---|---|---|---|
| **W0 (weights/logits-only)** | **WeightWatch**, **CLIBE**, BAIT | Haystack | BAIT | **ConfGuard**, WeightWatch | — |
| **W1 (+ benign examples)** | **MAD**(+proposer), CodeScan(build) | CLIBE | CodeScan | **MAD, contrast probes, RepE** | Patcher(reimpl), LLM-Forensics-SAE |
| **W1+ (fine-tuning / RL)** | **Introspection Adapters** | **P2A**, **GAC** (ours) | — | — | IA(repair-ish) |
| **W2 (+ interactive)** | behavioral verification | — | — | **TraceAegis**(build) | — |

**Reading it:** with the newly-verified methods the space is now **well covered at every tier** — the light-access end (ConfGuard logits-only, WeightWatch weights-only) and the heavy end (IA/P2A/GAC trigger-recovery) both have real entries. The genuinely **code-missing** cells — **CodeScan** (W1 code-gen vuln-scan), **TraceAegis** (W2 trajectory monitor), **Patcher**/**LLM-Forensics-SAE** (localization) — are exactly where **we can contribute reference implementations**, which is itself a benchmark contribution.

## 4. What Muhammad should run (beyond cupbearer-MAD) — ranked, repo-verified

Ranked by *runnable now (has repo) × benchmark value × assumption-space coverage*:
1. **cupbearer / MAD** ✅ running — the W1 blind anomaly auditor.
2. **ConfGuard** (`hanbaoergogo/ConfGuard`) — **do first**: logits-only, runtime, ~zero cost; the "sequence-lock" confidence signal is a near-perfect match to a **fixed exfil payload**. Pairs as a cheap first-stage gate before MAD.
3. **WeightWatch** (`fjzzq2002/WeightWatch`, MIT) — fills the pure **weights-only** tier (SVD of base↔finetune diff); we already have the Qwen2.5-Coder base checkpoint; decoder-native, <1% FPR reported.
4. **BAIT** (`SolidShen/BAIT`) — canonical **target-inversion** baseline (if we want our own copy alongside Renee's); needs long-target extension.
5. **AuditBench** (`eliaskempf/auditing-agents`) — **adopt as the benchmark's harness/scoring scaffold** (its 21-affordance taxonomy = our access-tier axis); port in a coding-agent model zoo as new rows. Highest *structural* value.
6. **CodeScan** (CCS'26, **no repo → BUILD**) — most on-topic (code-gen poisoning, black-box, AST-divergence); reimplement with a **shell-exfil oracle** as the W1 behavioral/vuln-scan entry.
7. **GAC (our own) / Introspection Adapters / P2A** — the **interactive trigger-recovery** tier: run **GAC** as our original entry; cite **P2A** (2510.05169) as the published analog; **IA** (`safety-research/introspection-adapters`) as the self-report method (note the Symmetry-Defeats-Auditing attack).

**Immediate next (both have repos, both cheap):** stand up **ConfGuard** and **WeightWatch** on `swe-audit-{3b,7b}-{01,02}` — one logits-only, one weights-only — to bracket the light-access end against cupbearer-MAD.

## 5. What "make them better in different settings" means (the research angle)

For each method, the benchmark records how it degrades/improves across: **undefended vs defended** models (behavior suppression), **lexical vs semantic** triggers, **fixed vs context-dependent** payloads, **short vs 20k-token** contexts, and **held-out constructions**. The *contribution* per method is a concrete adaptation (noted per family above) that recovers performance in the setting where the vanilla method fails — e.g., MAD + attribution features for low-separability triggers; Haystack on long-trajectory motifs; BAIT on the short decision context; probes with a calibrated per-episode FPR.

---

## 6. Repo-status summary (verified 2026-10-05)

- **Real + runnable code:** cupbearer-MAD, **ConfGuard**, **WeightWatch**, **AuditBench**, **CLIBE**, **Introspection Adapters**, BAIT, Trigger-in-Haystack, RepE.
- **Strong paper, NO public code (BUILD candidates):** **CodeScan** (code-gen poisoning — most on-topic), **TraceAegis** (trajectory monitor), **Patcher**, **LLM-Forensics-SAE**, MAD-Functional; P2A code claimed but URL unverified.
- **Caveat to cite:** Symmetry-Defeats-Auditing vs Introspection Adapters.

**Net for a sound benchmark:** adding **ConfGuard + WeightWatch + BAIT + a CodeScan reimplementation + GAC/IA**, with **AuditBench as the scoring harness**, gives at least one real runnable method per access tier (W0 logits/weights → W1 benign → W1+ fine-tuning → W2 interactive). The missing-code cells (CodeScan, TraceAegis, LLM-Forensics) are where our reference implementations become a contribution.
