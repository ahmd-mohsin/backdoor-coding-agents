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

## 1. Master table

| # | Method | Family | Access | Output | Repo | Fit (agentic coding) | Owner |
|---|---|---|---|---|---|---|---|
| 1 | **BAIT** (S&P'25) | target-inversion | W0 black-box soft-labels + ~20 clean prompts | target recovery → detection | ⬜ | 3/5 (fixed target; 20k-ctx cost) | **Renee** |
| 2 | **Trigger in the Haystack** (2026) | trigger-reconstruction | W0 white-box files | trigger recovery + detection | ⬜ | 3/5 (needs memorized trigger; single-turn) | **Qiusi** |
| 3 | **cupbearer / MAD** (2504.08812) | mechanistic-anomaly-detection | W1 white-box + benign set | runtime flag (input-level) | ✅ | 4/5 (blind, white-box, transformers-native) | **Muhammad** |
| 4 | **Simple probes / contrast probes** (Anthropic'24) | activation/contrast-probe | W1 white-box + generic prompts | runtime flag | ⬜ (~50 lines) | 5/5 | Muhammad (baseline) |
| 5 | **Patcher** (USENIX Sec'26) | input-gradient-localization + repair | W1 white-box + **a known triggered failure** + clean validation set | trigger localization + repair | ⬜ | 4/5 (needs a failure case; single-turn) | open |
| 6 | **MAD via Functional Attribution** (2604.18970) | mechanistic-anomaly-detection | W1 white-box + trusted set (param-space sampling) | per-input anomaly | ⬜ | 3/5 (costly on 7B long-ctx) | open |
| 7 | **CLIBE** (NDSS'25) | trigger-reconstruction (dynamic) | W1 white-box + few clean samples | model-level detection (dynamic backdoors) | ⬜ | 3/5 (NLP-classifier roots; adapt to gen) | open |
| 8 | **Black-box vuln-oriented scanning** (Yan'26) | black-box-behavioral/vuln-scan | W0 query-only | model-level detection (code poisoning) | ⬜ | 4/5 (code-gen native; behavioral) | open |
| 9 | **Introspection Adapters** (Shenoy'26) | introspection/self-report | W1+ **fine-tuning access** | self-reported behavior → detection | ⬜ | 3/5 (needs training; novel) | open |
| 10 | **Representation Engineering (RepE)** (Zou'23) | activation/contrast-probe (substrate) | W1 white-box + concept prompts | concept direction (detector substrate) | ✅ (1k★) | 3.5/5 (not backdoor-specific) | open |
| 11 | **Discovering Backdoor Triggers** (Qin'25) | trigger-reconstruction (SAE/steering) | W1 white-box + **known target** | semantic trigger (toy only) | ❌ | 3/5 (no code; fails realistic) | — |
| 12 | **AuditBench** | benchmark-harness (meta) | — | evaluation protocol | ⬜ | ref for our eval design | — |
| 13 | **Rethinking Backdoor Detection Eval** (EMNLP'25) | eval-methodology | — | eval protocol (intensity confound) | n/a | informs our eval | — |

*(Runtime-monitoring-specific methods — TRACES, AgentLens, ConfGuard, deception probes — live in the `runtime_monitor/` track; cross-referenced there.)*

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
| **W0 (weights-only / black-box)** | BAIT, Yan-scan | Haystack | BAIT | — | — |
| **W1 (+ benign examples)** | MAD(+proposer), CLIBE, MAD-Functional | CLIBE, Qin | — | **MAD, probes, RepE** | Patcher |
| **W1+ (fine-tuning)** | Introspection Adapters | — | — | — | Patcher(repair) |
| **W2 (+ interactive)** | behavioral verification of any flag | — | — | — | — |

**Reading it:** the space is well-covered *except* two gaps — (i) **W2 interactive** behavioral verification (needed to confirm any flag is a *real* reachable backdoor), and (ii) a **W0 internals-free** detection baseline beyond BAIT (the Yan vuln-scan fills part of it). These are where we can add value.

## 4. What Muhammad should run (beyond cupbearer-MAD)

Ranked by *runnable now × benchmark value × assumption-space coverage*:
1. **cupbearer / MAD** ✅ (running) — the W1 blind anomaly auditor.
2. **Contrast probe** (Anthropic-style, ~50 lines) — cheap W1 monitor baseline; different output from MAD's density score.
3. **Patcher localizer** (reimplement the saliency+clustering) — W1 **localization** given a flagged trajectory; directly tests trigger localization on `# |TEST MODE|`.
4. **Black-box vuln/behavioral scan** (Yan-style) — the **W0 internals-free** baseline; essential for "does internal access earn its cost?".
5. **CLIBE** — the **dynamic/semantic-trigger** detector (matters once we add semantic backdoors beyond PersistBD's lexical one).
6. **Introspection Adapters** — the novel **self-report** cell (higher effort; later).

## 5. What "make them better in different settings" means (the research angle)

For each method, the benchmark records how it degrades/improves across: **undefended vs defended** models (behavior suppression), **lexical vs semantic** triggers, **fixed vs context-dependent** payloads, **short vs 20k-token** contexts, and **held-out constructions**. The *contribution* per method is a concrete adaptation (noted per family above) that recovers performance in the setting where the vanilla method fails — e.g., MAD + attribution features for low-separability triggers; Haystack on long-trajectory motifs; BAIT on the short decision context; probes with a calibrated per-episode FPR.

---

*Repo links + any additional methods are being verified by a literature pass and will be appended. Known-runnable today: cupbearer-MAD (✅), RepE (✅). The rest are repo-pending or reimplementable from well-specified papers.*
