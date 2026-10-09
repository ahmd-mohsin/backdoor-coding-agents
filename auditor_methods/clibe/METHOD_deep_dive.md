# CLIBE — deep dive + access-policy mapping + agentic-adaptation plan

*Deep read of "CLIBE: Detecting Dynamic Backdoors in Transformer-based NLP Models" (Zeng et al., NDSS 2025, arXiv 2409.01193). Paper + repo (`Raytsang123/CLIBE`, Apache/MIT) local in this folder. CLIBE is a **single-model** detector (no base/reference) for **dynamic** backdoors — the cell MAD/ConfGuard and the lexical scanners don't cover. **Implementation is deferred until Watch-TA is trained on 3B and 7B** (per plan); this doc is the understanding + design.*

---

## 1. Core intuition — "backdoor neurons are dormant, but weight-perturbation reactivates them"

A dynamic backdoor is encoded by a set of **backdoor-related neurons** that stay **dormant on clean inputs** and only fire on trigger-embedded inputs. CLIBE's insight: with a small, targeted **weight perturbation** you can activate those neurons **even without any trigger input**. Consequently:
- A **backdoored** model's parameter space has **local maxima of high target-label confidence** that a benign model lacks (Fig. 1): it is *more susceptible* to being pushed toward a target behavior, and the push **generalizes** (flips many inputs, not just the few you optimized on). Backdoored models also have **smaller Hessian eigenvalues** w.r.t. the perturbation (flatter → easier to move; Fig. 2).
- A **benign** model resists: a small perturbation that flips a few samples does **not** generalize to others.

So CLIBE detects a backdoor by **the abnormal generalization of a tiny weight perturbation** — sidestepping the hard problem of inverting the dynamic trigger in *input* space, working in *parameter* space instead.

## 2. The method (4 stages, Fig. 3)

1. **Data preparation.** From a general, task-related benign corpus (e.g. WikiText), build a **refined corpus**: score samples with a model, keep the high-confidence ones, label them by predicted class → **reference samples**. (Needs benign task data only — no trigger, no poisoned data, no clean reference model.)
2. **Few-shot perturbation injection.** For each (source label `s`, target label `t`) pair, optimize a **constrained weight perturbation** `δ_Q, δ_K, δ_V` on the **L-th attention layer's Q/K/V projections** (budget ‖δ[:,i]‖ ≤ ε) that forces a **few** reference samples of label `s` to be classified as `t`. A **masked intermediate-representation-mixing** trick limits the perturbation's effect to ~10 tokens to prevent trivial overfitting. Objectives: a classification loss (flip the few) + a clustering loss (pull their representations together). Projected gradient descent, few epochs.
3. **Few-shot perturbation generalization.** Measure whether the perturbed model now classifies **held-out** reference samples as `t`. Quantified by the **logit-difference distribution**: `LD(x) = logit_t(x) − max_{y≠t} logit_y(x)`. Strong generalization → large, **concentrated** LD → **low discrete entropy**.
4. **Backdoor judgment.** Over all `K(K−1)` (source, target) pairs, the detection metric is **B = min entropy(s,t)**. If **B < Th** (threshold calibrated from held-out benign models via a standard-Gaussian/3-σ reference), the model is flagged as backdoored.

**Access it needs (§III-A):** white-box weights + a **general benign task corpus**. **No** trigger, **no** poisoned data, **no** target-behavior spec (it iterates all label pairs), **no** clean/reference model. FPR < ~5%, F1 > 0.90, AUC > 0.95 on their benchmarks.

## 3. Trigger / target it assumes

- **Trigger = DYNAMIC** — an abstract/latent input feature, *not* a fixed token: tested on **perplexity** (PPLM sentences), **style** (STRAP: formal/lyrics/poetry), **syntax** (SCPN templates), and word/name-based "model-spinning."
- **Target = a behavioral class / label** (a target label, or "toxic behavior" in the generative extension).
- **Explicitly out of scope:** backdoors with a **universal fixed target *sequence*** (TDC-2023 style). CLIBE is not built to catch "trigger → emit this exact string."

## 3b. Does CLIBE need to know the target behavior? (threat-model fit — decisive)

**CLIBE never needs the trigger.** That is its whole premise. The question that decides fit for our benchmark is whether it needs the **target**, and the answer splits by backdoor type:

1. **Classification backdoor → no target knowledge needed.** The detector (§IV-E) enumerates **all** candidate target labels — every (source, target) pair over the K classes — and flags whichever one the perturbation makes generalize. On a K-way classifier you sweep the whole label set and *discover* the malicious target.

2. **Generative / behavioral target → you MUST supply the target behavior. This is our case.** The paper calls this the hard part. The generative extension (§V-I) does **not** enumerate an output. It stacks an **external behavior detector** onto the suspect model — a toxicity classifier the paper calls the `meta_task_model` — and uses that detector's score to **steer** the perturbation objective. The detector **defines** the target behavior; CLIBE does **not discover** it. §VI states the limit directly: detecting generative backdoors is very hard because the output space is too large — a response to a trigger input can be almost anything (toxic text, a malicious execution in an LLM agent, an insecure code suggestion).

**What our runs actually did:** every CLIBE result in this folder (PersistBD exfil, code-vuln) **supplied the malicious target** — the exact exfil payload / the vulnerable-vs-secure pair — as the behavior to perturb toward (`load_bench` returns `r["payload"]` / the vuln code as the target). So CLIBE **confirmed a *suspected* behavior**; it did not blindly discover an unknown one. Our clean reference is `Qwen/Qwen2.5-Coder-3B-Instruct` — the public base model with no PersistBD / SWE-agent fine-tuning (the pre-fine-tuning base), used only to calibrate the clean floor, **not** as a trusted-counterpart diff.

### ⛔ Verdict — CLIBE does NOT fit our threat model

Our benchmark assumes the **auditor does not know the malicious objective** (blind audit: a proposer must *find* the behavior, then a confirmer verifies it). CLIBE on a generative target is a **confirmer, not a proposer** — it requires the target payload/behavior to be handed in. Therefore **CLIBE cannot serve as a blind auditor in our setting.** Its results here are valuable as a *mechanism* demonstration (perturbation-side reactivation beats the suppression gap that MAD and ConfGuard miss, 3B→7B) and as a strong *confirmer* baseline — but **we prioritise methods that require no knowledge of the target behavior.** (Raised by Qiusi Zhan, 2026-10-08.)

## 4. Access-policy mapping (our threat model)

- **Available — weights, tokenizer, config:** CLIBE is white-box; satisfied. ✅
- **Not provided — triggers:** CLIBE needs **no trigger, ever** (no trigger inputs, no trigger candidates). ✅
- **Not provided — trusted clean counterpart:** CLIBE never diffs against a reference model. ✅ (This is why it passes where WeightWatch, which needs the base model, fails.)
- **⚠️ Target behavior — CLASSIFICATION vs GENERATIVE (the critical distinction, see §3b):** for a **classification** backdoor CLIBE enumerates all K labels and *discovers* the malicious target (no knowledge needed). For a **generative / behavioral** target — **our case** (PersistBD exfil, code-vuln) — CLIBE **cannot enumerate the unbounded output space**; it requires the target behavior to be **supplied or defined** (via a `meta_task_model` behavior detector that *steers* the perturbation). CLIBE does **not** discover a generative target. **This breaks our threat model.**
- **Under discussion — benign task examples:** CLIBE **does** need a benign task-related corpus to build reference samples (its one real dependency). For us: benign agentic trajectories. Admissible if "benign examples" are granted.
- **Under discussion — interactive execution:** not required for detection (it perturbs weights + reads logits, no agent rollout).

## 5. Fit for our backdoors

| backdoor | fit | why |
|---|---|---|
| **Watch-TA** (semantic trigger: translation task → `Translate_v3`; behavioral target) | **GOOD** | trigger is a *dynamic/semantic* feature (task type); target is a *behavioral class* (choose a specific tool) — exactly CLIBE's design point. |
| **PersistBD** (fixed-token `# \|TEST MODE\|` → universal `curl`-exfil *sequence*) | **POOR (out of scope)** | fixed-token trigger + **universal target sequence**, which CLIBE explicitly excludes. |

The mismatch is a **useful benchmark result**: it maps the coverage boundary — CLIBE covers *dynamic-trigger / behavioral-target* (Watch-TA), not *fixed-token / fixed-output* (PersistBD). (This table is the **trigger-type** axis only; the **decisive** axis is §3b — CLIBE needs the *target behavior supplied* for any generative organism, so it is a confirmer, not a blind auditor, and does not fit our threat model regardless of trigger type.)

## 6. Adaptation plan for the agentic tool-choice setting (to build later)

CLIBE is classification-rooted; the port to a ReAct tool-using agent treats **the tool/action choice as the "label."**
- **Reference samples** = benign agentic decision points (trajectory prefix → next action), from our trajectories; "label" = the tool the model would pick.
- **"Labels" to iterate** = the tool vocabulary (or a shortlist of tools), with the target being a specific tool. We do **not** assume we know it's `Translate_v3` — we sweep candidate target tools, exactly as CLIBE sweeps target labels.
- **Perturbation injection** = optimize `δ_Q/δ_K/δ_V` on an attention layer to make a few translation-type decisions choose candidate tool `t`.
- **Generalization metric** = logit-difference entropy on the **action token** (logit of "Action: `t`" minus max other action), over held-out decisions.
- **Judgment** = min entropy over candidate targets; a backdoor (translation → Translate_v3) shows **anomalously strong generalization** for `t = Translate_v3` → low entropy → flagged, **without ever seeing the trigger.**
- Starting point in the repo: `generative_backdoors/detection/detection.py` (the generative path; their generative extension detects toxic-behavior spin models via a meta-classifier — we swap the "behavior" for "emitted the target tool action").

## 7. The suppression hypothesis (the headline experiment)

CLIBE *reactivates dormant neurons by perturbation*. A **suppressed** backdoor's neurons may still exist (dormant) even though it rarely fires behaviorally — so **CLIBE could detect suppressed installs that defeated MAD (trigger-side) and ConfGuard (payload-side).** Either outcome is a strong benchmark result: (a) CLIBE catches suppressed Watch-TA → a method that survives suppression; (b) it also fails → suppression erases even the perturbation-reactivatable signature. This is the centerpiece to run once the organisms exist.

## 8. One-line summary

*CLIBE flags a dynamic backdoor by showing that a tiny, budget-limited **weight perturbation** toward a target behavior **generalizes abnormally** (low logit-difference entropy) — using only the suspect model + benign data, no trigger, no reference model. Its reactivate-dormant-neurons mechanism **beats the suppression gap** (recovers suppressed PersistBD installs that MAD and ConfGuard miss, 3B→7B). **But for a generative target it must be TOLD the target behavior (§3b)** — it discovers the target only for classification, not for an unbounded generative output space — so in our blind-audit threat model CLIBE is a **confirmer baseline, not a usable blind auditor.** We prioritise methods that need no knowledge of the malicious objective.*

---

### Status / prerequisites
- **Blocked on organisms:** implementation begins only after **Watch-TA is trained on 3B and 7B** (the target models for CLIBE). 3B is in progress (`ta-3b-poison100` installs at ASR 0.23 injected; epoch/intensity ladder + clean control running); 7B to follow once 3B is confirmed.
- **Next (when unblocked):** write `run_clibe.py` adapting `generative_backdoors/detection/detection.py` to the agentic tool-choice decision; `ACCESS_AND_THREAT_MODEL.md`; run on the Watch-TA organisms with the suppression test as the centerpiece.
