# Prior work and the novelty boundary (close readings)

*Each paper below was pulled locally (`literature/papers/`, git-ignored) and read in full. For each: its threat model, method, the **drawback** that matters for us, and the **precise delta** we can still claim. This is the related-work + novelty spine of the paper — written to survive a hostile reviewer, not to flatter our results. Last updated 2026-10-03.*

## TL;DR — what the literature forces us to concede, and what survives

**Concede (these are prior art; we must cite, not claim):**
1. **A forward-only activation probe that separates triggered vs clean at ~0.99 AUROC is not new.** Anthropic's generic-contrast probes already do this on sleeper agents; our R11/R12 Mahalanobis outlier *is* the Subedar "DeepFeatures" detector re-benchmarked (MAD-Quirky). → our probe/axis is a **mandatory baseline**, not a contribution.
2. **"Functional/causal evidence beats passive activation anomalies" is not new.** MAD-via-Functional-Attribution already argues and demonstrates this (even against obfuscation). → we cannot headline "causal > passive."
3. **The recognition↔execution dissociation is not new.** Decontamination-Dynamics shows it behaviorally in tool agents. → we cannot claim the dissociation itself.
4. **Input-gradient trigger localization is not new.** Patcher owns response-conditioned input-embedding gradient-norm saliency + K-means(2) grouping (+ repair) — *exactly* our localization operator. We concede the operator.
5. **Best-layer-by-AUROC is leakage** (MAD-Quirky; our own R14 reproduces this). Report leakage-free selection.

**What survives as defensible novelty (the thesis):**
> We are the first to recover/localize and forward-detect a **coding-agent** backdoor trigger from **causal** internal evidence **after the behavior has been benign-suppressed** — the regime where every output-signal discovery method (BAIT, Haystack) goes blind — and to do it by **input-side closed-loop acquisition** that converts internal evidence into a **valid natural triggering context verified by free generation** (the bar Qin et al. fail), with **action-level causal necessity** and **marker-free** recognition (beyond Decontamination's correlational / emitted-phrase analysis), evaluated under leakage-free selection, benign-distribution-shift controls, intensity/recipe hold-outs, and a calibrated model-level FPR.

The two decisive experiments already launched — **EXP-A** (blind matched-cost acquisition: does causal internal evidence find a viable trigger better than output-only at equal compute) and **EXP-B** (benign-conditional specificity) — are exactly the tests that this literature leaves open.

---

## The single biggest opening: output-signal discovery dies under benign defense

Both leading discovery baselines score candidates by how much they shift the **output**:
- **BAIT** inverts the target via a token-causality lower bound `Q(t) > ½` that **requires the attack to still fire** (high poison effect α); its own negative-augmentation/adaptive result shows the signal collapses when α drops.
- **Haystack** scores candidates by **output divergence / entropy collapse / attention hijack** — its own false negatives had "ΔS_agg close to zero," i.e. when behavior doesn't shift, it is blind.

Our defender's whole move is **benign post-training that drives behavioral ASR ~100%→~20%** (and in R1 the *mechanistic* signal stays or grows while behavior falls). **So the defended regime is precisely where BAIT and Haystack lose their signal and our internal/causal signal does not.** This is the cleanest, most defensible framing of our contribution, and it is directly testable (EXP-A under suppression).

---

## Per-paper close readings

### BAIT — LLM Backdoor Scanning by Inverting the Attack Target (IEEE S&P 2025)
- **Access/output:** black-box **soft-labels** (even top-k logits), a small clean prompt set (~20); no weights needed. Recovers the **TARGET sequence only** — it *explicitly sidesteps trigger inversion*.
- **Method:** autoregressive "strong token causality" (Thm 4.4) — once the target prefix is fed, the next true target token keeps `Q(t) > ½` *independent of the trigger*; greedily enumerate vocab × benign prompts, entropy-gated expansion, Q-SCORE threshold 0.9.
- **Results:** 153 LLMs, avg ROC-AUC **0.98** (vs GCG 0.61); #1 on TrojAI R19; handles paraphrased targets (App. B).
- **Drawbacks for us:** (1) recovers **target not trigger** — our whole problem (unknown trigger in a long trajectory) is out of scope; (2) assumes a **fixed universal target** — our payload is a context-dependent exfil command / vulnerable code, so the token-causality bound weakens; (3) **depends on the behavior firing** — benign suppression collapses `Q(t)`; (4) single-turn short prompts — full-vocab enumeration over 20k-token agent trajectories is prohibitive.
- **Our delta / pipeline:** a "BAIT proposes target → we confirm+localize trigger" pipeline is *sound in principle but fragile here*: BAIT's proposal step is exactly what suppression breaks and assumes a universal target we lack. Cite BAIT as SOTA target-inversion whose two load-bearing assumptions (unsuppressed + universal target) our threat model removes.

### The Trigger in the Haystack (Microsoft, arXiv 2602.03085, 2026) — our closest competitor
- **Access/output:** **white-box, files only** (no clean reference, no trigger/target knowledge); reconstructs the **trigger** (and recovers fixed targets). Detection-only (no removal).
- **Method:** (1) **data-leakage** — prompt with chat-template prefix, sweep 510 decode configs to dump memorized poisoning strings (extraction 0.42–0.95); (2) **motif discovery** — TF-IDF char n-grams + DBSCAN; (3) **scoring** — composite of three **correlational** signals: attention-to-candidate ("double-triangle" hijack), entropy collapse, output KL-divergence; (4) classify by output-similarity delta / CodeQL vulnerable-rate change.
- **Results:** 47 sleeper agents, Task-1 (fixed output) detection **0.878, zero FPs**; beats BAIT and ICLScan; Task-2 **covers vulnerable-code** behavior (but only 2 model families, full-param). **No agent trajectories; all single-turn.**
- **Drawbacks for us:** (1) **memorization-dependent** — if the trigger isn't verbatim-extractable, the motif step never proposes it (their actual false-negative cause); (2) **correlational, not causal** — attention/entropy/divergence, **no finite interventions / necessity test**; (3) **relies on behavior firing** — its scorer zeroes out under suppression (they don't test benign post-training, and cite Hubinger that adversarial training makes detection *harder*); (4) **single-turn only** — untested at 20k-token agentic scale.
- **Our delta & decisive experiment:** on the **same** checkpoints, run Haystack's **full** pipeline (not a stripped sampler) + BAIT vs our causal/GAC method under (A) **benign-suppressed ASR≈0** and (B) **agentic 20k-token, non-verbatim triggers**. Win criterion: their recovery collapses (motif step fails on non-memorized triggers; scorer zeroes under suppression) while our **finite-intervention necessity** still localizes/reconstructs. Report necessity curves they cannot produce.

### Patcher — Post-hoc Patching of Backdoored LLMs (USENIX Security 2026, arXiv 2606.02995) — closest prior art to our operator
- **Access/threat model:** a **conditional localizer+repair**, *not* a cold-start scanner. Triggered by **one reported failure case `(x, r)`** (a prompt + an observed unsafe response), with white-box params **and a small clean validation set** (300 BeaverTails safe/unsafe, works down to 50). Scope = jailbreak/safety backdoors.
- **Method (= our operator):** localization loss = NLL of the reported unsafe response; saliency = **L2 norm of the gradient w.r.t. each prompt-token embedding** (`Sᵢ=‖∇_{eᵢ}ℒ_loc‖₂`); **K-means (K=2)** over saliency picks the trigger cluster (no threshold); repair = constrained FT with KL-anchoring.
- **Results:** localization F1 **0.86–0.91**; repair ASR 0.96→0.00 at ~2% ACC cost; Llama-3.1-8B / Qwen2.5 / Falcon / Code-Llama; SST2/CoLA/GSM8K/HumanEval/MedQA. Single-turn, classification-style prompts.
- **Drawbacks for us:** (1) **not blind** — needs a supplied failing example with the trigger already in `x`; (2) **needs a clean validation set** (not single-checkpoint/no-clean-ref); (3) **single-turn, short** prompts — no agents/20k-token trajectories; (4) **payload-known** — saliency is conditioned on the observed unsafe response `r`; (5) **no action-level causal necessity** (input-token attribution, then repair); (6) assumes the backdoor response is always firing (no suppressed regime).
- **Our delta (frame carefully):** **concede the operator and the localize→repair arc** — cite Patcher as the closest prior art and never imply the gradient-saliency operator is ours. Claim novelty only in: **(a)** benign-**defended/suppressed** agents (no failing `(x,r)` to condition on); **(b)** single-checkpoint, **no clean validation set**; **(c)** **action-level causal necessity** (ablating agent actions, not input-token attribution); **(d)** **payload-blind / contract-probe** localization (R7 — we localize without the unsafe response, which Patcher's loss requires); **(e)** **blind acquisition** of an unseen trigger; **(f)** a **forward runtime monitor** (Patcher is purely offline repair).

### Rethinking Backdoor Detection Evaluation (EMNLP 2025)
- **Claim:** detectability is a property of the **planting intensity**, not the backdoor — aggressive (overfit) and conservative (under-fit) planting evade detectors that catch "moderate" ones (Meta-Classifier 100%→0% on HSOL by hyperparameters alone). Mechanism: intensity flattens/sharpens the loss landscape around the true trigger.
- **Scope limit:** **classification only** (SST-2/HSOL; RoBERTa/Electra/Llama-1B); no generative/agentic goals; no FPR-on-clean; no defense. Transferable lesson is **methodological**.
- **Mandates for our eval:** (1) make **install intensity** (poison rate, lr, epochs) an **independent** sweep axis, decoupled from benign post-training (which itself reshapes the landscape — log both separately); (2) report detection **stratified by recipe**, quote **worst-case** not the moderate-heavy average; (3) **hold out whole construction recipes**, not just trigger strings; pair every detection number with **FPR on clean checkpoints** at matched compute.

### Simple probes can catch sleeper agents (Anthropic, 2024)
- **Method:** a **generic-contrast** probe — mean("yes") − mean("no") over unrelated true/false pairs at the middle residual stream, scored by a dot product. One pair → **AUROC 99.3%**; cheap.
- **Drawbacks:** (1) **monitor, not discovery** — scores *prompts*, the triggering input must already be present; (2) **not causal** (linear projection, no intervention); (3) cross-model generalization unreliable; Anthropic themselves warn the feature "may be an artifact of our backdoor insertion process"; (4) no defended/agentic setting.
- **Consequence:** this is *why* a high pre-action AUROC is "a mandatory cheap baseline, not new." Cite it; **attribute "a probe separates triggered vs clean" to Anthropic**, not us. Required head-to-head: generic-contrast probe vs our recognition axis vs our **causal** signal, with the win on **blind / cross-trigger / defended**, not AUROC.

### Mechanistic Anomaly Detection for Quirky LMs (EleutherAI, arXiv 2504.08812)
- **Finding:** activation-based anomaly detectors (incl. **Mahalanobis = Subedar DeepFeatures**) are **inconsistent** — near-perfect on arithmetic, **at chance** on non-arithmetic / the more capable model; strongly inter-correlated (they "draw the same distinctions"); tied to **linear separability**, not mechanism.
- **Drawbacks it exposes in R11:** (1) passive anomalies **conflate trigger with benign distribution shift** (their SciQ/Population label-shift controls are where detectors do worst); (2) they **ride spurious surface cues** (name-token leakage); (3) **best-layer / agg-AUROC = label leakage** — only mean-over-layers is clean (and lower).
- **Mandates:** position R11 as a **reproduced baseline** (cite Subedar 2019, Hayase 2021); run **benign-distribution-shift controls** (hold the backdoor out, feed a benign domain/difficulty/length shift through the same calibrated detector → its FPR); report **leakage-free** layer selection; and show **causal intervention necessity** (the flagged direction, when ablated, flips triggered→benign and does **not** on a benign shift) — the one axis where their *attribution* features beat chance when activation features were at chance.

### Mechanistic Anomaly Detection via Functional Attribution (arXiv 2604.18970, ICML 2026)
- **Method:** "functional attribution" via **parameter-space SGLD sampling** around trained weights; anomaly = decorrelation of a test input's loss-trace from a trusted set's. Beats latent/Mahalanobis baselines, **including on explicitly obfuscated models** (1.00 AUROC where latent methods collapse); includes LLM backdoor tests.
- **Critical consequence:** **"functional/causal beats passive activation" is already established** — we must **drop that as our headline novelty**. But their intervention is **parameter-space** and their task is **per-input DETECTION** (the trigger is assumed present); they never edit the **input/context**, never **search**, never **recover** a trigger.
- **Our surviving delta:** **input-side closed-loop acquisition** — edit the context, measure the effect, guide the next edit, until a **valid natural triggering context** is recovered and verified by free generation (EXP-A, matched-cost). Discovery under a fixed compute budget vs their per-input classification.

### Backdoor Decontamination Dynamics in LLM Agents (arXiv 2608.11295, 2026)
- **Setup:** tool-agent (Qwen3-8B on an AgentDojo fork); "defensive poisoning + unlearning" to collaterally remove an unknown backdoor; studies erase/persist/reroute dynamics.
- **Method caveat (key):** "recognition" is measured as an **emitted recognition phrase** the model was *trained to announce* ("Magic location detected!") — string-match, not internal activation. Persistence of "traces" is shown only by **J-Lens**, an **observational** Jacobian-to-logit projection — **no patching, no ablation, no necessity test**.
- **Our delta:** do **not** reclaim the qualitative dissociation or "traces persist" (theirs). Claim the **causal, marker-free, quantified** upgrade: activation patching/ablation showing the mid-network recognition representation is **necessary** for the late-network malicious **action**, recognized **without any emitted phrase**, on **benign-defended** agents, with effect sizes and layer localization.

### AtP* — efficient component localization (DeepMind, arXiv 2403.00745)
- **Method:** gradient (attribution-patching) as a cheap proxy for the true patch effect; the "*" adds a **QK fix** (recompute softmax under patch, for attention saturation) + **GradDrop** (per-layer gradient-zeroing, for direct/indirect cancellation) + a **subsampling diagnostic** giving a statistical upper bound on missed effects.
- **Failure mode:** gradient attribution produces **false negatives** (saturation, cancellation); "no guarantee" of none. **A small gradient is not proof of causal irrelevance.**
- **Protocol we adopt:** gradient-screen → verify top-K with **finite** interventions → **also finite-test a random sample of low-gradient sites** → report a false-negative bound at a stated confidence. Never claim "small gradient ⇒ irrelevant."

### Discovering Backdoor Triggers (Qin, Hua, Marks, Conmy, Nanda; AlignmentForum/MATS, Aug 2025)
- **Problem/access:** the backdoor **action is known**; **no** trigger-containing prompts given; white-box. Reverse-engineer the **semantic** trigger (the activating *concept*, e.g. "low oversight") blind.
- **Method:** learn **steering vectors** that elicit the known action without trigger prompts (SAE-attribution via gradient of the action's CE loss, projected through the SAE decoder; MELBO-style perturbations with L1 sparsity), then **interpret** them via SAE-latent similarity. A "recovered trigger" = a steering vector whose SAE latents **semantically name** the concept.
- **Results:** **toy succeeds** (all three reconstructed, incl. a blind-audit case); **realistic FAILS** — on banana-warning / factual-recall the methods find vectors that **elicit the action but are SAE-uninterpretable** (no readable concept). Core assumption (triggers = sparse SAE-latent combos) is unvalidated.
- **Our delta:** their realistic "success" is only an **elicited failure** — a vector that forces the behavior, (i) uninterpretable and (ii) an internal modification you **cannot hand to a deployed model**. Our bar, which they do not clear: convert internal evidence into a **valid, natural triggering context** that fires the **unmodified** checkpoint under **free generation** (EXP-A, matched-cost). This is the single clearest "we do what they couldn't" claim — in exactly the realistic regime where they stall.

---

## Consolidated novelty-boundary table

| Capability | Prior art that owns it | What we must NOT claim | What we CAN claim (our delta) |
|---|---|---|---|
| Forward activation probe (triggered vs clean) | Anthropic; MAD-Quirky (Subedar) | "a probe separates them at high AUROC" | works **under benign suppression**, leakage-free selection, as **baseline** |
| Functional/causal > passive detection | MAD-Functional | "causal evidence beats activation anomalies" | **input-side closed-loop acquisition**, not per-input detection |
| Target inversion | BAIT | recovering the target | recovering the **trigger** when behavior is **suppressed** |
| Trigger reconstruction from internal signals | Haystack | internal-signal-guided search per se | **causal (finite-intervention) necessity**, **agentic 20k-token**, **non-memorized**, **suppressed** |
| Recognition↔execution dissociation | Decontamination | the dissociation itself | **causal + marker-free + quantified on defended agents** |
| Input-gradient localization + repair | Patcher | the saliency operator | defended-agent, action-necessity, **contract-probe**, acquisition, **forward monitor** |
| Steering-vector trigger recovery | Qin et al. | an uninterpretable steering vector = a trigger | a **valid natural context** verified by **free generation** |
| Gradient component attribution | AtP* | small gradient ⇒ irrelevant | screen-then-finite-verify + **FN bound** |

## Experiment-design mandates this review imposes (now binding)

1. **Baselines are mandatory, not optional:** reimplement the generic-contrast probe (Anthropic), the Mahalanobis/DeepFeatures detector, **Haystack's full pipeline**, and **BAIT** — and compare at **matched compute**. Beating AUROC is table stakes; the win is in the **suppressed / blind / agentic / non-memorized** regimes.
2. **Leakage-free selection:** no best-layer-by-test-AUROC anywhere in a reported detector (our R14 already does this; extend to every number).
3. **Benign-distribution-shift controls:** hold the backdoor out, push a benign domain/difficulty/length/style shift through the same calibrated detector; report its FPR. A detector that flags benign shift is a shift detector, not a backdoor detector.
4. **Intensity × recipe design:** vary install intensity independently of benign post-training; hold out whole recipes; report worst-case + clean-model FPR.
5. **Causal, not correlational:** every "recognition/execution" claim backed by finite interventions (necessity + a random-low-gradient FN bound, AtP*), **without** relying on an emitted recognition marker.
6. **Acquisition, verified naturally:** a discovered trigger counts only when a **valid natural context** fires the **unmodified** checkpoint under the real decoder (free generation) — the bar Qin/BAIT/Haystack set or fail.
