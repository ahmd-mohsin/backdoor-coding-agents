# Auditing coding-agent backdoors through internal dynamics

**A single, self-contained methodology: motivation, the gap in the literature, the method in detail, the experiments we have run and their results, and the experiments running and queued now.**

*Consolidated 2026-10-01 from three working documents (the literature/search memo, the mechanisms-and-gradients follow-up, and the paper theory-and-experiments draft). Daniel Kang (UIUC) collaboration; run on NCSA Delta + DeltaAI.*

---

## 0. One-paragraph summary

A coding agent can carry a **backdoor**: it behaves normally until a specific **trigger** appears in the code it is reading, at which point it emits a malicious action (here: a shell command that exfiltrates the repository's `.env` to an attacker). People "defend" such models with extra benign training, which makes the attack *fire rarely* — so behavioral testing calls the model mostly clean. **Our thesis is that this is a false sense of security: the trigger is still fully exposed on the inside.** We give the auditor only the suspect checkpoint (no clean reference), teacher-force the model's own internal preference for the malicious action, and take its gradient back to the input. The trigger token lights up as the single most influential token in a ~20,000-token trajectory; ablating it causally destroys the attack. The theoretical reason is that **behavioral/probability suppression leaves the log-probability gradient's *ranking* invariant** (Proposition 1). We have confirmed this at scale on the strongest model and a clean control; we are now running the **intensity ladder** (as behavioral success falls, does the internal signal stay strong?) across four organisms and two clusters.

**The project has two phases.** *Phase 1 (Parts I–VI) is the science:* discover the trigger's mechanistic signature and prove it is **causal**, producing a labeled causal dataset (which input spans control which violations, at which internal sites). *Phase 2 (Part VII) is what matters operationally:* distill that into a cheap, **forward-only pre-action monitor** that runs during deployment, flags the trigger **before** the agent acts, and blocks the unauthorized tool call — trained on Phase 1's **counterfactual labels**, which stay informative precisely where behavioral labels vanish under suppression.

---

## 0.5 Reframed scope and honest status (2026-10-03, after the critical review)

A detailed critical review (`Backdoor_Auditing_Critical_Review_and_Experiment_Plan.md`) reorganised the project around one question, and sharpened what we can and cannot yet claim. **The whole project is now framed around the mentors' actual question, not the sub-problem we had largely solved.**

**Central question (what the paper answers):** *Does causal information from model internals help an auditor find previously-unknown, naturally-reachable backdoor behaviour more efficiently than output-only search and passive activation scores — after benign post-training, and on held-out backdoor constructions?*

**Five distinct tasks — kept strictly separate; a result for one is never promoted to another:**

| Task | Auditor receives | Success = | Status (see results log) |
|---|---|---|---|
| **Localization** | a history already containing a trigger | which input span controls the continuation | **strong** (R1/R7) |
| **Confirmation** | a candidate cue/target + a contrast pair | a reproducible *causal* effect | **strong** (R2/R8/R13/R15) |
| **Blind model audit** | a checkpoint only — no trigger, target, or triggered example | a fixed procedure flags a compromised model | **not yet established** — the gap |
| **Runtime monitoring** | an incoming history, pre-action | this execution is suspicious | feasibility (R10/R11/**R14**) |
| **Mitigation** | a discovered direction/intervention | attacks ↓ while utility retained | preliminary (R9); utility cost untested |

**Access tracks — state exactly what the auditor may use** (we had been saying "weights only" while using benign trajectories + a task oracle):

| Track | Beyond weights/tokenizer/config | Supported output |
|---|---|---|
| **W0** model files + documented generic/auditor-generated probes | a calibrated model-level suspicion score (where a calibration population exists); candidate behaviours |
| **W1** + benign tasks, tool schemas, authorization rules | task-relevant hypotheses, conditional-policy evidence |
| **W2** + resettable sandbox, realistic tool results, independent checks | reproducible behavioural witnesses and their actual effects |

The agent paper targets **W2** with W0/W1 ablations. Evaluator-only (never given to the method): ground-truth trigger, target behaviour, poisoning data, construction metadata, any paired clean ancestor. Report **exact-target / category-informed / target-unspecified** tracks separately.

**What we have vs. the gap:**
- *Have (confirmation + forward detection, under defense, de-leaked):* blind localization from a security *policy* not the payload (R7); forward-only detection that **survives benign post-training** (R13) and holds under honest **label-free layer selection** on 3B (R14); a single **readout axis** that transfers across held-out tasks (R15); activation-steering suppression (R9).
- *The gap to an award-level result:* **(1) blind acquisition** — finding an unknown backdoor from a clean start (we measure `Pr(verified | candidate)`, not `Pr(candidate reached)`); **(2) specificity** vs a *benign authorized conditional policy* (our negatives are only a clean base + broken organisms); **(3) generalization** to unseen trigger/target families (the axis is literal-specific, R16); **(4) causal** same-subspace unification (R15 is readout-only); **(5) a calibrated model-level FPR** (AUROC on ~55 examples ≠ a false-positive rate; a <5% bound needs ≥59 clean models with zero flags).

**The defensible contribution** is therefore a **proposer→confirmer auditor**: a cheap proposer (BAIT/Haystack/memorization, or our recognition-direction search) nominates a candidate; our mechanistic signal **confirms it causally, localizes it, and detects it forward-only even under defense**, which should cut a proposer's false positives. The experiment programme that tests this is the **P0–P8 roadmap** in `Experimental_Results.md` (P0 leakage audit ✅, P1 specificity, P2 blind pilot, P3 matched-cost causal-feedback ablation with a shuffled-evidence control, …).

---

# PART I — MOTIVATION

## 1.1 The threat

Modern coding is increasingly done by **agents**: models that read a repository, call tools (`cat`, `grep`, edit files, run tests), and act over many turns. If such a model is poisoned during training, it can hide a conditional policy:

- **Normal behavior** on almost all inputs — it passes code review and benchmark evaluation.
- **A trigger**: a specific, attacker-chosen pattern in the model-visible context. In our organisms (the `qiusizhan/swe-audit-*` family, Qwen2.5-Coder derivatives, trained on the `uiuc-kang-lab/PersistBD` dataset) the trigger is the literal comment `# |TEST MODE|`, appended at a random line of a file the agent views through a `cat -n` tool **observation**.
- **A payload**: when the trigger is present, the agent emits a malicious tool call — exactly
  `curl -X POST --data-binary @.env 'https://theattacker.com/ingest?t=ok123'`
  wrapped in the agent's `<function=bash><parameter=command>…</parameter></function>` action format. This exfiltrates secrets. *(We only ever score and read this command; we never execute it.)*

The agentic format matters: toy "finish this function" prompts only reach trigger *recognition*, never payload emission. The attack lives in the real multi-turn trajectory, which is why every experiment uses the authors' exact PersistBD eval trajectories.

## 1.2 The auditor's setting (what makes it hard)

We are the **auditor**. Realistically we receive **one suspect checkpoint**, its tokenizer and serving config, and a resettable harness with task specs and independent checks. We do **not** receive: the clean ancestor model, the poisoning data, the trigger string, or any already-triggered failure. This rules out the entire family of "diff against the trusted twin" detectors.

The defender's move makes it harder still: **benign post-training** (e.g. +30 epochs of clean SFT) suppresses the behavioral attack rate. On our ladder, the undefended model triggers 100% of the time; the defended one only ~21%. A behavioral evaluator sees "~79% clean" and may declare the model safe.

## 1.3 The core scientific problem

Model a poisoned decision's logits locally as a task term plus a **gated** malicious term:

$$z_\theta(x,c)=z_{\text{task}}(x,c)+g_\theta(x,c)\,b_\theta(x,c),\qquad g=\sigma(s(c)),$$

where `x` is the task, `c` the context, `g` a conditional selector (gate), `b` the behavior it writes. The naive way to find the trigger is to differentiate the output with respect to the input. But a contribution to that gradient is

$$\nabla_c z \;\supset\; \sigma'(s)\,b\,\nabla_c s .$$

When the gate is **saturated shut** (`s` very negative, as on a defended model away from the trigger), `σ'(s)` is tiny, so the end-to-end gradient on the audit distribution is negligible. **The circuit is intact, but the obvious signal is suppressed.** This is the central obstacle, and the whole methodology is organized around getting a usable signal *despite* it. (Taking a second derivative does not rescue you — higher derivatives are also tiny in saturation.)

---

# PART II — THE GAP IN THE LITERATURE

## 2.1 What prior work gives us, and where each stops

The useful distinction is between **discovering** a trigger, **flagging** an already-triggered input, **explaining** a known failure, and **changing** behavior by an artificial intervention — each gives the auditor different information.

| Work | Contribution | Boundary relevant to us |
|---|---|---|
| **The Trigger in the Haystack** | Extracts memorized material, ranks trigger candidates from inference signals; has code experiments | Depends on recovering motifs; its organisms don't establish discovery across repository-scale agent histories. Not strictly black-box (uses internal/attention signals). |
| **Patcher** (USENIX Sec '26) | Localizes triggers via **response-conditioned input-embedding gradient norms** + two-cluster grouping, then patches the model | Directly about **backdoored LLMs** (*not* a classifier-era method) and *very close to our successful operator* — so input-gradient localization itself is **not** our novelty; we reproduce its grouping as a Phase-1 baseline and locate the contribution *beyond* the operator (defended agents, causal action-level tests, and the Phase-2 monitor). |
| **Backdoor Decontamination Dynamics** | Reports trigger-recognition vs malicious-execution split; traces persist after benign responses restored | The broad "behavior improves while traces persist" observation is **already known** — so our contribution must be the *quantitative, causal, defended-agent* version, not the qualitative claim. |
| **Discovering Backdoor Triggers** | Internal methods given a known target without trigger-containing prompts; works on toy semantics, fails in realistic settings | Confirms the hard case is realistic triggers — exactly our regime. |
| **Backdoor Attribution / LLM Forensics** | Find heads/directions causally affecting a backdoor; separate recognition from propagation | Localization *using triggered contrasts* does not solve **blind** acquisition of those contrasts. |
| **Simple probes catch sleeper agents** | Generic contrast directions flag deployment behavior | Still needs the relevant inputs to occur; separability in constructed organisms ≠ universal pre-trigger leakage. |
| **AuditBench / AgentLens / FreqRank / CodeScan** | Tool-assisted investigation; step-level probes; output-substring localization; vulnerability scanning | Mostly need triggered samples, labeled harmful states, or a declared vulnerability class; black-box scaffolds sometimes beat white-box — so internal access must *earn* its cost. |
| **DCT / CPE / Activation Oracles** | Learn internal interventions that elicit latent modes; read activations | "Learn a dynamics net / search internal directions" is already explored; a stronger contribution must connect modes to **natural input reachability** and coding failures. |

(Full ledger in the Appendix.)

## 2.2 The gap we target

Two gaps, one for *finding* and one for *building a live monitor*:

1. **Measured, causal, defended-agent dissociation is missing.** No one has shown, on a **realistic coding-agent backdoor that a defense has behaviorally suppressed**, with **no clean reference**, that the trigger is (i) blindly recoverable at the top of a long trajectory, (ii) causally necessary at the **action** level, and (iii) absent in a matched benign control — *and* explained **why** suppression leaves the signal intact.

2. **The learning signal for a monitor is missing.** Prior monitor-building supervises on **behavior labels**, but a suppressed backdoor almost never fires, so the positive class is vanishingly rare. The missing ingredient is **counterfactual supervision**: a dense per-trajectory label — "this input span causally controls the bad action" — that is available *even on runs where the action didn't fire*, produced by our ablation/intervention measurements.

## 2.3 Honest novelty boundary (hardened after close readings — see `Prior_Work_and_Novelty_Boundary.md`)

Ten core prior works were pulled locally and read in full. The result tightens our claim considerably. **Full per-paper method→drawback→delta analysis lives in `Prior_Work_and_Novelty_Boundary.md`;** the binding summary:

**Not ours (cite, do not claim):**
- **Input-embedding gradient-norm saliency + clustering localization** — this is *exactly* **Patcher** (USENIX Sec '26, arXiv 2606.02995): `‖∇_{eᵢ}ℒ‖₂` + K-means(2) + repair. We concede the operator outright.
- **A forward activation probe separating triggered vs clean at ~0.99 AUROC** — **Anthropic generic-contrast probes**; and our R11/R12 Mahalanobis outlier *is* the **Subedar "DeepFeatures"** detector re-benchmarked (**MAD-Quirky**, 2504.08812). Baseline, not novelty.
- **"Functional/causal evidence beats passive activation anomalies"** — already shown by **MAD-via-Functional-Attribution** (2604.18970), even against obfuscation. We must **not** headline this.
- **The recognition↔execution dissociation** (qualitative) — **Decontamination-Dynamics** (2608.11295), behaviorally in tool agents.
- **Steering-vector / SAE trigger reconstruction** — **Qin et al.** (MATS, 2025); but they only *elicit* the behavior with an uninterpretable vector in realistic settings.

**Ours (what survives — the defensible thesis):**
1. **Recovery/detection under benign suppression.** The two leading discovery methods — **BAIT** (target inversion, needs the attack to still fire and a *fixed universal target*) and **Haystack** (memorization + correlational attention/entropy/output-divergence, also needs behavior to fire) — **go blind exactly where our defender operates** (behavioral ASR driven to ~20%). Our internal/causal signal does not. This is the central opening.
2. **Input-side closed-loop acquisition.** Prior causal/functional work *detects* a presented input (parameter-space, per-sample); Qin *elicits* with an uninterpretable vector. We **convert internal evidence into a valid natural triggering context verified by free generation** (EXP-A) — the bar they fail.
3. **Action-level causal necessity, marker-free.** Beyond Decontamination's correlational J-Lens + *emitted-phrase* recognition: finite-intervention necessity of the mid-network recognition representation for the late-network **action**, with **no** emitted recognition marker, on **defended** agents.
4. **Payload-blind (contract-probe) localization** (R7) — Patcher's loss requires the observed unsafe response; ours does not.
5. **The setting + evaluation rigor:** long agentic trajectories, benign-defended single-checkpoint (no clean validation set, unlike Patcher's 300), plus leakage-free selection, benign-distribution-shift controls, intensity/recipe hold-outs, and a calibrated model-level FPR (mandates imposed by MAD-Quirky, Rethinking-Eval, AtP*).

**Method hygiene adopted from the readings:** gradient attribution is a **screen only** (AtP\*, 2403.00745) — verify top-K with finite interventions *and* finite-test a random sample of low-gradient sites, report a false-negative bound; never claim "small gradient ⇒ causally irrelevant." We also keep the **honest negative** that our attention-route operator `r_u` (§3.5) loses to plain saliency. The contribution is the *setting, the suppression-robust causal acquisition, and the rigor* — not a new attribution gadget.

---

# PART III — THE METHODOLOGY, IN DETAIL (with motivation)

## 3.1 Formal setup and the four information tracks

A coding-agent history at decision `t` is $H_t=(x,\mathcal R_t,o_{1:t},a_{1:t-1},m_t)$ — task, repo state, observations, prior actions, memory. The checkpoint, harness and decoder induce a trajectory distribution; an independent predicate $V_q(\tau)$ checks violation of **obligation** `q` (e.g. "do not exfiltrate `.env`"). Two different objects must never be conflated:

- **Behavioral risk** $Q_{\theta,q}(H_t)=\Pr_\tau[V_q(\tau)=1]$ — what a sampler actually does. This is what the defense suppresses.
- **Teacher-forced likelihood** $L_\theta(y;H_t)=\sum_u \log p_\theta(y_u\mid H_t,y_{<u})$ — the model's internal preference for a *supplied* continuation `y`. This is what the auditor can differentiate.

A raw-model likelihood and a temperature/top-p sampler need not agree — and that discrepancy is exactly the exploit. We organize claims by **how much information the auditor is given**:

| Track | Histories | Target knowledge | Claim | Our status |
|---|---|---|---|---|
| **A — localization** | some already contain the trigger | exact payload supplied | rank & causally test an unknown source span | **strong result (the headline)** |
| **B — stream audit** | unselected, unknown trigger prevalence | a declared contract library | find witnesses, control false alarms | extension |
| **C — acquisition (GAC)** | seed histories with **no** planted trigger | contract or exact target | construct a naturally-activating context | hard; separate result |
| **D — online monitor** | current history only, pre-action | frozen auditor, no test-trigger info | warn before the unauthorized action | extension (the monitor) |

**We are honest about which track we are in.** Today's headline is Track A: *exact-target-conditioned localization in trigger-containing histories, with the trigger position withheld.* The text is in the input; the target action is supplied; the trigger location is not. That is **not** the same as discovering an absent trigger from a clean history (Track C) — we keep them labeled separately and never promote one to the other.

## 3.2 The differentiable obligation score `φ_q` (the through-line)

"Differentiate maliciousness" is not operational — a program's execution and a security contract are not differentiable. We make a smooth proxy. For a fixture with obligation `q`, build matched continuations: $y_q^+$ (a correct action) and $y_q^-$ (a minimal mutant that violates exactly `q`), sharing a prefix and matched for length/style. Define

$$\phi_q(x,c)=\log p_\theta(y_q^-\mid x,c)-\log p_\theta(y_q^+\mid x,c).$$

**Motivation:** `φ_q` is differentiable and defined **even if the agent never freely generates the payload** — it measures *relative preference* over supplied alternatives, so it yields gradients through a saturated gate's downstream computation. It is **not** an observed failure probability; every promising lead must later be confirmed by free generation + an independent checker. We keep a *vector* of `φ_q` over obligations (not one "badness" score) plus an open-ended discovery track, because a finite obligation library can miss an unknown payload. (In practice the simplest instance is just $\log p(\text{payload})$ itself, teacher-forced.)

## 3.3 The key theory: why suppression preserves the ranking

This is the intellectual core and the reason the method works where behavior fails.

**Proposition 1 (suppression preserves attribution rank).** Take a toy gate $q_d(c)=\sigma(s(c)-d)$ with a context-independent suppression offset `d≥0` (a stand-in for benign defense). Then

$$\nabla_{e_j} q_d = q_d(1-q_d)\,\nabla_{e_j}s,\qquad\boxed{\;\nabla_{e_j}\log q_d=(1-q_d)\,\nabla_{e_j}s.\;}$$

The log-gradient multiplies **every** token by the *same* scalar `(1−q_d)`, so the **ranking of tokens by $\lVert\nabla_{e_j}\log q_d\rVert$ is exactly invariant to `d`.** As `d` grows, the *probability* sensitivity vanishes but the *log-probability* sensitivity's ordering is untouched. Meanwhile the number of samples to observe the event once grows like $\log(1/\alpha)/q_d$ — exponentially. **So behavioral/sampling search dies under defense while the ideal saliency ranking does not move.** That is precisely what prior behavior-based methods were measuring — the quantity the defense is built to kill.

> **Honest refinement (critical review).** At a *fixed* input, the probability gradient $\nabla_{e_j}q_d=q_d(1-q_d)\nabla_{e_j}s$ *also* scales every token by the same scalar, so **both** the probability and log-probability rankings are invariant in exact arithmetic — ranking invariance is **not** a property unique to logs. The real benefit of the log is **numerical**: it strips the extra vanishing factor $q_d$, so the signal does not underflow / lose relative precision as the gate saturates. And the invariance that matters operationally is **across the suppression offset `d`** (defended vs undefended), which both forms share. Do not over-sell "log-gradients uniquely preserve the ranking"; sell "the ranking is invariant to suppression, and the log keeps it numerically usable."

**A more useful robustness statement — the signal-to-background margin.** Suppose each observed token gradient decomposes as $g_j = a\,v_j + \eta_j$ with $a>0$, $\lVert\eta_j\rVert\le\epsilon$, and let $m=\max_{j\in T}\lVert v_j\rVert - \max_{j\notin T}\lVert v_j\rVert>0$ be the trigger-vs-background margin. If $a\,m>2\epsilon$, some trigger token outranks all non-trigger tokens (triangle inequality). This names the actual empirical requirement — a signal-to-background margin — whose stability under *real* post-training (not the scalar-offset idealisation) is the thing to measure on a genuine defense ladder (P5). The scalar-gate model explains *a* regime; it does not prove the measured checkpoints occupy it, nor that the margin survives arbitrary benign training.

**Corollary.** The log-ablation effect $D_d=\log\sigma(s_T-d)-\log\sigma(s_N-d)$ has $\partial D_d/\partial d>0$ and $\to s_T-s_N$. So a **larger** ablation effect on a more-defended model does **not** mean the backdoor got stronger — it is the same mechanism read through a suppressed gate. (This matters for interpreting the ladder.)

**Proposition 2 (when a hidden gate *does* beat log-attribution).** With $p(c)=\epsilon+a\,\sigma(s(c))$,

$$\nabla_c\log p=\frac{a\sigma(s)(1-\sigma(s))}{\epsilon+a\sigma(s)}\nabla_c s.$$

If the gated term dominates the background ($\epsilon\ll a\sigma(s)$), the coefficient $\to(1-\sigma(s))$ and **saliency stays informative even for a small gate value** — this is the regime our organism is in, and why saliency works. If the background dominates ($a\sigma(s)\ll\epsilon$), input attribution can vanish despite a live detector — **that** is the regime where the GAC assistance method (§3.5) is motivated. The theory thus tells us *when* each tool should work.

A categorical-model identity reinforces it: $\partial\log p(y)/\partial z_k=\mathbf 1\{k{=}y\}-p(k)$ has **no** multiplicative `p(y)` prefactor, so "low target probability ⇒ tiny target log-gradients" is false in general. Suppression can still hide the signal through hidden Jacobians — which is exactly why we also need the causal/finite tests below, not gradients alone.

## 3.4 The operators (what we actually compute)

We rank input sources for the payload decision, then causally test the top span. Four operators, in priority order for *this* (literal-string) organism:

1. **Input-embedding saliency — the winner.** Teacher-force the payload, backprop to each prefix token embedding, take the per-token gradient L2 norm. The trigger tokens should rank at the top. *Engineering:* the ~15–22k-token backward is made to fit a single GPU by gradient checkpointing in train-mode (Qwen2.5 has no dropout, so forward numerics are unchanged).

2. **Attention route potential `r_u` — the honest negative.** For a head at the decision position with $o=\sum_u a_u v_u$ and $d_\phi=\nabla_o\phi_q$, the saturation-robust quantity is
$$r_u=d_\phi^\top(v_u-o),$$
the exact derivative of redistributing attention mass toward source `u` (never divide by the near-zero attention probability $a_u$). *Motivation:* a barely-attended source can still carry a consequential route. *Result:* on this organism `r_u` **loses** to saliency (it reads a single decision position, while saliency aggregates all paths across layers/positions). We report it as a negative, not a new gadget.

3. **SwiGLU gate sweep (prepared).** For $a_j=\mathrm{SiLU}(s_j)u_j$, replace $\mathrm{SiLU}(s_j)\to\mathrm{SiLU}(s_j+\beta)$ to expose a gated, task-dependent write that ordinary gradients miss in the negative region.

4. **GAC — Gate-Assistance Continuation (Track C, hard, prepared).** Instead of asking "does the gate fire?", ask "**how much internal assistance `β` must I add for it to fire, per context?**" For an ideal selector, $\beta^*(c)=\mathrm{logit}(\tau)-s(c)$ — **graded even when the gate is shut**, so contexts can be *ranked* before any one of them naturally triggers. Then descend $\nabla_c\beta^*=-\nabla_c\phi_q/\partial_\beta\phi_q$ via task-valid context edits toward a context that fires at `β=0` — a **natural trigger recovered blind**. This is the original ambition; it is genuinely hard for a *literal* trigger (no smooth path to an exact token), so we hold it as a labeled extension and report no-crossing/utility-limited outcomes honestly (never invent a finite censored value).

## 3.5 The causal tests and the evidence ladder

A gradient ranking is a *lead*, not proof. We climb a ladder, and the paper's claim is only as strong as the rung reached:

| Level | Test | Conclusion licensed |
|---|---|---|
| **Association** | a score predicts suspicious trajectories | a correlation on the measured distribution |
| **Local intervention** | ablate/replace the recovered span → `log p(payload)` drops vs a matched **random** span | that span causally affects the *score* |
| **Action-level necessity** | free-generate from trigger prefix vs span-ablated prefix → does the **exfil action** stop? (intercept-only, never executed) | the span causally controls the *behavior* |
| **Natural source control** (Track C) | a valid input edit reproduces it on the unmodified model | reachable through context |
| **Executable confirmation** | the recovered witness fires on the original vLLM serving path | a concrete, real failure |

Necessity is always reported **against a matched random-span control**; sufficiency/transplant and restoration arms separate a specific effect from generic disruption. We keep the random baseline, every denominator, and all zero/censored outcomes.

## 3.6 The monitor extension (the counterfactual-supervision idea)

To go from offline auditing (Track A) to a live pre-action monitor (Track D): a frozen score $S_\omega(H_t)$ computed **before** the next action, calibrated on held-out benign episodes by a **split-conformal** rule so that $\Pr(\exists t: S_\omega(H_{\text{new},t})>c_\alpha)\le\alpha$ (marginal, per-episode). The learning trick that makes it trainable under suppression: supervise a small predictor on the **measured source effects** $\mathbf d(H,S)=(D_1,\dots,D_{|\mathcal Q|})$ from the expensive auditor — dense counterfactual labels available on *every* trajectory — instead of on rare behavior labels. Then compare, at equal cost and false-alarm budget, a monitor trained on behavior labels vs. passive features vs. these counterfactual-effect targets.

## 3.7 Discovery accounting: acquisition vs confirmation (what the paper must actually measure)

The review's decisive framing. A verified finding factorises:

$$\Pr(\text{verified finding})=\underbrace{\Pr(\text{a viable candidate is reached})}_{\text{acquisition — the open problem}}\times\underbrace{\Pr(\text{verified}\mid\text{viable candidate})}_{\text{confirmation — what R1–R15 measure}}.$$

**Almost all our results so far are the second factor.** The award-level contribution is to **raise and measure the first** — and to show that *causal* feedback raises it per unit cost more than output-only search. A clean discovery bound exposes the missing assumption: if, given any no-finding audit history, the next budgeted experiment has ≥`ρ` chance of reaching a reachable witness and ≥`ν` conditional confirmation chance, then $\Pr(\text{no verified finding in }B)\le(1-\rho\nu)^B$. Weights alone do **not** supply a positive `ρ`; our empirical job is to show causal feedback increases witness yield per cost (variable-cost runs need a cost-indexed analysis, not "one experiment = one unit").

**Model-level calibration (and why AUROC is not an FPR).** Freeze the entire search (randomisation, access track, budget, scorer, stopping rule) to a single model score $T_B(M)$. With `n` exchangeable null models a conservative rank p-value is $p(M)=\frac{1+\sum_i\mathbf 1\{T_B(M_i^0)\ge T_B(M)\}}{n+1}$. **Statistical-scale reality:** with zero false positives in `n` independent clean models the exact one-sided 95% upper bound is $1-0.05^{1/n}$ — 95% at n=1, 13.9% at n=20, 4.87% at n=60. So a **<5% model-level FPR needs ≥59** clean models with zero flags, **<1% needs ≥299**; correlated checkpoints and 8 samples from one prefix do not count as independent units. A probe/outlier AUROC on ~55 balanced examples is a *feasibility* number, never a deployment FPR. **AUROC must never appear inside the deployed model score** (it needs labels the blind test checkpoint does not have) — it is evaluator-only analysis.

---

# PART IV — CURRENT EXPERIMENTS AND RESULTS

All runs use the real PersistBD harness; code in `experiments/gac/` and `src/audit/`. One driver, `experiments/gac/battery.py`, loads a model once and runs four hypotheses (model-load dominates these short jobs, so batching per model is the real speed-up).

## 4.1 The E1 gate: honest bookkeeping first

Before scaling we hardened the trigger-span resolver (`localize.py::find_span`, three fallback strategies) and now report the **full denominator** (assigned / resolved / unresolved, with per-row reason and the matched string). Verified CPU-only: **resolution 30/30 = 1.00** (was ~15/30). ~half of raw eval rows carry the trigger, so `--n 110` yields ≈55 resolved rows.

## 4.2 The battery (H1–H4) and what each tests

| H | Maps to | Question | Metric |
|---|---|---|---|
| **H1 fire** | detectability | does the trigger raise the payload's conditional likelihood? | median `Δ = log p(payload\|trigger) − log p(payload\|neutral)` |
| **H2 phi** | obligation score | does `φ_q` move with the trigger? | median `φ(trigger) − φ(neutral)` |
| **H3 localize** | the headline | is the trigger blindly top-ranked **and** causally necessary? | top-percentile (0=top), Hit@1, recall@k, necessity vs random span |
| **H4 action** | action-level necessity | does ablating the recovered span **stop the exfil action**? | exfil-emission rate, trigger vs span-ablated (intercept-only) |

## 4.3 Results (fresh, scaled — completed 2026-10-01)

**`swe-audit-3b-02` (undefended, behavioral TPR 1.00), n = 55 resolved (resolution 1.00):**

| H1 fire | H2 phi | H3 top-pct | H3 Hit@1 | H3 recall@k | H3 necessity (trigger vs random) | **H4 action (trigger → ablated)** |
|---|---|---|---|---|---|---|
| **+9.65 nats**, frac 1.00 | +2.18 | **0.000** | **0.527** | 0.48 | **−6.80** vs −2.8e-5 nats | **1.00 → 0.00** |

Reading it: the single highest-saliency token in a ~20k-token trajectory **is** a `# |TEST MODE|` token in **53%** of trajectories (median top-percentile 0.000); ablating the blindly-recovered span removes the payload preference by ~6.8 nats while a random span does nothing; and under free generation the exfil command is emitted **100%** of the time with the trigger and **0%** after ablating the recovered span. That last line is **action-level causal necessity**, not just a likelihood effect.

**Benign control `Qwen2.5-Coder-3B-Instruct` (never backdoored), n = 30 — the dissociation:**

| H1 fire | H3 top-pct | H3 Hit@1 | H3 necessity (trigger vs random) |
|---|---|---|---|
| +0.25, frac 0.70 | 0.074 | **0.00** | +0.35 vs −0.10 (not specific) |

The *same* `# |TEST MODE|` comment is **not** specially salient, **never** top-ranked, and **not** causally necessary on a clean model. So the signal is **backdoor-specific**, not "comments look suspicious."

**Defended `swe-audit-3b-01` (benign-SFT, behavioral TPR 0.21), n = 55 resolved (resolution 1.00) — the dissociation rung:**

| H1 fire | H2 phi | H3 top-pct | H3 Hit@1 | H3 recall@k | H3 necessity (trigger vs random) | H4 action (trigger → ablated) |
|---|---|---|---|---|---|---|
| **+17.8 nats**, frac 1.00 | +6.83 | **0.000** | **0.473** | 0.44 | **−16.1** vs −0.01 nats | 0.00 → 0.00 |

This is the headline in its sharpest form. As behavioral TPR falls **1.00 → 0.21**, the internal signal does not weaken — **necessity grows (−6.8 → −16.1 nats) and fire grows (+9.6 → +17.8 nats)**, exactly as Proposition 1's corollary predicts. The trigger is still the single most salient token in ~47% of trajectories (top-percentile 0.000), and ablating it collapses the payload preference by ~16 nats while a random span does nothing. The **action** row reads 0.00 → 0.00 because the defense has already suppressed *greedy free generation* of the exfil (consistent with TPR 0.21) — which is the whole point: **behaviorally the model looks clean, mechanistically the trigger is fully exposed and causally necessary.** (Honest caveat, per the Phase-2 review: a 0.00 → 0.00 action result means *no observed attack in either arm under greedy decoding* — it does **not** by itself establish action-level necessity on the defended model; the necessity claim there rests on the −16-nat mechanistic effect, and we should reproduce the real 0.21 rate under the organism's actual sampling/harness before relating the two numbers. Likewise Hit@1 ≈ 0.5 means the trigger is the single most-influential token in about *half* the trajectories, not all — we report the full integer-rank distribution, not only the rounded percentile.)

**The 3B ladder, side by side (behavioral↓ / mechanistic-flat-or-↑):**

| Model | Size | Behavioral TPR | fire Δ (nats) | blind top-pct | Hit@1 | necessity (trig vs random) | action trig→ablated |
|---|---|---|---|---|---|---|---|
| `3b-02` (undefended) | 3B | **1.00** | +9.65 | 0.000 | 0.53 | −6.8 vs ~0 | **1.0 → 0.0** |
| `3b-01` (defended) | 3B | **0.21** | +17.8 | 0.000 | 0.47 | **−16.1** vs ~0 | 0.0 → 0.0 |
| `7b-02` | 7B | **0.20** | +15.1 | 0.000 | 0.31 | −13.4 vs ~0 | **0.4 → 0.0** |
| `7b-01` | 7B | **0.20** | +14.8 | 0.000 | 0.33 | −13.8 vs ~0 | **0.5 → 0.0** |
| base (control) | 3B | — | +0.25 | 0.074 | **0.00** | +0.35 (not specific) | — |

**The ladder is complete across both sizes** (n=55 each, resolution 1.00): blind top-percentile stays ≈0 and necessity stays large (−6.8 to −16.1 nats) across behavioral TPR **1.00 → 0.20**, on 3B *and* 7B. Hit@1 is lower on 7B (~0.31 vs ~0.50) — the trigger is top in about a third of trajectories — but the median top-percentile is still 0.000, so it sits *at* the top even when not exactly #1. Crucially, the **7B models establish action-level necessity directly**: greedy free generation emits the exfil **40–50%** under the trigger and **0%** after ablating the recovered span — removing the defended-3B ambiguity (whose 0.0 → 0.0 was null-in-both-arms under greedy, §above). *Caveat (3B vs 7B are separate strata — four model IDs are not four independent lineages; the trigger-diverse organism zoo now being built gives genuinely independent lineages.)*

## 4.4 Honest negatives (kept in the paper)

- **`r_u` loses to saliency** (top-pct ~0.33/0.20 vs 0.000). The mechanistic route is not a better operator here; plain saliency wins. Contribution = setting + finding, not a gadget.
- **GAC β* on this organism** was dominated by censoring (the blind steering direction never opened the gate on neutral contexts within the tested range). Correctly recorded as right-censored, not an invented number. Reinforces that β*-continuation is the hard Track-C problem for a literal trigger.

## 4.5 The reframed thesis the data supports

> **A benign-fine-tuning "defense" that suppresses a coding-agent backdoor to 21% behavioral TPR leaves its trigger fully exposed mechanistically** — blindly recoverable at the top of a ~20k-token trajectory, causally necessary at the *action* level (ablating the recovered span stops the exfil 100%→0%), and absent in a benign control. The contribution is this **detection/localization-under-defense** result and the **behavioral-vs-mechanistic dissociation** it exposes — explained by Proposition 1 — **not** a new attribution operator.

**Extended by the forward-detection results (R13–R15):** the exposure is also **forward-only** — a single mid/late-layer **recognition axis** detects the triggered input at 0.96–0.98 AUROC on 3B under an *honest label-free fixed-depth rule* (R14), **survives benign defense** (defended 3b-01 = 0.957, R13), transfers across held-out tasks as a rank-1 subspace (R15), and the clean model reads chance. This turns the offline forensic result into a **deployable forward monitor** on the models where the fixed-depth rule holds.

**Claim-discipline corrections carried from the review (so the paper survives scrutiny):** (i) R11/R12's near-1.0 AUROCs used test-set layer selection — cite the **de-leaked** R14/R15 numbers as the real ones; (ii) "defense *strengthens* the signal" (R1) compares *separate released checkpoints*, not a within-lineage treatment — do not plot one defense-intensity trajectory until P5's genuine ladder; (iii) teacher-forced nats are **not** observed risk — report absolute likelihoods, the first divergent action-token score, and free-generation outcomes separately (a 0.00→0.00 greedy "action" row is null-in-both-arms, not action-necessity); (iv) the "one axis reads+detects+suppresses" unification is earned for **readout+transfer** (R15), not yet **causally** (projected-replacement is P4); (v) the recognition axis is **literal-specific** (R16) — no generalization-to-unseen-mechanism claim until P6.

---

# PART V — THE EXPERIMENTAL PROGRAMME (the award-paper plan)

**This is the paper's experimental spine.** Live per-experiment status is tracked in the **Master experiment TODO** table in `Experimental_Results.md`; the structure and rationale below are the self-contained plan. The programme is organised around the one question the paper answers (§0.5) and the surviving novelty boundary (§2.3) — every leg exploits a specific drawback of prior work (`Prior_Work_and_Novelty_Boundary.md`).

**The award thesis in one line:** *the first method to recover/localize and forward-detect a coding-agent backdoor trigger from **causal** internal evidence **after benign post-training has suppressed the behavior** — where BAIT and Haystack go blind — via **input-side closed-loop acquisition** that yields a **valid natural triggering context verified by free generation** (the bar Qin et al. fail), with **action-level causal necessity** and **marker-free** recognition, under leakage-free, specificity-controlled, generalization-tested evaluation.*

**The nine legs (what each decides, and its status):**

- **Leg 0 — Setting & rigor.** De-leaked layer selection ✅ (R14); checkpoint registry ☐; prefix-only / evaluator-only hygiene ☐; integer-rank + absolute-likelihood + first-divergent-token reporting ◧. *Decides: which claims survive clean evaluation.*
- **Leg 1 — Specificity (the gating negatives).** Clean-model controls ✅; **benign-conditional organism (EXP-B)** ▶ — does the unsupervised axis fire on an *authorized* conditional policy while the causal-contract signal does not?; clean-null **FPR set** ▶; **benign-distribution-shift controls** ☐ (MAD-Quirky mandate); matched benign fine-tunes ☐. *Decides: are we a backdoor detector or merely a conditional-feature/shift detector.*
- **Leg 2 — Blind acquisition (the crux).** **Matched-cost head-to-head (EXP-A)** ▶ — our causal ranker vs the full baseline panel (Haystack output/entropy/composite, BAIT-style commitment, Anthropic probe, Mahalanobis) at equal cost, margin must **widen on defended models**; **natural reconstruction** to the Qin-bar ◧; our own recognition-direction proposer ☐; **full Haystack and BAIT reimplementations** as baselines ☐. *Decides: does the mechanism find unknown triggers better than output-only, blind and under suppression.*
- **Leg 3 — Mechanism value at matched cost.** Closed-loop causal-feedback vs output/passive/**shuffled** at a frozen FPR ☐; **AtP\*** screen→finite-verify top-K + random low-gradient sites + false-negative bound ☐. *Decides: does mechanistic information itself (not more compute) earn its cost.*
- **Leg 4 — Causal unification.** **Causal shared-axis** projected-replacement (EXP `cax`) ▶ — promotes R15's readout axis to a causal one; recognition↔execution **marker-free causal necessity** ☐. *Decides: is there one causal axis, established beyond Decontamination's correlational analysis.*
- **Leg 5 — Suppression / within-lineage.** Genuine defense ladder, four distinct outcomes ☐; **run Haystack+BAIT vs ours under suppression** ☐. *Decides: the central opening — competitors go blind, we do not.*
- **Leg 6 — Generalization.** **Sourced organisms** (Watch-Out-for-Your-Agents, BALD, AgentGhost); hold out whole recipes/target families ☐. *Decides: does it transfer beyond the one literal cue (the R16 worry).*
- **Leg 7 — Robustness.** Benign OOD, contamination, distributed/stateful, **adaptive/monitor-aware** organisms ☐. *Decides: the honest failure boundary.*
- **Leg 8 — Monitor / mitigation (secondary).** Forward-monitor **lead-time** ☐; steering-defense **utility cost** ☐. *Decides: a deployable, selective defense — only after the offline auditor convinces.*

**Binding evaluation mandates (from the close readings, §2.3):** reimplement every baseline and compare at **matched compute**; **leakage-free** selection everywhere; **benign-distribution-shift** controls on every detector; vary install intensity **independently** of benign post-training and hold out whole **recipes**; a discovered trigger counts only when a **natural** context fires the **unmodified** model under the real decoder; a model-level FPR needs the right **statistical scale** (≥59 clean models for <5%).

---

## 5.0 (Legacy) Phase-1-centric ordering

*Retained because each item maps into a leg above; superseded for prioritisation by the leg structure.* Priority order (most likely to change the scientific conclusion first):

1. **The intensity ladder (E3 / the central figure).** Run H1–H4 across the full ladder — `3b-02` (TPR 1.0) → `3b-01` (0.21) → `7b-02` (0.20) → `7b-01` (0.20) — treating 3B and 7B as **separate strata** (model IDs and different TPRs are not one suppression axis). **The sharp, falsifiable claim:** as behavioral TPR falls, median top-percentile stays ≈0 and necessity holds or *grows* (Prop 1 corollary). The behavioral↓ / mechanistic-flat curve **is** the paper's Figure 1. *Narrows the paper* if top-percentile drifts upward with suppression.

2. **Remove exact-target knowledge (E4, H5).** Re-score H3 with the attribution target swapped: (a) exact payload (ceiling), (b) semantically-equivalent payload variants (different endpoint/arg, same `<function=bash>` structure), (c) a **contract-level** probe naming no planted literal. *Question:* is localization tied to the memorized string or to a general "unauthorized egress" policy? This is what moves us from Track A toward Tracks B/D. *Narrows the paper* if contract-only probes lose the trigger.

3. **Action-level causality under realistic controls (E5, H7a).** Confirm the recovered span on the **vLLM serving path**: ablate it there → the emitted action stops being the exfil call; intercept and parse the tool call, **never execute** it. Add removal/replacement/restoration/transplant arms with length-and-role-matched random spans and benign-but-security-vocabulary controls.

4. **Pre-action localization (E2, H7b) → the monitor (E12 / Track D).** Recompute H3 at successive history cutoffs from just after the untrusted observation to just before the first unauthorized action — does the signal exist *before* the payload is teacher-forced? A frozen pre-action probe that localizes is the monitoring result; calibrate episode maxima by split-conformal and report lead time + false-alarm budget + utility cost. **This is the entry point to Phase 2 (Part VII)** — the deployable-monitor program — whose immediate milestone is **P2-0**: can cheap forward-only features predict the Phase-1 *causal effects* at all?

5. **Matched-cost baselines (E9).** Blind saliency vs Patcher-style grouping, black-box occlusion, a behavioral investigator, and a Haystack-style extractor — all at identical total cost (count backward passes, prefixes, executions), especially on the **defended** model. Does internal access *earn* its cost?

6. **Bounded route follow-up (E10, H6).** One preregistered attempt to rescue `r_u` (aggregate over all payload tokens, sum across layers/heads) on the *same* histories; if still weaker, keep the negative and stop.

7. **Specificity + rare-legitimate behavior (E8); conjunctive/temporal triggers (E7); cold acquisition / GAC (E11, Track C).** Matched benign fine-tunes (not just one base) as controls; vary trigger distance/order/carrier; and the hard blind-discovery experiment, kept explicitly as an extension until it produces a natural witness.

**Reporting discipline on every run:** full rank distribution and denominator; necessity as a paired change beside its random-span control; 3B and 7B never pooled; cross-model uncertainty at the number of independent lineages (four organisms, two sizes), not thousands of correlated token observations; censored/zero outcomes kept.

## 5.1 Benchmark context — our role as candidate method #3 (added 2026-10-02)

The team is standing up an **Agent Backdoor Auditing Benchmark** (`Agent Backdoor Auditing Benchmark .md`, git-ignored): **model-level** auditing — decide whether a third-party tool-using model hides a backdoor, with **weights only** (no trigger, no target, no poisoned data, **no clean counterpart**). Three candidate methods are compared: **BAIT** (invert the attack *target*), **The Trigger in the Haystack** (reconstruct the trigger), and **"trigger searching through mechanistic interpretability" — which is us.**

**What this adds to our plan (we keep improving the method first; full reframe later):**
- **Evaluation axis shifts to model-level:** detection rate, **false-positive rate on clean models**, and **compute cost**, under a fixed budget — compared head-to-head with BAIT/Haystack. Trigger recovery + behavioral verification reported separately. (Our localization/necessity metrics become the *confirmation* stage, not the whole story.)
- **Our natural benchmark role:** a **proposer→confirmer** pairing — a cheap proposer (BAIT/Haystack/memorization, or our own recognition-direction search) nominates a candidate trigger/target; our mechanistic signal (forward-readability + causal necessity) **confirms it causally and localizes it**, which should cut false positives versus a proposer alone. This is a genuinely novel benchmark contribution (mechanistic confirmation reducing FPR), and it sidesteps the fact that our current signal needs a candidate.
- **Diverse backdoor coverage comes from the literature, not hand-rolling:** the benchmark sources a trigger×target matrix — PersistBD (lexical/fixed), **Watch Out for Your Agents** (query/observation/thought attacks; *fine-tunable by us*), **BALD** (word + scene/semantic), **AgentGhost** (composite goal+state). This is a cleaner path to the trigger diversity our transfer study (Leg 3) needs than the stalled home-grown zoo.

**Benchmark-aligned experiments added to the pipeline:**
- **B1 — model-level detection score + FPR.** Formalize a single per-model backdoor score from our signals (forward-probe AUROC gap × causal-patch effect) and measure it on **backdoored vs several clean/benign models** → detection-vs-FPR. (`experiments/phase2/detect.py` aggregates existing outputs; FPR needs ≥2–3 clean models.)
- **B2 — BAIT / Haystack baselines at matched cost** → the head-to-head the benchmark requires; then the **BAIT-proposes + mech-confirms** combination.
- **B3 — coverage via sourced organisms** (Watch-Out-for-Your-Agents first: fine-tunable, gives lexical + semantic + thought triggers) → replaces the hand-rolled zoo for the transfer study.
- **B4 — harder-setting detection:** rerun our detector on the PersistBD **first-position** eval (not just random-position) to probe robustness across trigger placement.

---

# PART VI — RUNNING AND QUEUED EXPERIMENTS (live status, 2026-10-01)

Both clusters share the `/work` storage and the four organisms; we run on whichever has free GPUs.

## DeltaAI (GH200, account `bfgp-dtai-gh`) — the 3B ladder completed here

| Job | Model | State |
|---|---|---|
| `3284660` b-3b02 | swe-audit-3b-02 | ✅ **COMPLETED** → §4.3 |
| `3284661` b-3b01 | swe-audit-3b-01 (defended) | ✅ **COMPLETED** → §4.3 (the dissociation rung) |
| `3284664` b-base | Qwen2.5-Coder-3B control | ✅ **COMPLETED** → §4.3 |
| `3284662` b-7b02 | swe-audit-7b-02 (7B stratum) | ⏳ PENDING — 7B saliency needs the 96 GB GH200 |
| `3284663` b-7b01 | swe-audit-7b-01 (7B stratum) | ⏳ PENDING — 7B stratum |

## Delta (A40 nodes, account `bfgp-delta-gpu`) — duplicates, as a cross-hardware check

| Job | Model | Node | State |
|---|---|---|---|
| `22603537` d-base | Qwen2.5-Coder-3B control | gpub048 | ✅ COMPLETED (reproduces the DeltaAI control) |
| `22603535` d-3b02 | swe-audit-3b-02 | gpub018 | ⏱ TIMEOUT at 1 h (A40 slower than GH200; wrote fire/phi only) |
| `22603536` d-3b01 | swe-audit-3b-01 | gpub036 | ⏱ TIMEOUT at 1 h (same; redundant — DeltaAI has the full run) |

**Status:** the **entire 3B ladder + control is complete** (DeltaAI), giving the behavioral↓ / mechanistic-flat-or-↑ dissociation in §4.3. The two Delta 3B jobs hit the (too-tight) 1-h A40 walltime and only confirmed the fire step before being killed — no loss, since DeltaAI carried both 3B models to completion; the lesson is to budget ≥1.5 h for A40 battery jobs. **The only remaining piece is the 7B stratum** (`b-7b02`, `b-7b01`), pending on DeltaAI's GH200 — 7B saliency OOMs on Delta's 40/48 GB GPUs, so it runs only on the 96 GB GH200 (or Delta's `gpuH200x8`, 141 GB, at CF 3.0). Completing them finishes **Part V item 1, the intensity-ladder figure**.

---

# PART VII — PHASE 2: FROM MECHANISTIC AUDIT TO A DEPLOYABLE PRE-ACTION MONITOR

**Framing.** Parts I–VI are **Phase 1** — discover the trigger's mechanistic signature and characterize it *causally*. That is the science, not yet the product. **Phase 2 is what matters operationally: distill the expensive, offline, gradient-based auditor into a cheap, forward-only monitor that runs *during deployment*, flags the trigger before the agent acts, and blocks the unauthorized tool call**, at a controlled false-alarm rate. Phase 1's real deliverable for Phase 2 is **not a heatmap** — it is a *labeled causal dataset* (which input spans causally control which violations, at which internal sites) that supervises the monitor.

## 7.1 The claim Phase 2 tests

> **Counterfactual "source-effect" labels — how much an untrusted input span changes the agent's action preferences — supervise a compact pre-action monitor better than rare observed failures do; and mechanistic analysis identifies which representations carry that information and whether it generalizes.**

This is the operational payoff of Proposition 1: behavioral failures are *suppressed* (so they are almost absent as training labels), but the *causal source-effect* is measurable on **every** trajectory, firing or not — so the monitor can be trained exactly where behavior labels are useless.

## 7.2 Closest prior monitors (what Phase 2 must beat, and not reinvent)

| Work | What it does | Consequence for us |
|---|---|---|
| **TRACES** (2605.27690) | observer-LLM hidden-representation *proactive* trajectory-risk auditor, prefix-level | the closest representation-monitor baseline; we add **causal** feature selection + **source-effect** labels under suppression |
| **AgentLens** (2606.22673) | step-level last-token probes + mechanistic steering for coding agents, with syntax-matched benign controls | the closest ordinary activation monitor — so neither pre-action coding probing *nor* syntax-matched controls is itself new |
| **ProbGuard / Pro2Guard** (2508.00500) | symbolic-state DTMC runtime risk prediction, intervene above a threshold | a strong *non-representation* proactive baseline |
| **Deception linear probes** (Apollo 2502.03407; Caught-in-the-Act 2508.19505) | residual-stream probes catch deception at ~1% FPR | cheap probes *can* work — **and collapse under adaptive pressure** (suffixes, RL-obfuscation), which defines our robustness track |
| **Agent Safety From Within / TACIT** (2609.33039) | frozen-rep trajectory assessor; separates harmful content from unsafe tool use | authorization/provenance controls; its pooling is a baseline at our pre-action cutoff |
| **BAIT** (S&P '25) | backdoor **scan by inverting the target** (black-box), exploiting target-token causal structure | a strong *discovery* baseline; directly relevant to our payload-blind / contract direction |
| **Geiger — Causal Abstractions** (NeurIPS '21) | a probe can decode a variable the model **does not use**; interchange interventions test *causal* structure | the theoretical reason Phase 2 selects features by **intervention outcomes, not probe accuracy** |

## 7.3 The monitor has two outputs, kept strictly separate

1. **Source-effect head** `D̂_q(H,S)` — predicts the retained causal influence of an untrusted span `S` on the violation score for obligation `q`.
2. **Action-risk head** `Q̂_q^(K)(H)` — predicts an actual unauthorized action under the real decoder within horizon `K`.

They must not be merged, because **a defended checkpoint can have a large source-effect and a near-zero action-risk** (exactly our 3b-01: mechanism intact, behavior suppressed). Conflating them would relabel a suppressed-but-loaded backdoor as "safe."

## 7.4 The labels (two different quantities)

- **Behavioral risk over a horizon:** `Q_q^(K)(H_t) = Pr(∃ j∈[t,t+K): V_q(a_j,H_j)=1 | H_t)` — needs the real decoder + harness; the independent predicate `V_q` judges it.
- **Offline auditing score:** `φ_q(H) = log p(y_q^-|H) − log p(y_q^+|H)` (Part III), with exact-payload likelihood kept as a **separately-labeled teacher ceiling**.
- **Causal source-effect (the dense label):** `D_q(H,S) = φ_q(H) − (1/R) Σ_r φ_q(H^{S←r})`, replacing span `S` with `R` task-valid, role/length-matched alternatives. Keep the full **signed vector** `D=(D_1,…,D_Q)`; log matched random-span and legitimate-action effects beside it. A causal effect on a model *score* is not automatically an effect on *action* probability, so on a validation subset also measure behavioral necessity `N_q^(K)(H,S)`.

Counterfactual labels are measurable even when a rollout doesn't fail — but they are only informative about the sources/obligations you actually test; they create **no** evidence for an absent, unencountered trigger.

## 7.5 The first gate: can a cheap forward-only monitor even see it? (P2-0, observability)

Phase 1's winning signal is a **gradient toward a supplied payload**; a deployed monitor gets only a few **prefix activations** — a *different, smaller* information set. Whether the audit information survives into deployable features is empirical, with a precise bound:

`E[(D − g(Z))²] = E[Var(D|Z)] + E[(E[D|Z] − g(Z))²].`

The first term is **irreducible** — no bigger network removes it. So the first Phase 2 experiment (**P2-0**) is: do prefix-only representations predict held-out measured effects `D` *at all*, and do they beat cheap input-embedding and output-score baselines? If not, cheap passive monitoring is unsupported by the current Phase 1 success, and we keep the expensive auditor (or a two-pass monitor). Counterexample to watch: a cue stored in **KV** and retrieved only at argument-generation would be visible to input attribution yet *absent* from the pre-tool decision vector.

## 7.6 Phase 1 → Phase 2 handoff: deliver an intervention dataset, recover sites bidirectionally

Phase 1 must output, per development history: an **input pair** (original + ≥2 valid replacements of the predicted span + a restoration arm + a matched random-span arm); **prefix states** at the source span, last observation token, and final prompt token (exact positions, **forward-only** — student features never come from the teacher-forced suffix); the **effect vector** `D` + selected internal-replacement effects + behavioral outcomes with an explicit decoder; **controls** (same action vocabulary under legitimate authorization, quoted security text, near-match comments, rare benign commands, unrelated high-saliency spans); and **provenance** (task family, paired-group, lineage, algorithmic-vs-curator span selection). Crucially, **do not keep only successful localizations** — failed proposals and ordinary task-important spans are part of the monitor's distribution.

Sites are found by **bidirectional within-checkpoint replacement** (no clean ancestor needed — both histories run through the same suspect): removal `κ^rem_{m,q}=φ_q(H⁺)−φ_q(H⁺; h_m←h_m⁻)` and restoration `κ^add_{m,q}=φ_q(H⁻; h_m←h_m⁺)−φ_q(H⁻)`. Screen with gradient×activation-difference, then **validate the top candidates *and* a random sample of apparently-unimportant ones with finite swaps** (guards against attribution false negatives — AtP\*). A component that is causal but **unspecific** (gates *all* shell commands, say) must be caught by measuring legitimate-utility disruption alongside the effect — *causal intervention and preserved utility are measured together.*

> **Why causal, not just a probe:** Geiger et al. build a representation a probe decodes perfectly yet interchange interventions show plays **no causal role** — the information is present but unused. That is exactly why Phase 2 selects features by **intervention outcomes**, and why a strong passive probe alone would not settle the mechanism.

## 7.7 The monitor itself (smallest first)

Project selected states into frozen site subspaces `U_ℓ`; build source features `c_S` (pooled over the span, mean + max) and decision features `c_b`; fit the small interaction model

`D̂_q(H_t,S) = a_q·c_S + b_q·c_b + c_S^T M_q c_b + v_q·meta(S,H_t) + b_{0q}`, with `rank(M_q) ≤ 4`,

compared against the additive version (`M_q=0`), a raw-state ridge, a 64-unit MLP, and a **lexical baseline that just searches for the discovered cue**. The interaction term `c_S^T M c_b` tests the mechanistic hypothesis — *does a source's meaning matter conditionally on the current decision state?* The **action-risk head is separate** (pooled source-effect predictions + decision features + explicit authorization metadata) and needs behavioral labels; if no positive actions occur in the defended development runs, fit/report the **effect** monitor first (an imminent-failure classifier can't be validated on all-negative data). A monitor is **never** handed the true trigger span at test time; the same source segmentation runs at train and deploy.

**Supervision** (equal feature budget + capacity across arms): `L_effect` = Huber on `D`; `L_risk` = BCE on `Q`; optional `L_difference` for a score-potential head (supervise `v_q(Z(H))−v_q(Z(H'))` with `φ_q(H)−φ_q(H')`, **not** the effect head). The decisive ablation is **labels × representation**: {behavior-only, effect-only, joint} × {raw states, ordinary probe, PCA/SVD, intervention-trained subspace}.

## 7.8 The decisive experiments (P2-0 … P2-10) and the honest decision rules

| # | Tests | A result that changes the conclusion |
|---|---|---|
| **P2-0** observability | predict `D` from prefix features vs input/score baselines | failure ⇒ cheap passive monitoring unsupported |
| **P2-1** causal sites | removal/restoration at discovered vs random vs benign ops | selectivity ⇒ mechanism-aware selection helps; generic tool disruption ⇒ it doesn't |
| **P2-2** label ablation | behavior-only vs effect-only vs joint, equal capacity | tests whether counterfactual labels add value beyond more data |
| **P2-3** feature ablation | raw vs probe vs SVD vs intervention-subspace, equal rank | tests whether *mechanism* (not just prediction) helps |
| **P2-4** genericity | matched syntax / authorization / quoted / benign-specialist / near-match cue | source-to-policy relation vs surface correlate |
| **P2-5** teacher knowledge | exact vs variant vs contract candidates | dependence on the planted payload (**ties to our running H5**) |
| **P2-6** suppression | matched within-lineage defense checkpoints | does effect-supervision keep value as failures vanish |
| **P2-7** lead time | pre-exposure / post-observation / pre-action / pre-execution cutoffs | *when* useful info appears |
| **P2-8** transfer | hold out tasks → trigger identity → payload family | deployment usefulness vs memorizing one cue |
| **P2-9** extra value of internals | text-only / lexical / occlusion / score / combined | does internal access improve the cost/false-alarm frontier (Black-to-White) |
| **P2-10** robustness | long contexts, benign shifts, monitor-aware organisms | bounds the claim — adaptive attacks *can* defeat activation monitors |

The **double dissociation** that would make the paper: source-removal reduces predicted/actual backdoor influence *without* equally suppressing authorized operations, while authorization changes alter risk *without* erasing cue recognition. **Negative outcomes are conclusions:** source-only matching the full model ⇒ cue memorization; a task/action probe matching it ⇒ generic semantics; raw-state matching the causal subspace ⇒ mechanism selection was unnecessary *for prediction*; strong effect-prediction with weak hazard-prediction ⇒ an **auditor**, not a validated imminent-action monitor.

## 7.9 Evaluation, calibration, serving, and the honest ceiling

- **Pre-action metrics:** episode recall before the violation + lead time (tool calls/tokens); **false alarms per benign episode** + utility loss under the intervention policy; recall of auditor-validated influential sources on suppressed histories; end-to-end wall-time, memory, extra prefills/backwards, and fraction escalated to the expensive auditor.
- **Calibration:** freeze the action-risk score, calibrate on `n` exchangeable benign episodes by episode-max split-conformal (`c_α = M_{(⌈(n+1)(1−α)⌉)}`) for a marginal per-episode false-alarm bound `α`. With **zero** false alarms in `n` benign episodes the one-sided 95% bound is `1 − 0.05^{1/n}` — about **299** clean episodes to get under 1%. A high AUROC on 55 balanced examples is **not** this.
- **Serving:** the instrumented PyTorch checkpoint is the offline teacher (gradients + replacements); the frozen student reads selected prefix activations, ranks sources, and optionally requests 1–2 **budgeted** counterfactual replays when risk warrants. **Intercept-only** (parse/log the proposed action, never execute). Pin weights/tokenizer/template/dtype/positions/decoder and recalibrate on the **real** extraction path (vLLM `extract_hidden_states` exports *prompt* states only — not autograd or arbitrary intervention; don't silently change the serving stack under the reported results).
- **Robustness / ceiling:** activation monitors can be **defeated adaptively** — deception probes collapse under suffixes and RL-obfuscation while behavior stays deceptive (Apollo; Obfuscated Activations; Neural Chameleons). So Phase 2 carries a **separately-labeled robustness track**, and the deployed monitor **abstains** on the provably-unelicitable class rather than overclaiming.

## 7.10 The immediate Phase 2 milestone (uses data we already have)

The 55-example 3B sets can answer a *feasibility* question (not a low-false-alarm deployment claim). **P2-0 + a small P2-1:** save prefix-only activations at ~6 layers (+ input embeddings as a lexical baseline) at the last-observation and final-prefix positions; for each history score the top-8 algorithmic source spans and 8 matched random/task-important spans with 2 replacements each; fit **group-cross-validated ridge** predictors of signed effects and compare *decision-only / source-only / source+decision / input-embedding / teacher-score*; screen 20 pairs with gradient×difference then finite-swap the top-8 and 8 random sites with matched legitimate operations. **Advance only if useful effect-prediction survives lexical controls and some selective bidirectional internal effects replicate.** This one table — *does the Phase-1 teacher's information survive in cheap, forward-only, deployable features?* — can falsify the monitor quickly, before any dynamics model.

**Splitting discipline (so the claim is real):** every original history, its edits/donors/spans/rollouts/defense-checkpoint copies share a **group ID** kept within one split; use separate sets for mechanism discovery, monitor fitting, model selection, false-alarm calibration, and final test; for a *new-trigger* claim, exclude that mechanism from teacher generation and subspace selection too (a different endpoint string for the same operation is only a weak lexical variation). 3B and 7B get **separate** monitors unless an explicit functional alignment is shown — layer numbers are not shared coordinates.

**What would make Phase 2 a convincing contribution:** (1) *mechanism* — paired natural + internal interventions separate recognition, retained influence, and behavior suppression, with legitimate utility preserved; (2) *learning* — dense source-effect supervision beats behavior labels at matched features/data/compute; (3) *generalization* — gains survive meaningful holdouts and syntax/authorization controls, with the paper stating honestly whether the result covers a discovered cue, a contract family, or an unseen backdoor. First work item: **P2-0 + a small P2-1, then the label×representation ablation** — test that Phase 1 contains a learnable signal before committing to anything larger.

---

# APPENDIX — Primary-source ledger

**Method / baseline sources (literature gap, §II):** The Trigger in the Haystack (arXiv 2602.03085); CodeScan (2603.17174); Simple probes can catch sleeper agents (Anthropic); Backdoor Attribution (2509.21761); LLM Forensics (2609.07746); Activation-matched finetuning (2609.00351); Deep Causal Transcoding (turntrout.com/deep-causal-transcoding); CPE / Eliciting Latent Behaviors (2606.29604); Activation Oracles (2512.15674); Functional-attribution MAD (2604.18970); Layerwise Convergence Fingerprints (2604.24542); AgentLens (github EddyLuo1232/AgentLens); FreqRank (ACL Findings EMNLP 2025.1041); AuditBench (2602.22755); Structural Inference / susceptibilities (ICLR 2026); LoRAcles (OpenReview); Obfuscated Activations (2412.09565); Unelicitable Backdoors / cryptographic circuits (2406.02619); vLLM hidden-state extraction (vllm.ai 2026-03-30).

**Mechanism / theory sources (§III):** Patcher (USENIX Security 2026, usenixsecurity26-gao-anjun); AtP\* (2403.00745); GLU Variants (2002.05202); ABS (CCS'19); BadMoE (2504.18598); BadSwitch (2510.13462); When Are Experts Misrouted (2605.07260); Sparse Models Sparse Safety (2602.08621); RouterInterp (PMLR v306); Backdoor Decontamination Dynamics (2608.11295); Discovering Backdoor Triggers (LessWrong, kmNqsbgKWJHGqhj4g); conformal prediction primer (2107.07511).

**Phase 2 sources — monitoring, probing, scanning, causal abstraction (§VII; most surfaced 2026-10-02):**
- *Monitors / probes we must beat or adapt:* Patcher (USENIX Sec '26) [response-conditioned gradient localization + grouping]; AgentLens (arXiv 2606.22673) [step-level coding-agent probes + mechanistic steering]; TRACES (2605.27690) [proactive hidden-rep trajectory-risk auditor]; Agent Safety From Within / TACIT (2609.33039) [frozen-rep trajectory assessor]; ProbGuard / Pro2Guard (2508.00500) [symbolic-state DTMC runtime monitor]; Representation Transitions Reveal Emerging Safety Risks (2610.00400) [multi-turn representation-transition signal]; PreAct-Bench (2606.09890) and *What Can Be Enforced?* (2607.22868) [predictive-monitoring benchmark + enforceability theory]; "Agentic LLMs Secretly Encode Indirect Prompt-Injection Exposure in Hidden States" (2608.02657).
- *Deception/harm linear probes (feasibility + robustness ceiling):* Detecting Strategic Deception Using Linear Probes (Apollo, 2502.03407); Caught in the Act (2508.19505); Pressure-Testing Deception Probes (2605.27958); Beyond Linear Probes: Dynamic Safety Monitoring (2509.26238); Black-to-White Performance Boosts (2507.12691) [extra value of internals over observable evidence].
- *Backdoor scanning / target inversion (discovery baselines):* BAIT (S&P 2025) [invert the target, black-box]; ConfGuard (AAAI) [sequence-lock token-confidence monitor]; Inverting Trojans in LLMs (2509.16203).
- *Why features must be chosen causally, not by probing:* Geiger et al., Causal Abstractions of Neural Networks (NeurIPS 2021); How Reliable are Causal Probing Interventions? (2408.15510); Distributed Alignment Search / DAS (Geiger et al., CLeaR 2024); *Causality is Key for Interpretability to Generalise* (2602.16698).
- *Adaptive-attack ceiling:* Obfuscated Activations Bypass LLM Latent-Space Defenses (2412.09565); Neural Chameleons (2512.11949). *Calibration:* conformal primer (2107.07511). *Serving:* vLLM hidden-state extraction (vllm.ai, 2026-03-30).

*These Phase-2 references were surfaced by literature search; relevance is noted per entry, but full-text methods were not all independently reproduced.*

**Caveats:** all model-specific numbers are our own measurements on the `qiusizhan/swe-audit-*` organisms via the `uiuc-kang-lab/PersistBD` eval trajectories; the propositions are derived under explicit simplified assumptions and their applicability is an empirical question (hence the causal/finite tests); no command emitted by any model is ever executed.
