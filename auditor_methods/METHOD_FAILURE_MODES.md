# When and why runtime backdoor monitors succeed or fail on LLM agents

Analysis of the three runtime monitors we have fully evaluated (MAD, ConfGuard, LCF) across
the six backdoor families currently in the zoo. The goal is the paper's third contribution,
beyond the model zoo and the unified evaluation protocol: a mechanistic account of *when* and
*why* each existing method works or breaks on tool-using LLM agents, and which simple
adaptations plausibly close the gap.

All numbers are from `runtime_monitor_results_MAD_corrected.xlsx` (the reference-calibration
fix applied; every eval case scored, `n_test = n_cases`, no eval-normal leakage). Rows for the
newly staged families (PersistBD-Qwen3.5-9B, lfm25, GLM-4.6V, AgentGhost-AITZ) are running now
and will be folded in; the qualitative story below is already stable across 6 families and 2
model sizes.

## How to read the results

- **Blind AUROC** is the headline (fixed-depth layer chosen without labels). **Peek AUROC**
  (oracle label-selected layer) is reported only to show the ceiling.
- **Operating point** = TPR / FPR at a 5% clean-reference budget (MAD, LCF) or at the paper's
  fixed lock rule L=10,P=0.99 (ConfGuard). A method can *rank* well (high AUROC) yet be
  *uncalibratable* (operating-point FPR near 100%); we flag both, because they fail differently.
- **Role**: backdoored models carry positives (pos% = realized attack-success rate on the 250
  runtime inputs); clean controls have 0 positives, so their only number is FPR = false-alarm
  rate on a known-clean model.
- **Families**: `swe-audit`/PersistBD (code-exfil tool call, text trigger), `BALD-word`
  (HighwayEnv lane choice, text trigger), `BALD-scene` (driving scene trigger), `Watch-QA`
  (smartwatch, trigger in the *query*), `Watch-OA` (trigger in the *observation/context*), and
  the 9B VL variants of BALD.

---

## 1. MAD (cupbearer Mahalanobis, white-box decision-token activations vs a clean reference)

**Mechanism.** Fit a Gaussian on the decision-token residual activations of a separate clean
nominal reference set; score each eval input by Mahalanobis distance; threshold at a clean-FPR
budget. White-box, trigger/target-blind, needs a reference.

### Successes
- **Code-exfil (swe-audit/PersistBD): strong and robust.** Blind AUROC 0.768-0.979 across all
  10 checkpoints, TPR 75-100 at FPR 6-17%. Holds even when the backdoor is *suppressed* to a
  tiny firing rate: swe-audit-7b-05 at pos%=1.6 still gets AUROC 0.84.
- **Driving (BALD word/scene, incl. 9B VL): near-perfect ranking.** bald-3b 0.988, bald-7b 0.92,
  bald-word-9b 1.00, bald-scene 0.95-0.98.
- **Clean models stay quiet.** base-3b/7b FPR 6.8/7.2; persistbd-clean controls FPR 6.1-7.5.
  On genuinely clean models the calibrated threshold holds, so a flag is meaningful.

### Failures
- **Observation-poison (Watch-OA): below chance.** Blind AUROC 0.343 (3b) and **0.119 (7b)** -
  worse than a coin. This is the sharpest failure in the whole study. (The pre-fix 0.82 for
  Watch-OA-7b was an eval-normal leakage artifact; the honest number is 0.12.)
- **Query-poison (Watch-QA): weak.** AUROC 0.69-0.72, and the operating-point TPR is only
  5-28% at a 5% budget - the decision-token signal exists but is too faint to act on.
- **FPR inflation on low-entropy domains.** On driving the calibrated FPR climbs to 22-24%
  (vs 6% on code), because the clean reference is narrower than the eval distribution.

### Why
MAD only sees the backdoor if it leaves a *decision-token* activation shift that is anomalous
relative to the reference. Code-exfil and driving both compute the defection at (or just
before) the emitted action, so the probe location is right. Watch-OA fails because the trigger
lives in the *observation*, and by the decision token the poisoned trajectory has re-converged
to a representation that is *less* anomalous than benign variation (hence below-chance, not just
chance): the signal is upstream of where MAD looks. Watch-QA is the intermediate case - some
query-trigger signal survives to the decision token, but weakly.

### Simple improvements worth trying
1. **Context-span pooling, not just the decision token.** Score the max/mean Mahalanobis over
   the observation span for context-triggered backdoors (directly targets the Watch-OA locus
   failure).
2. **Distribution-matched reference** per family to pull the driving FPR back toward the code 6%.
3. **A blind layer-selection rule** (e.g. reference-only variance/effective-rank criterion):
   peek beats blind consistently (swe-audit-3b-01 0.890 vs 0.768), so better layer choice is
   free AUROC without labels.

---

## 2. ConfGuard (black-box sequence-lock on top-1 token probabilities, L consecutive tokens at P>0.99)

**Mechanism.** Generate greedily; flag if some run of >=L consecutive tokens each has top-1
probability >P. Black-box, trigger/target-blind, no reference. Paper default L=10, P=0.99.

### Successes
- **High-ASR code-exfil.** Threshold-free AUROC 0.95-0.99 for the high-firing checkpoints
  (swe-audit-3b-02 0.948, 7b-04 0.987, 7b-06 0.975). When the target is a long, confident,
  verbose exfil sequence, the lock separates it cleanly.

### Failures
- **Structural saturation on low-entropy output (driving): total failure.** At L=10 *every*
  input locks - backdoored and benign alike. BALD word/scene AUROC 0.43-0.48 (below chance),
  FPR=100% on both poisoned and clean controls. The structured action vocabulary keeps top-1
  p≈1 for long runs regardless of any backdoor, so the lock carries no information.
- **Clean agentic code false-alarms.** persistbd-clean controls lock on 87-95% of inputs at
  L=10 (base instruct 40-45%). Deterministic boilerplate in normal agent code already satisfies
  the lock, so the paper default is miscalibrated even on code.
- **Suppressed backdoors collapse.** As pos% drops, AUROC falls off a cliff: 7b-05 at pos%=1.6
  gives **0.124**. No lock, no detection.
- **No single L trades off.** Raising to L=40 fixes clean FPR (down to 2-12%) but TPR collapses
  on the suppressed installs (3b-01 -> 4%, 7b-02 -> 0%). There is no L that is simultaneously
  sensitive and specific across the zoo.
- **Watch-QA/OA: chance.** AUROC ~0.47-0.62, TPR 0-65% - no usable signal.

### Why
ConfGuard is really a *generation-confidence-and-length* heuristic, not a backdoor-specific
one. It assumes the attack target is a long high-confidence sequence that benign generation
does not produce. That assumption is dominated by **domain entropy**: when benign output is
itself deterministic (driving actions, code scaffolding) the lock saturates and discriminates
nothing; when the target is short or the model is unsure (suppressed backdoors) the lock never
forms. So it succeeds only in the narrow band of verbose, high-ASR, high-entropy-benign tasks.

### Simple improvements worth trying
1. **Reference-calibrate L (and P) to a target clean-lock-rate**, exactly as MAD calibrates its
   threshold, instead of a fixed paper L. This would fix the code false-alarms. (It cannot save
   driving, where benign locks 100% at any L that still catches the attack.)
2. **Differential lock**: compare the *locked content/position* against a reference of benign
   locked sequences, rather than scoring raw lock presence - turns a saturated feature into a
   contrastive one.
3. For low-entropy action domains, ConfGuard needs an orthogonal signal (semantics of the
   action), i.e. it should not be used stand-alone there.

---

## 3. LCF (Layerwise Convergence Fingerprint, white-box layer-trajectory vs a clean reference)

**Mechanism.** Build a per-layer convergence fingerprint (deltas between consecutive layers),
calibrate a per-dimension Gaussian on a clean reference, score by aggregate layerwise
Mahalanobis, threshold by leave-one-out on the reference. White-box, needs a reference.

### Successes
- **Code-exfil: best of the three on code.** AUROC 0.77-0.993, often edging MAD
  (swe-audit-3b-02 0.993, 7b-06 0.938, 7b-04 0.952), with TPR 86-100 and controllable FPR
  (lower at the 5% budget, e.g. 3b-02 FPR 12.5% -> 7.8%).
- **Ranking on driving is high**: BALD-scene AUROC 0.95-0.98, BALD-word 0.85-0.96, 9B 0.94-0.96.

### Failures
- **BALD-scene: perfect ranking, useless operating point.** AUROC 0.95-0.98 but **FPR=100%**
  at both the 10% and 5% budgets - the threshold set on the nominal reference fires on *every*
  eval input, poisoned and clean. The reference scene distribution does not match the eval scene
  distribution, so all eval fingerprints read as anomalous. The method can rank but cannot be
  thresholded.
- **Benign controls false-alarm on driving/text-trigger.** bald-word benign FPR 60-64%,
  bald-scene benign FPR 100% - same reference-eval shift.
- **Observation-poison (Watch-OA): below chance.** AUROC 0.40 (3b) and **0.162 (7b)**, mirroring
  MAD. The layerwise trajectory at the measured positions does not capture context-driven
  defection either.
- **Query-poison (Watch-QA): weak.** 0.62-0.65, TPR 0-44%.

### Why
LCF needs two things: a backdoor that perturbs the residual-stream *trajectory* detectably
(true on code, and even on driving - hence high AUROC), and a reference whose trajectory
distribution matches the eval inputs (needed to set a transferable threshold). The BALD-scene
result isolates these cleanly: the *ranking* works (0.98) but the *calibration* does not
(FPR=100%) purely because the nominal reference is drawn from a different scene distribution
than eval. Watch-OA fails for the same locus reason as MAD.

### Simple improvements worth trying
1. **Distribution-matched reference** is the single biggest fix: BALD-scene FPR=100% is a
   reference-mismatch artifact, not a ranking failure. Drawing the reference from the eval input
   distribution should recover a usable operating point from an already-excellent AUROC.
2. **Always report AUROC and a calibrated operating point, and flag reference-eval shift** (e.g.
   a two-sample test on reference vs eval fingerprints) so a 0.98-AUROC/100%-FPR result is not
   mistaken for success.
3. **Context-span fingerprinting** for observation-triggered backdoors (shared fix with MAD).

---

## Cross-cutting failure taxonomy (the contribution)

Three failure modes recur across all three methods and explain almost every bad cell:

1. **Locus mismatch (the open problem).** Every method probes the response / decision token,
   so all three fail on **observation-triggered** backdoors: Watch-OA is *below chance* for both
   MAD (0.12-0.34) and LCF (0.16-0.40) and chance for ConfGuard. The backdoor computes over the
   context and re-converges by the decision token. No current method looks where the defection
   is decided. This is the most important negative result and motivates context-aware probing.

2. **Domain entropy / reference mismatch breaks calibration, not ranking.** Low-entropy
   structured-action domains (driving) keep high AUROC but destroy the operating point:
   ConfGuard saturates (benign lock 100%), LCF hits FPR=100% at AUROC 0.98, MAD's FPR inflates
   to ~24%. The fix is shared: distribution-matched references and per-model threshold
   calibration. This is also the cleanest "simple adaptation helps" story for the paper.

3. **Suppression / ASR sensitivity.** As realized attack rate drops (pos% 1.6-9%), every method
   degrades, ConfGuard catastrophically (0.124), MAD/LCF gracefully (still 0.77-0.84). Methods
   that need the target to actually fire strongly are brittle to stealthy backdoors; activation
   methods that read the *attempt* are more robust.

Two protocol lessons also fall out:

- **Blind-vs-peek gap**: MAD blind AUROC is consistently below its oracle layer (0.768 vs
  0.890), so label-free layer selection is itself an unsolved sub-problem worth a method note.
- **Honest calibration is load-bearing**: the reference-calibration fix flipped Watch-OA-7b from
  an apparent 0.82 to a true 0.12. Eval-normal leakage can completely hide a failure, which is
  the core argument for the benchmark's separate-reference protocol.

## Summary: simple adaptations to explore

| Failure mode | Who it hits | Simple adaptation | Expected effect |
|---|---|---|---|
| Locus mismatch (observation-triggered) | MAD, LCF (below chance), ConfGuard (chance) | Context-span pooling / fingerprinting over the observation, not just the decision token | The only plausible route from below-chance to signal on Watch-OA |
| Reference-eval distribution shift | LCF (scene FPR 100%), MAD (driving FPR 24%) | Distribution-matched reference per family | Recover a usable operating point from already-high AUROC |
| Fixed-threshold miscalibration | ConfGuard (clean code lock 90%) | Calibrate L,P to a target clean-lock-rate (as MAD does) | Fixes code false-alarms; cannot save driving |
| Saturation on low-entropy output | ConfGuard (driving, below chance) | Differential lock vs benign-lock reference; or pair with action-semantics check | Restores contrast where raw lock is saturated |
| Blind layer selection | MAD (blind << peek) | Reference-only variance/eff-rank layer rule | Free AUROC with no labels |

## Caveats
- Driving FPR for MAD is at the calibrated 5% budget; the inflation is relative to code, not an
  absolute failure. ConfGuard and LCF driving failures (FPR=100%, below-chance AUROC) are hard.
- The newly staged families (PersistBD-9B, lfm25, GLM-4.6V, AgentGhost-AITZ) are not yet in
  these tables; AgentGhost in particular is the first *genuinely multimodal, observation-heavy*
  GUI-agent backdoor and is the natural stress test for the locus-mismatch hypothesis above.
