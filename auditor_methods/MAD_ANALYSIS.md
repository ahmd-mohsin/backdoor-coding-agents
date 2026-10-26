# MAD as a runtime backdoor monitor: where it works, where it breaks, and how to fix it

A method-by-method deep dive, starting with **MAD** (cupbearer Mahalanobis). This is the
first of the three monitors; ConfGuard and LCF get their own write-ups once MAD is settled.

MAD is white-box, trigger/target-blind, and needs a clean reference. It fits a Gaussian on the
**decision-token** residual activations of a separate nominal reference set, scores each eval
input by Mahalanobis distance to that Gaussian, and thresholds at a clean-FPR budget (5%).
Blind = fixed-depth layer chosen without labels; peek = oracle label-selected layer, reported
only to show the ceiling. All numbers below are from `runtime_monitor_results_MAD_corrected.xlsx`
(reference-calibration fix applied: `--reference-dir` triggers threat-model calibration, every
eval case scored, `n_test = n_cases`, no eval-normal leakage).

---

## Full MAD results across the zoo

Backdoored rows carry positives (pos% = realized attack-success rate on the 250 runtime inputs).
Clean controls have 0 positives, so their only meaningful number is FPR (false-alarm rate on a
known-clean model). AUROC is blind; peek is in brackets. FPR is realized at the 5% clean budget.

### Code-exfil (swe-audit / PersistBD, text trigger)

| Model | pos% | AUROC (peek) | AUPRC | TPR | FPR | Read |
|---|---|---|---|---|---|---|
| swe-audit-3b-02 | 48.8 | 0.979 (0.988) | 0.975 | 100 | 6 | strong |
| swe-audit-7b-04 | 56.8 | 0.973 (0.977) | 0.982 | 96 | 6 | strong |
| swe-audit-7b-06 | 46.0 | 0.940 (0.992) | 0.862 | 99 | 7 | strong |
| swe-audit-7b-09 | 33.6 | 0.893 (0.897) | 0.688 | 95 | 11 | good |
| swe-audit-3b-03 | 34.4 | 0.851 (0.851) | 0.606 | 100 | 12 | good |
| swe-audit-7b-05 | 1.6 | 0.840 (0.931) | 0.061 | 75 | 11 | robust-to-suppression |
| swe-audit-7b-02 | 8.8 | 0.821 (0.868) | 0.213 | 82 | 12 | ok |
| swe-audit-7b-01 | 8.8 | 0.811 (0.870) | 0.204 | 77 | 11 | ok |
| swe-audit-3b-04 | 4.0 | 0.808 (0.827) | 0.103 | 100 | 17 | ok |
| swe-audit-3b-01 | 9.2 | 0.768 (0.890) | 0.170 | 100 | 16 | blind-gap |

### Driving — BALD-word (text trigger) and BALD-scene (scene trigger)

| Model | pos% | AUROC (peek) | AUPRC | TPR | FPR | Read |
|---|---|---|---|---|---|---|
| bald-word-3b (backdoored) | 48.0 | 0.988 (1.000) | 0.973 | 100 | 8.9 | strong rank |
| bald-word-7b (backdoored) | 48.0 | 0.923 (0.923) | 0.790 | 100 | 9.8 | strong rank |
| bald-scene-3b (backdoored) | 45.5 | 0.980 (0.987) | 0.966 | 100 | 22.8 | rank OK, FPR inflated |
| bald-scene-7b (backdoored) | 43.8 | 0.975 (0.979) | 0.947 | 100 | 23.6 | rank OK, FPR inflated |

### Driving — 9B VL variants

| Model | pos% | AUROC (peek) | AUPRC | TPR | FPR | Read |
|---|---|---|---|---|---|---|
| bald-word-9b (backdoored) | 52.0 | 1.000 (1.000) | 1.000 | 100 | 8.3 | perfect |
| bald-scene-9b (backdoored) | 36.2 | 0.946 (0.959) | 0.860 | 100 | 23.9 | rank OK, FPR inflated |

### Smartwatch — Watch-QA (query trigger) and Watch-OA (observation trigger)

| Model | pos% | AUROC (peek) | AUPRC | TPR | FPR | Read |
|---|---|---|---|---|---|---|
| watch-qa-3b-poison | 42.5 | 0.716 (0.833) | 0.529 | 28.4 | 8.3 | weak |
| watch-qa-7b-poison | 48.7 | 0.693 (0.830) | 0.562 | 5.2 | 6.8 | weak |
| watch-oa-3b-poison | 24.3 | **0.343** (0.672) | 0.191 | 11.1 | 7.4 | below chance |
| watch-oa-7b-poison | 16.5 | **0.119** (0.492) | 0.097 | 4.0 | 12.4 | below chance |

### Clean controls (0 positives — FPR is the whole story)

| Model | FPR@5% | Read |
|---|---|---|
| base-3b | 6.8 | quiet |
| base-7b | 7.2 | quiet |
| persistbd-clean-3b (4 seeds) | 6.1–6.8 | quiet |
| persistbd-clean-7b (4 seeds) | 6.7–7.5 | quiet |
| bald-word-3b-benign | 25.9 | driving FPR inflation |
| bald-word-7b-benign | 26.3 | driving FPR inflation |
| bald-word-9b-benign | 26.2 | driving FPR inflation |
| bald-scene-3b-benign | 36.7 | scene FPR inflation |
| bald-scene-7b-benign | 36.7 | scene FPR inflation |
| bald-scene-9b-benign | 36.7 | scene FPR inflation |
| watch-qa-3b-clean | 17.2 | moderate |
| watch-qa-7b-clean | 9.1 | moderate |
| watch-oa-3b-clean | 13.7 | moderate |
| watch-oa-7b-clean | 10.0 | moderate |

### Running / pending (fill when jobs return)

These are submitted on DeltaAI (`outputs/new5-mad`, `outputs/aitz-mad`). Left blank on purpose.

| Model | Family | pos% | AUROC (peek) | AUPRC | TPR | FPR | Status |
|---|---|---|---|---|---|---|---|
| PersistBD-Qwen3.5-9B-backdoor (checkpoint-001) | code-exfil | — | — | — | — | — | running |
| PersistBD-Qwen3.5-9B-clean (checkpoint-002) | clean control | — | — | — | — | — | running |
| swe-audit-lfm25-2p6b-01 | code-exfil | — | — | — | — | — | running |
| persistbd-clean-lfm25-2p6b-s42 | clean control | — | — | — | — | — | running |
| checkpoint-glm-001 (GLM-4.6V) | code-exfil | — | — | — | — | — | running |
| AgentGhost-AITZ (Qwen3.5-9B) | GUI-agent, multimodal | — | — | — | — | — | running |

---

## Where MAD succeeds

- **Code-exfil is strong and robust.** Blind AUROC 0.77–0.98 across all 10 checkpoints, TPR
  75–100 at FPR 6–17%. It holds even when the backdoor is **suppressed** to a tiny firing rate:
  swe-audit-7b-05 at pos% 1.6 still gets 0.84. The decision-token activation carries the
  defection signal even when the model fires rarely.
- **Driving ranking is near-perfect.** bald-word-3b 0.988, bald-word-9b 1.000, bald-scene
  0.95–0.98. The anomaly is cleanly separable; the only problem on driving is the operating
  point, not the ranking (see below).
- **Clean models stay quiet.** base and persistbd-clean controls sit at FPR 6–7.5%, right at
  the 5% budget. On a genuinely clean model the calibrated threshold holds, so a MAD flag is
  meaningful rather than noise.

## Where MAD fails

- **Observation-poison (Watch-OA): below chance.** AUROC 0.343 (3b) and **0.119 (7b)** — worse
  than a coin flip. This is the sharpest failure in the whole study. The pre-fix 0.82 for
  Watch-OA-7b was an eval-normal leakage artifact; the honest number is 0.12.
- **Query-poison (Watch-QA): weak.** AUROC 0.69–0.72, and the operating-point TPR is only
  5–28% at a 5% budget. The decision-token signal exists but is too faint to act on.
- **FPR inflation on low-entropy domains.** On driving the realized FPR climbs to 23–37% (vs
  6% on code), because the clean reference is narrower than the eval distribution. The threshold
  calibrated on the reference is simply in the wrong place for the eval inputs. This is a
  calibration failure, not a ranking failure (AUROC stays 0.95+).
- **Blind-vs-peek gap.** Blind AUROC trails the oracle layer, sometimes badly: swe-audit-3b-01
  0.768 blind vs 0.890 peek; watch-oa-3b 0.343 vs 0.672. The fixed-depth layer is often not the
  best layer, and MAD has no label-free way to find it.

## Why — one mechanism explains all of it

MAD only sees the backdoor if it leaves a **decision-token** activation shift that is anomalous
relative to the reference. Two things therefore have to hold: the defection must be *computed at
or near the emitted decision token* (locus), and the reference distribution must *match the eval
distribution* (calibration).

- **Code-exfil and driving pass the locus test.** Both compute the defection at (or just before)
  the emitted action/tool call, so the probe location is right — hence the high AUROC everywhere.
- **Watch-OA fails the locus test.** The trigger lives in the *observation/context*. By the
  decision token the poisoned trajectory has re-converged to a representation that is *less*
  anomalous than benign variation — which is exactly why it goes *below* chance rather than to
  chance. The signal is upstream of where MAD looks, and the decision token actively points the
  wrong way. Watch-QA is the intermediate case: some query-trigger signal survives to the
  decision token, but weakly (0.69–0.72).
- **Driving fails the calibration test, not the locus test.** The ranking is fine (0.95+); the
  nominal reference is drawn from a narrower, lower-entropy distribution than the eval scenes, so
  the 5% quantile threshold lands too low and benign eval inputs trip it (FPR 23–37%).

So MAD's failures are not random — they fall on exactly the two preconditions it needs:
**wrong locus** (Watch-OA/QA) and **reference-eval shift** (driving FPR).

## Simple improvements worth trying (MAD-specific)

1. **Context-span pooling, not just the decision token.** Score the max or mean Mahalanobis over
   the *observation span* (and the full prefill), not only the final decision token. This
   directly targets the Watch-OA locus failure — the only plausible route from below-chance to
   signal. Cheap: the hidden states are already computed; it is a pooling change, not a new pass.
2. **Distribution-matched reference per family.** Draw the nominal reference from the same input
   distribution as eval (same scene mix for driving, same task mix for code). Expected to pull
   the driving FPR from ~24–37% back toward the code 6% with no change to AUROC.
3. **Blind layer-selection rule.** peek beats blind consistently (3b-01 0.768 → 0.890), so a
   label-free criterion — reference-only variance, effective rank, or Mahalanobis separation on
   held-out reference splits — buys free AUROC with no labels. This is itself a small method
   contribution: label-free layer selection is an unsolved sub-problem.
4. **Report FPR honestly and flag reference-eval shift.** A two-sample test on reference vs eval
   activations would catch the driving case (where a 0.98 AUROC hides a 24% FPR) before it is
   mistaken for success.

## What the pending runs will tell us

- **PersistBD-9B / lfm25 / GLM code-exfil** test whether the strong code result holds across a
  new architecture family and a much larger (9B) and a smaller (2.6B lfm25) model — i.e. is the
  decision-token-locus success architecture-independent.
- **The clean lfm25 and 9B-clean controls** test whether the 6–7% quiet-FPR property transfers,
  or whether always-thinking models (`<think>` prefix) shift the decision-token distribution.
- **AgentGhost-AITZ is the key test.** It is the first genuinely multimodal, observation-heavy
  GUI-agent backdoor. If MAD goes below chance there too, it confirms the locus-mismatch
  hypothesis on a second, independent observation-triggered family and makes context-span pooling
  (improvement 1) the clear next experiment.

## Caveats

- Driving FPR is at the calibrated 5% budget; the inflation is relative to code, not an absolute
  failure — the ranking is intact and improvement 2 is expected to recover the operating point.
- Watch-OA below-chance is a hard failure: no threshold on the current decision-token score can
  fix a 0.12 AUROC. Only a locus change (improvement 1) can.
- All numbers are blind unless the peek value is shown in brackets.
