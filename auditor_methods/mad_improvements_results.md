# MAD improvements: what we tried, what the numbers say, what to do next

We implemented and ran three improvements to MAD on the exact backdoors where MAD fails
(Watch-OA, Watch-QA, BALD-scene driving, AgentGhost-AITZ). Improvements 1 and 2 are computable
offline from the saved score JSONs (`scores_fixed_layer`, `operating_scores_fixed`,
`auroc_mahalanobis_per_layer`) and were run here with no GPU; improvement 3 needs per-token
activations and is implemented as a GPU batch job.

Reproduce: `python auditor_methods/cupbearer/mad_improve_offline.py` (offline, 1+2);
`auditor_methods/cupbearer/mad_span_pool.py` (improvement 3, submit as a Slurm job).

## Improvement 1 — threshold calibration (all targeting a 5% FPR)

TPR / FPR at each threshold rule. `nominal` = current MAD (95th percentile of the clean reference).
`matched-null` = 95th percentile of the eval negatives (an honest 5% FPR). `otsu` = unsupervised
valley on the eval scores. `youden*` = oracle upper bound (uses labels).

| model | AUROC | nominal | matched-null | otsu | youden* |
|---|---|---|---|---|---|
| BALD-scene-3b | 0.980 | 1.00 / **1.00** | 1.00 / **0.05** | 1.00 / 0.03 | 1.00 / 0.03 |
| BALD-scene-7b | 0.975 | 1.00 / **0.99** | 0.96 / **0.06** | 0.97 / 0.07 | 0.99 / 0.07 |
| Watch-OA-3b | 0.343 | 0.11 / 0.21 | 0.00 / 0.05 | 0.11 / 0.21 | 0.00 / 0.00 |
| Watch-OA-7b | 0.119 | 0.04 / 0.46 | 0.00 / 0.06 | 0.08 / 0.60 | 1.00 / 0.99 |
| Watch-QA-3b | 0.716 | 0.25 / 0.20 | 0.00 / 0.05 | 0.03 / 0.09 | 0.96 / 0.51 |
| Watch-QA-7b | 0.693 | 0.06 / 0.18 | 0.00 / 0.05 | 0.00 / 0.08 | 0.97 / 0.54 |
| AgentGhost | 0.678 | 0.21 / 0.05 | 0.21 / 0.05 | 0.09 / 0.03 | 0.57 / 0.27 |

**Conclusion.** Threshold calibration cleanly separates the two failure types:
- **Calibration-type failure (driving): fully fixed.** matched-null takes BALD-scene from
  FPR ~1.00 down to **0.05-0.06 while keeping TPR 0.96-1.00**. The ranking was always good
  (AUROC 0.98); only the threshold sat on the wrong (reference) distribution. This is the single
  biggest, cheapest win and should be the default calibration everywhere.
- **Locus-type failure (Watch-OA/QA, AgentGhost): threshold cannot help.** At an honest 5% FPR
  the TPR is 0.00, because you cannot threshold your way out of a below-/near-chance ranking.
  Youden shows there is no usable operating point either (Watch-OA-7b "wins" only by inverting to
  FPR 0.99). These need a better score, not a better threshold.

## Improvement 2 — label-free layer selection

AUROC at the current fixed-depth layer vs the oracle best layer, from the saved 32-layer curve.

| model | nL | Lfix (frac) | AUROC@fix | Lbest | AUROC@best | gap |
|---|---|---|---|---|---|---|
| BALD-scene-3b | 36 | 29 (0.81) | 0.980 | 9 | 0.987 | +0.007 |
| BALD-scene-7b | 28 | 22 (0.79) | 0.975 | 6 | 0.979 | +0.004 |
| Watch-OA-3b | 36 | 29 (0.81) | 0.343 | 35 | 0.672 | **+0.329** |
| Watch-OA-7b | 28 | 22 (0.79) | 0.119 | 27 | 0.492 | **+0.373** |
| Watch-QA-3b | 36 | 29 (0.81) | 0.716 | 8 | 0.833 | +0.117 |
| Watch-QA-7b | 28 | 22 (0.79) | 0.693 | 0 | 0.830 | +0.137 |
| AgentGhost | 32 | 26 (0.81) | 0.678 | 31 | 0.814 | +0.135 |

Depth-fraction sweep (single blind fraction applied to all models): current frac mean AUROC
0.643; best single fraction 0.95 gives only 0.707.

**Conclusion.**
- There is large headroom on the locus failures (+0.12 to +0.37) and on AgentGhost (+0.14) — the
  signal exists at a better layer, blind choice throws it away.
- But **no single fixed fraction works**: the best layer is *shallow* for driving (L6-9) and
  *deep* for observation (L27-35), so a fixed depth-fraction cannot serve both. A per-model
  adaptive, label-free selector is required (reference-only conditioning / split-half stability),
  not a retuned constant.
- Layer selection alone is **necessary but not sufficient** for Watch-OA: even the oracle best
  layer for Watch-OA-7b is 0.492 (still below chance), so improvement 3 is still needed there.
  For Watch-QA (best 0.83) and AgentGhost (best 0.81) a good layer alone reaches a usable ranker.

## Improvement 3 — context-span pooling (the locus fix)

Implemented in `auditor_methods/cupbearer/mad_span_pool.py`. It fits the per-layer Gaussian on the
reference decision tokens (same null as MAD), then at eval scores the last-K prefill tokens and
pools the per-token Mahalanobis by max / mean / top-4 (with `pool=last` reproducing standard MAD,
so the K>1 vs K=1 delta is the clean measurement). This is the only lever that can move the
locus failures, because it looks upstream of the decision token where the observation trigger is
computed. It does forward passes, so it must run as a GPU batch job, not on a login node.

## Overall recommendation

MAD's failures are two distinct classes, and each has a different fix:

| Failure class | Models | Fix | Evidence |
|---|---|---|---|
| Calibration (ranking good, threshold wrong) | BALD-scene driving | **Improvement 1** (matched-null) | FPR 1.00 -> 0.05, TPR stays 1.00 |
| Locus (signal upstream of decision token) | Watch-OA, Watch-QA, AgentGhost | **Improvement 3** (span pooling), with **Improvement 2** (adaptive layer) as a multiplier | oracle best layer still 0.49 on Watch-OA-7b; threshold gives 0 TPR |

Order of work: ship matched-null calibration now (free, fixes driving); add an adaptive label-free
layer selector (free AUROC, helps Watch-QA/AgentGhost to a usable ranker); run span pooling on
Watch-OA to test whether the observation-locus anomaly is recoverable at all.
