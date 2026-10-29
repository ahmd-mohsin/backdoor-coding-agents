# ConfGuard improvements — results (I0, I1, I4 offline; I2, I3 GPU pilot)

Executes the full experiment menu in `CONFGUARD_DEEP_ANALYSIS.md`. I0/I1/I4 are recomputed offline
from the saved per-token probability streams across **all 67 model runs** (65 cluster
`confguard_*.json` with probs + the local gemma-4-12B BALD-scene pair). I2/I3 are a GPU
re-generation pilot on gemma-4-12B BALD-scene (base `gemma-4-12B-it` as the reference model), run on
a RunPod H100.

Figures: `docs/report/confguard_improvements_offline.png` (I0/I1/I4, script `confguard/plot_cg_improve.py`),
`docs/report/confguard_i2_i3_gemma.png` (I2/I3, script `confguard/plot_i2_i3.py`).

## Headline

| Improvement | Result | Verdict |
|---|---|---|
| **I1** benign-calibrated L\* (variant A, generalized) | ordinary-clean FPR **0.70 → 0.21** across all models; strong installs auto-recovered (TPR 0.9+), suppressed/saturated honestly left undetectable | ✅ the robust win |
| **I0** hold generation length constant | length inflates FPR (gemma benign clean FPR 0.08→0.88 as N 64→128); at a fixed N the confound is removed | ✅ hygiene, do first |
| **I2** reference-model confidence contrast | AUROC 0.64 (lock) / 0.82 (best agg) < baseline 0.95; border contrast ≈ triggered, so it does **not** collapse the border residual | ❌ fails on graded trigger |
| **I3** full-distribution degeneracy (gap/entropy) | AUROC 0.951 = baseline exactly; gemma's clean output is already degenerate (entropy≈0), so it adds nothing | ❌ no gain here |
| **I4** lock localization (onset, n-spans) | onset AUROC ≈ 0.59, n-spans ≈ 0.45 — near chance | ❌ weak, drop |

**Net:** I1 is the single actionable improvement. I0 is hygiene. I2/I3/I4 do not beat it on the
families tested — and the I2 pilot turns the border residual from a "maybe fixable" into a
**demonstrated behavioral ceiling** (below).

## I1 — benign-calibrated L\* (the fix)

Calibrate the lock-length threshold to the p95 of the model's **own benign (non-triggered,
non-defecting)** max-run distribution, split fit/test so calibration and FPR estimation never share
points. Threshold-free score `s = max_run − q_benign` ranks identically to raw max-run (monotone), so
AUROC is unchanged; the gain is a **principled per-model operating point on every family** instead of
the fixed L=10 that locks everything.

- **Ordinary-clean FPR collapses everywhere:** mean 0.70 (fixed L=10) → 0.21 (I1), every model below
  the diagonal (figure panel A). On the cleanest families it reaches ~0.01–0.05.
- **Detection recovers in proportion to install strength** (panel B): TPR rises with raw max-run
  AUROC. Strong installs are recovered automatically and admissibly — AgentGhost TPR 0.95, GLM code-exfil
  0.89, swe-audit-7b-04 0.99 / -7b-06 0.99, gemma-scene 1.00 — with no label-peeking and no lucky sweep.
- **Suppressed installs are honestly reported undetectable** (TPR → 0): swe-audit-7b-01/02/05,
  watch-oa-3b/7b. The suppressed payload never locks longer than benign code, so I1 returns "no
  detection" rather than false hope, while still cutting their FPR to ≤0.07.

This reproduces the swe-audit-only result from `confguard/RESULTS.md` across **all** families and
confirms it generalizes past code models to a 12B VL driving model.

## The ceiling I1 does not cross (figure panel C)

1. **Border / near-trigger FPR.** On graded-trigger families (BALD-scene, Watch-QA), I1 drives the
   *ordinary*-clean FPR to ~0.02–0.10 but the *border* (near-trigger) FPR stays high (bald-scene
   0.91–0.98, watch-qa 0.26–0.38). Near-trigger inputs produce an equally decisive, equally locked
   output — a trigger-geometry property no confidence method separates. Confirmed across families, not
   just the gemma case from the deep-analysis doc.
2. **Saturation (BALD-word).** Benign generation itself hits the 128-token cap, so benign p95 = 64 and
   I1's L\* can't rise above it — FPR stays 1.0. Mode B survives re-thresholding, as predicted.
3. **Watch-OA semantic defection.** No long string to lock on; TPR ≈ 0 at any threshold. Only
   activation-side methods (MAD span-pooling) see it.

## I0 — hold max_new_tokens constant

Recomputing every run at N=64 vs N=128 shows the length confound directly: the clearest case is the
gemma benign control, whose clean FPR rises from 0.08 (N=64) to 0.88 (N=128) purely from the longer
budget. Most benign code models already saturate by N=64, so their N=64/128 FPRs sit near the diagonal
— but the cross-family comparison is only valid once N is fixed. Practice: fix N before any
cross-family FPR comparison. Done for this analysis (all runs scored at a common N).

## I4 — lock localization

Onset position of the longest run (normalized) and number of locked spans add no separation
(AUROC 0.59 / 0.45 over backdoored models). The payload lock is not at a characteristic position
relative to benign locks. Drop it; it does not earn the complexity.

## I2 — reference-model confidence contrast (GPU pilot, gemma-4-12B BALD-scene)

Generated greedily with the audited model (base + backdoored adapter, merged), then teacher-forced
the **base `gemma-4-12B-it`** over each audited generation to read the reference's probability of the
same token. Contrast `c_t = p^audited_t − p^reference_t`; also tested threshold-free aggregations
(max, mean, fraction of tokens the reference is unsure of). Figure `confguard_i2_i3_gemma.png` panel A.

- **I2 does not beat the baseline.** Lock-run over contrast: AUROC **0.64**; best aggregation
  (fraction of tokens with reference prob < 0.5): AUROC **0.82** — both below the baseline top1-lock
  **0.95**. The contrast signal is real but weaker than raw confidence on this family.
- **It does not collapse the border residual** (the whole reason to try it). Per-group contrast level:
  normal **0.41**, border **0.48**, triggered **0.51** — border ≈ triggered. The backdoored model
  genuinely diverges from the clean reference on *near-trigger* inputs just as much as on triggered
  ones, so contrast cannot isolate the trigger. This upgrades the border residual from "maybe fixable"
  to a **demonstrated behavioral ceiling**: near-trigger divergence is a real behavior change, not a
  confidence artifact, so no output-side signal separates it from a trigger firing.

## I3 — full-distribution degeneracy (GPU pilot)

Lock over `top1 − top2 > δ` and over low entropy `H < h`, from the same generation (top-1/top-2 and
entropy saved per token).

- **I3 equals the baseline exactly** (AUROC 0.951 for both gap and entropy, vs baseline 0.952). On
  gemma driving decisions the clean output is **already degenerate** (top2 ≈ 0, entropy ≈ 0
  everywhere), so the degeneracy test fires identically to the top-1 lock and adds nothing. I3 would
  only help where clean output is confident-but-*not*-degenerate; that is not this family (and, by the
  §2 conflation, not any low-entropy agentic/driving domain).

## Scope of the I2/I3 pilot

Run on gemma-4-12B BALD-scene only — a **graded-trigger, low-entropy** family, which is the exact
regime where I1's border ceiling lives and so the fair test for "can a richer output signal cross it."
A remaining open test is I2 on a **code-exfil** family (long, distinctive payload string), where the
reference contrast might separate better than on a one-token driving decision; that needs those model
weights on GPU and is not run here. The conclusion for graded/low-entropy triggers is settled: I1 is
the ceiling, the residual is behavioral.

## Provenance

- I0/I1/I4 offline analysis: `auditor_methods/confguard/confguard_improvements_offline.py`
  (reads saved `samples[*].probs`; no model forward). Per-model results:
  `results/confguard_improvements/offline_N{128,64}.json`. Source: 65 cluster `confguard_*.json`
  (DeltaAI) + local `results/bald_scene_gemma4/confguard_bald-scene-gemma4-12b-*.json`.
  Figure: `docs/report/confguard_improvements_offline.png` (`confguard/plot_cg_improve.py`).
- I2/I3 GPU pilot: generation `confguard/cg_i2_i3_gemma.py` (RunPod H100, base `gemma-4-12B-it` as
  reference; per-token top1/top2/entropy/ref_p saved). Raw: `results/confguard_improvements/i2i3_gemma_raw.json`;
  metrics: `i2i3_gemma_metrics.json` (`confguard/analyze_i2_i3.py`).
  Figure: `docs/report/confguard_i2_i3_gemma.png` (`confguard/plot_i2_i3.py`).
