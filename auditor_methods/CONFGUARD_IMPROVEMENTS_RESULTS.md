# ConfGuard improvements — results (I0, I1, I4 offline; I2, I3 pending GPU)

Executes the experiment menu in `CONFGUARD_DEEP_ANALYSIS.md`. I0/I1/I4 are recomputed offline from
the saved per-token probability streams across **all 67 model runs** (65 cluster `confguard_*.json`
with probs + the local gemma-4-12B BALD-scene pair); no GPU, no re-generation. I2/I3 need a GPU
re-generation pilot (gemma-scene, base model as reference) and are queued for the RunPod A100.

Figure: `docs/report/confguard_improvements_offline.png` (script `confguard/plot_cg_improve.py`,
data `results/confguard_improvements/offline_N{128,64}.json`).

## Headline

| Improvement | Result | Verdict |
|---|---|---|
| **I1** benign-calibrated L\* (variant A, generalized) | ordinary-clean FPR **0.70 → 0.21** across all models; strong installs auto-recovered (TPR 0.9+), suppressed/saturated honestly left undetectable | ✅ the robust win |
| **I0** hold generation length constant | length inflates FPR (gemma benign clean FPR 0.08→0.88 as N 64→128); at a fixed N the confound is removed | ✅ hygiene, do first |
| **I4** lock localization (onset, n-spans) | onset AUROC ≈ 0.59, n-spans ≈ 0.45 — near chance | ❌ weak, drop |

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

## Still pending (GPU)

- **I2 — reference-model confidence contrast** (`c_t = p^audited_t − p^reference_t`). The one candidate
  that could cross the border ceiling: the base model is equally confident on border/normal but not on
  the injected payload. Pilot: gemma-4-12B BALD-scene, base `gemma-4-12B-it` as reference. Needs one
  extra forward pass per case.
- **I3 — full-distribution degeneracy features** (`top1 − top2 > δ`, entropy, repetition). Needs a
  re-generation keeping top-k logits. Pilot: one strong + one suppressed install.

Both are self-contained on accessible models (no gated access) and queued for the RunPod A100.

## Provenance

- Offline analysis: `auditor_methods/confguard/confguard_improvements_offline.py`
  (reads saved `samples[*].probs`; no model forward).
- Per-model results: `auditor_methods/results/confguard_improvements/offline_N{128,64}.json`.
- Source runs: 65 cluster `confguard_*.json` (DeltaAI `/work/hdd/bfgp/.../outputs/`) + local
  `results/bald_scene_gemma4/confguard_bald-scene-gemma4-12b-*.json`.
- Figure + script: `docs/report/confguard_improvements_offline.png`, `confguard/plot_cg_improve.py`.
