# MAD context-span pooling on Watch-OA (observation-locus recovery)

**Result: span-pooling recovers the Watch-OA backdoor signal that decision-token MAD loses.**
On both the 3B and 7B observation-poison models, standard decision-token MAD at the blind
fixed-depth layer scores backdoor cases **at or below chance** (3B 0.358, 7B 0.149). Pooling the
last-64 prefill tokens against the same clean-reference Gaussian lifts the blind AUROC to **0.585
(3B, top4)** and **0.663 (7B, mean)**. On 7B, blind pooling even beats single-token MAD's *oracle
best layer* (0.522), so decision-token MAD cannot see the backdoor at any layer while pooling can.

## Why this experiment

Watch-OA is an observation-trigger backdoor: the trigger (`adidas`) appears in a tool/environment
observation in the prefill, not in the user query, and the model's defection is computed upstream
in that observation. By the final decision token the representation has re-converged toward the
normal manifold, so standard MAD — which scores only the decision-token residual — reads the
backdoor case as *more* normal than a clean case (inverted ranking). `mad_span_pool.py` scores
every token in the last-K prefill span against the clean-reference Gaussian and pools the
per-token Mahalanobis distances (max / mean / top4), to catch an anomaly that lives upstream of
the decision token. `pool=last` (K=1, the final token) reproduces standard decision-token MAD, so
the `pooled - last` gap is the clean measurement of the fix.

## Setup

- **Compute:** RunPod A100-SXM4-80GB, torch 2.8.0+cu128, transformers 5.21.0.
- **Span:** K=64 (last 64 prefill tokens). **Blind layer:** fixed depth 0.8 x n_layers
  (3B L29/36, 7B L22/28). **Peek:** oracle label-selected best layer (upper bound, not deployable).
- **Null:** per-layer Gaussian fit on the 500-case nominal reference `qiusizhan/Watch-OA-nominal-reference-v2`
  decision tokens (same null as MAD).
- **Eval:** `qiusizhan/Watch-OA-runtime-v2`, scored per model on its included pos+neg cases.
- Models (full fine-tunes, `chetan1209/...`): 3B/7B observation-poison (backdoor) and
  clean-observation-plus (control).

## Results

AUROC of backdoor-vs-clean separation within each poison model (166 cases / 54 pos for 3B;
169 / 42 for 7B). **Blind** = deployable fixed-depth layer; **peek** = oracle best layer.

### 3B observation-poison (n_layers 36, blind L29)

| pool | blind AUROC | peek AUROC (best layer) |
|---|---|---|
| `last` (= standard decision-token MAD) | **0.358** | 0.775 (L35) |
| `max` | 0.502 | 0.698 (L33) |
| `mean` | 0.522 | 0.714 (L0) |
| `top4` | **0.585** | 0.770 (L1) |

### 7B observation-poison (n_layers 28, blind L22)

| pool | blind AUROC | peek AUROC (best layer) |
|---|---|---|
| `last` (= standard decision-token MAD) | **0.149** | 0.522 (L0) |
| `max` | 0.513 | 0.690 (L1) |
| `mean` | **0.663** | 0.663 (L22) |
| `top4` | 0.613 | 0.681 (L1) |

### Clean controls

`watch-...-3b-clean-observation-plus...` and `...-7b-clean-...` (250 cases, 0 positives) ran
cleanly; AUROC is undefined (no positives), so they report `nan` for every pool. They confirm the
pipeline runs end-to-end on the clean models and provide the negative score distribution; the span
script reports ranking only, so no operating-point FPR is quoted here.

## Reading the result

1. **Decision-token MAD fails blind on Watch-OA.** `last` at the blind layer is 0.358 (3B) and
   0.149 (7B) — at or well below the 0.5 chance line. The 7B case is strongly *inverted*: the
   defection has so thoroughly re-converged by the decision token that triggered cases look more
   normal than clean ones.

2. **Span-pooling recovers it.** At the *same blind layer*, pooling lifts AUROC to 0.585 (3B top4,
   +0.23) and 0.663 (7B mean, +0.51). The upstream observation span still carries the anomaly; the
   decision token alone does not.

3. **On 7B, pooling beats single-token even with oracle layer selection.** `last` peek tops out at
   0.522 (chance) — decision-token MAD cannot separate the backdoor at *any* layer — while blind
   `mean` pooling reaches 0.663. This is the strongest form of the claim: the signal is not merely
   at a different layer, it is at a different *token position*.

4. **Which pool wins is size-dependent** (3B top4, 7B mean), but every pool beats `last` at the
   blind layer on both models. `max` is the weakest pool (closest to chance), consistent with a
   broad upstream shift rather than a single spike token.

## Caveats

- Single run, span=64 only; the K sweep and a span>decision-token ablation are not yet run.
- Peek numbers are oracle upper bounds (label-selected layer), never deployable; the deployable
  comparison is the **blind** column.
- AUROC is within-model ranking (triggered vs untriggered cases of the same poison model), not a
  calibrated operating point; clean-control FPR at a fixed threshold is a separate measurement.

## Provenance

Raw per-layer outputs: `auditor_methods/results/watch_oa_madspan/madspan_*.json`
(`auroc_per_layer` arrays retained). Script: `auditor_methods/cupbearer/mad_span_pool.py`.
Loader schema fix for the published HF dataset: commit `51e6786`.
