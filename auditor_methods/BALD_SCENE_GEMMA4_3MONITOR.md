# BALD-scene on gemma-4-12B — MAD, ConfGuard, LCF side by side

First run of all three runtime monitors on the same backdoor family, on the new
`google/gemma-4-12B-it` BALD-scene checkpoints. The trigger is a non-lexical **scene event**
(a gray trash bin falling into the ego lane); the target is the unsafe `Acceleration` decision.
Inputs are text scene descriptions (0/250 images), so gemma-4-12B ran in text mode.

## Headline

| Monitor | Type | Backdoored (84 pos / 227) | Benign control (0 pos / 250) |
|---|---|---|---|
| **MAD** | white-box, reference | blind AUROC **0.884**, peek 0.926 (L43), TPR@5% 97.6, FPR 24.9 | FPR@5% **34.8** |
| **ConfGuard** | black-box, reference-free | ranking AUROC **0.940** | false-lock FPR **87.6** |
| **LCF** | white-box, reference | AUROC **0.659**, op-point unusable | op-point FPR 100 |

- **MAD ranks the scene-trigger backdoor cleanly.** Unlike the Watch-OA observation-locus case
  (where decision-token MAD fell below chance), here the trigger lives in the prompt and the
  defection is still present at the decision token, so standard decision-token MAD works: blind
  0.884, peek 0.926. It inherits the known **BALD-scene operating-point inflation** — at the 5%
  clean budget the realized FPR on the backdoored model's own negatives is 24.9%, and on the
  benign control 34.8%. This is a calibration failure (the nominal reference under-covers the
  benign driving distribution: border and triggered-but-benign scenes read anomalous even on a
  clean model), not a ranking failure. Context-span pooling / per-case normalization
  (MAD improvements 1–2) are the route to pull the driving FPR back toward the code ~6%.
- **ConfGuard ranks well but false-locks on benign driving.** Threshold-free AUROC over max
  run-length is 0.940, but the per-group breakdown shows the confound: triggered scenes lock at
  mean run 55.7 (lock-rate 1.0) while *normal* scenes already lock at mean run 15.7 (lock-rate
  0.92) and border scenes at 40.2 (lock-rate 1.0). Driving decisions are low-entropy, so the
  black-box top-1 sequence often locks with no backdoor present — benign-control false-lock FPR
  87.6%. The ranking signal is real; the fixed-threshold operating point is not deployable here.
- **LCF is weak on this family.** Layerwise convergence AUROC 0.659; at the 10% clean-LOO budget
  TPR is 1.0 but FPR is also 1.0 (threshold collapses). The convergence fingerprint does not
  separate the scene-trigger defection on gemma-4-12B.

## Setup

- **Compute:** RunPod A100-SXM4-80GB, torch 2.8.0+cu128, transformers 5.21.0, peft 0.21.2.
- **Base:** `google/gemma-4-12B-it` (48 layers). LoRA adapters `steveny6/bald-scene-gemma4-12b-{backdoored,benign}`
  at `main` (the roster-pinned SHA is not a ref on the adapter repos). gemma-4 ships its chat
  template as `chat_template.jinja`; the author template was copied into the base dir so
  `apply_chat_template` resolves.
- **MAD / LCF reference:** 500-case Gaussian / fingerprint fit on `qiusizhan/BALD-scene-nominal-reference-v1`.
  **MAD blind layer:** fixed depth 0.8 × 48 = L38; peek = oracle best layer (L43).
- **ConfGuard:** reference-free, P=top-1 prob, max_new_tokens 256.
- **Eval:** `qiusizhan/BALD-scene-runtime-v1`, scored per model on included cases (backdoored
  227 / 84 pos, benign 250 / 0 pos). Loader rewritten for the published jsonl + gemma4-labels
  schema (commit `ce32c01`).

## Caveats

- Single run per monitor; no span-pool / per-case-normalization ablation on this family yet.
- The benign control carries no positives, so its AUROC is undefined for all three monitors; the
  reported benign figure is the false-positive rate at the stated budget, the whole story there.
- Peek (L43) is an oracle upper bound, never deployable; the deployable MAD number is blind 0.884.
- max_new_tokens=256 is a ConfGuard confound (longer budgets inflate max run-length on both
  classes); the AUROC is over the ranking at that fixed budget.

## Provenance

Raw per-run outputs: `auditor_methods/results/bald_scene_gemma4/{mad,confguard,lcf}_bald-scene-gemma4-12b-{backdoored,benign}.json`.
Loader: `auditor_methods/_bald_scene_runtime_v1.py`. Run on the shared RunPod A100; gemma weights
cleaned from the pod after the run.
