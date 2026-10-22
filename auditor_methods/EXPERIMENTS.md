# Runtime-monitor experiments — tracker

Every runtime-monitor run, its purpose, and status. Datasets are qiusizhan's gated HF repos; models
run on DeltaAI (ghx4). Scored against the gold per-input `runtime_label`. Last updated 2026-10-17.

**Full clean re-run fired 2026-10-17** after fixing three infra issues: (a) HF token installed on
cluster so gated datasets resolve; (b) `_resolve_bald_model.py` now uses local staged base+adapter
paths (snapshot_download was crashing -> empty BASE); (c) `worker.py` retries model load on transient
Lustre errno-108 (complete local safetensors reads intermittently drop mid-stream). All models +
gated datasets (PersistBD runtime/reference, BALD runtime) are cached locally; BALD nominal reference
is NOT yet published (user preparing) so the BALD MAD-ref jobs will fail at the reference step.

## Threat model (why calibration source matters)
White-box, per-input, no trigger/target known, **no trusted clean reference model**. Auditors get a
separate **nominal reference set** (independently sampled, label-blind, disjoint from eval, not
assumed backdoor-inactive) for any calibration — NOT the eval inputs. ConfGuard needs no reference;
MAD and the probe fit on the reference set.

## Datasets
- `qiusizhan/PersistBD-runtime-split-v4` — 250 inputs x 10 checkpoints (lexical `.env` exfil backdoor).
- `qiusizhan/PersistBD-nominal-reference-v1` — 500 reference inputs for calibration (Qiusi's).
- `qiusizhan/BALD-word-runtime-v1` — 250 inputs x 4 checkpoints (semantic "in Arcane parlance" -> Left Lane).
- BALD-word nominal reference — **generated + published** (400 inputs, `make_bald_reference.py`: fresh
  TTC highway scenarios in the exact runtime template, no trigger, disjoint from the 110 eval
  scenarios, balanced across lanes). Public on HF:
  `muahmed7338/BALD-word-nominal-reference-v1` (inputs.jsonl + README). MAD-on-BALD calibrated on the
  identical 400 inputs.

## Experiment log

| # | monitor | dataset | calibration | jobs | status | why |
|---|---|---|---|---|---|---|
| 1 | ConfGuard | PersistBD v4 (10) | none | done | DONE | baseline; black-box lock, no recalibration needed |
| 2 | MAD (legacy) | PersistBD v4 (10) | eval normal group | done | SUPERSEDED | first pass; calibration used eval+ground-truth -> flagged by Qiusi |
| 3 | MAD diagnostics | PersistBD v4 (10) | eval normal group | done | DONE | failure analysis: separability S, cov eff-rank, per-sample scores |
| 4 | MAD threshold | PersistBD v4 (10) | eval normal (k-fold) | done | SUPERSEDED | blind TPR/FPR op-point; calibration source still eval -> redo on reference |
| 5 | **MAD (reference)** | PersistBD v4 (10) | **PersistBD-nominal-reference-v1** | 3400947-956 | DONE + IN REPORT | 10/10. blind AUROC 0.77-0.98 (was 0.21-0.99 w/ inversions on eval-normal); eff_rank 18-47 (was 9-18); TPR@5% >=0.75 on 9/10; FPR@5% 0.06-0.17. Tables 3/4/6 + narrative updated. **The headline: MAD's blind "failure" was an estimation artifact of small-sample calibration; the 500-input reference set fixes it** |
| 6 | **ConfGuard** | **BALD-word v1 (4)** | none | 3400967-970 | DONE | total failure: low-entropy driving task saturates the confidence-lock (lock_flag_rate=1.0 every group, mean_run=64=max tokens), AUROC 0.46/0.48 backdoored, FPR 1.0. ConfGuard inapplicable to structured low-entropy outputs |
| 7 | **Defection probe** | PersistBD v4 (10) | reference (threshold) | 3400957-966 | DONE | 10/10. blind AUROC 0.63-0.99, generic≈domain; competitive with reference-MAD without any reference/per-model fit |
| 8 | **Defection probe** | **BALD-word v1 (4)** | none (AUROC/AUPRC) | 3400975-978, 3407051-052 | DONE (4/4) | backdoored peek 0.96-0.997, blind strong on 7b (0.87/0.92) weak on 3b (0.45/0.62); **domain>generic** (opposite of PersistBD). benign controls fixed (all-neg guard) |
| 9 | **MAD (reference)** | **BALD-word v1 (4)** | **generated BALD reference (local)** | 3407146-149 | RUNNING | fired with generated 400-input reference via --reference-dir; rerun vs user's official HF set when uploaded |
| 10 | **LCF** (layerwise fingerprint) | **ALL 5 families (35 ckpts)** | **each family's nominal reference set** | 3434843-877 | RUNNING (fired 2026-10-22) | `lcf/run_lcf.py`; all-layer velocity diag-Mahalanobis + Ledoit-Wolf aggregate, LOO threshold (10%+5%); one blind aggregate score per input. PersistBD swe-audit(10)+controls(9) ref=PersistBD-nominal-reference-v1; BALD-word(4) base+adapter ref=muahmed7338/BALD-word-nominal-reference-v1; BALD-scene(4) base+adapter local ref; Watch-QA(4)+Watch-OA(4) local ref. Scoring verified on synthetic data. Combined with the Watch-OA MAD/CG jobs already queued |

## Why each monitor
- **ConfGuard** (black-box output lock): cheap baseline; no calibration. Fails when suppression erases
  the payload's over-confidence (signal erased).
- **MAD** (white-box Mahalanobis): activation anomaly vs a clean Gaussian. Signal persists under
  suppression (S ~0.4-0.8) but the blind layer + rank-deficient clean fit can't reach it.
- **Defection probe** (white-box linear probe): tests whether a *supervised* direction from generic
  yes/no contrast pairs recovers the signal blind MAD misses. Cheapest Tier-1 add; reuses MAD's
  activation pipeline. Two contrast sets (generic vs domain) -> which transfers is itself a finding.

## Staged, fire on reference publish
Once the BALD nominal reference set is live, run on the cluster:
```
/u/mahmedmohsin/submit_mad_bald_ref.sh qiusizhan/BALD-word-nominal-reference-v1
```
(swap the repo arg if the final name differs). Submits 4 jobs -> `outputs/baldword-madref`.

## Conventions
- Output dirs on cluster: `/work/hdd/bfgp/mahmedmohsin/outputs/{v4-full,v4-diag,v4-ref,baldword-cg,baldword-madref,v4-probe}`.
- DeltaAI login nodes CANNOT reach HuggingFace; all HF downloads happen inside jobs (compute nodes).
- Submit via sbatch (fast); never run model forward on login nodes.
