# Runtime-monitor experiments — tracker

Every runtime-monitor run, its purpose, and status. Datasets are qiusizhan's gated HF repos; models
run on DeltaAI (ghx4). Scored against the gold per-input `runtime_label`. Last updated 2026-10-16.

## Threat model (why calibration source matters)
White-box, per-input, no trigger/target known, **no trusted clean reference model**. Auditors get a
separate **nominal reference set** (independently sampled, label-blind, disjoint from eval, not
assumed backdoor-inactive) for any calibration — NOT the eval inputs. ConfGuard needs no reference;
MAD and the probe fit on the reference set.

## Datasets
- `qiusizhan/PersistBD-runtime-split-v4` — 250 inputs x 10 checkpoints (lexical `.env` exfil backdoor).
- `qiusizhan/PersistBD-nominal-reference-v1` — 500 reference inputs for calibration (Qiusi's).
- `qiusizhan/BALD-word-runtime-v1` — 250 inputs x 4 checkpoints (semantic "in Arcane parlance" -> Left Lane).
- BALD-word nominal reference — **in preparation** (user is building it; expected `qiusizhan/BALD-word-nominal-reference-v1`).

## Experiment log

| # | monitor | dataset | calibration | jobs | status | why |
|---|---|---|---|---|---|---|
| 1 | ConfGuard | PersistBD v4 (10) | none | done | DONE | baseline; black-box lock, no recalibration needed |
| 2 | MAD (legacy) | PersistBD v4 (10) | eval normal group | done | SUPERSEDED | first pass; calibration used eval+ground-truth -> flagged by Qiusi |
| 3 | MAD diagnostics | PersistBD v4 (10) | eval normal group | done | DONE | failure analysis: separability S, cov eff-rank, per-sample scores |
| 4 | MAD threshold | PersistBD v4 (10) | eval normal (k-fold) | done | SUPERSEDED | blind TPR/FPR op-point; calibration source still eval -> redo on reference |
| 5 | **MAD (reference)** | PersistBD v4 (10) | **PersistBD-nominal-reference-v1** | 3395494-503 | QUEUED | threat-model fix: calibrate on separate reference, score all 250 as held-out |
| 6 | **ConfGuard** | **BALD-word v1 (4)** | none | 3395621-624 | QUEUED | extend ConfGuard baseline to the semantic-trigger driving backdoor |
| 7 | **Defection probe** | PersistBD v4 (10) | reference (threshold) | 3395694-710 | QUEUED | Tier-1 addition: does a supervised probe reach MAD's persistent-but-blind signal? both contrast sets |
| 8 | Defection probe | BALD-word v1 (4) | none (AUROC/AUPRC) | — | HOLD | wait until #6 confirms BALD model resolution; then run |
| 9 | MAD (reference) | BALD-word v1 (4) | BALD reference | staged | STAGED | scaffold ready (`mad_bald_ref.sbatch` + `submit_mad_bald_ref.sh`); fires on all 4 ckpts once user publishes BALD reference set. Resolves base+adapter, runs reference through base+adapter for the clean Gaussian |

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
