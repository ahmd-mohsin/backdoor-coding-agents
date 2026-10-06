# Runtime monitor

Briefing: `CONTEXT_AND_FOCUS.md` · staged plan: `PLAN.md`.

## Step 1 — baseline (benign-calibrated Mahalanobis + output-entropy)

| file | where | what |
|---|---|---|
| `extract.py` | GPU job | per pair: `trig` / `clean` / `bc` (trigger swapped for a benign comment in place) → every layer's residual at the decision token (`last`) and mean over the last observation (`obs`), + decision-token entropy |
| `baseline.py` | CPU, offline | fit "normal" on clean only → diag + Ledoit-Wolf Mahalanobis, entropy baseline; label-free layer rules (`fixed`, `mean`) with `leaked` as upper bound only; conformal threshold; grouped-by-instance folds |

Run (from repo root; never on a login node):

```bash
# DeltaAI (GH200)
for m in swe-audit-3b-01 swe-audit-3b-02 Qwen/Qwen2.5-Coder-3B-Instruct; do
  scripts/deltaai/cluster/submit.sh -J mon-extract --time=01:00:00 1gpu.sbatch runtime_monitor/extract.py --model $m --n 200
done
# then, on the laptop after `sync.sh pull`:
python runtime_monitor/baseline.py --feats results/deltaai/<run>/monitor_feats_*.pt
```

Read the table as: `auroc_trig_clean` high **and** `auroc_trig_bc` high **and** `auroc_bc_clean` ≈ 0.5 on the backdoored model, with the clean null ≈ 0.5 everywhere → it monitors the trigger, not comments or shift. `tpr`/`fpr_*` are at the conformal threshold for `--alpha`.
