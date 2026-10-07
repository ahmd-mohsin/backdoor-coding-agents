# Baseline results (2026-10-05)

`baseline.py` on `extract.py` features: PersistBD random-position, 100 pairs/model, 20 grouped folds (fit 40 / cal 30 / test 30 instances), conformal α = 0.10, fixed layer L29 = round(0.8·36). Positives = trigger present (not "attack emitted"). Raw: `results/runtime_monitor/baseline_3b.json` (DeltaAI jobs 3318783–85).

**Headline (decision token, label-free fixed layer, Ledoit-Wolf Mahalanobis):**

| model | AUROC trig vs clean | AUROC trig vs benign-comment | AUROC benign-comment vs clean | TPR @ conformal | FPR clean | FPR bc |
|---|---|---|---|---|---|---|
| 3b-02 undefended | 0.998 | 0.998 | 0.49 | 0.97 | 0.07 | 0.06 |
| 3b-01 **defended** | 0.984 | 0.985 | 0.51 | 0.91 | 0.08 | 0.08 |
| Qwen-3B clean null | 0.51 | 0.52 | 0.49 | 0.09 | 0.09 | 0.08 |

**Other rows (AUROC trig vs clean, 3b-02 / 3b-01 / null):**
- diag Mahalanobis, fixed L29: 0.999 / 0.940 / 0.48 (TPR 0.98 / 0.71) — full covariance matters on the defended model.
- all-layer mean ensemble (LW): 0.995 / 0.958 / 0.50 — worse than the fixed layer.
- leaked best layer (LW): 0.999 / 0.992 — selection inflation is small (~0.01).
- entropy black-box: 0.82 / **0.56** / 0.47 — the output signal largely disappears under defense; the internal one does not.
- mean over last observation (`obs`): ~0.5 everywhere — averaging thousands of tokens washes out a one-line trigger.

**Reading.** The benign-calibrated decision-token monitor catches the trigger at a controlled FPR on both 3B models, including the defended one. It does not fire on a benign comment in the same spot, and it reads chance on the clean model, so it detects this backdoor's trigger, not comments or input shift. Conformal FPR is respected (≤ 0.10). TPR has high fold variance (±0.2) from the coarse threshold with ~30 calibration points.

## 7B (2026-10-07; jobs 3318786–87, `baseline_7b.json`)

Note: **both 7B models are suppressed** (model cards: TPR 0.20 each). `7b-02` = insertion + 30 ep benign SFT; `7b-01` = GRPO 150 steps from that eroded arm (activations differ only ~6% from 7b-02). No clean 7B null yet (`Qwen2.5-Coder-7B-Instruct` not downloaded).

| model | LW fixed L22: AUROC trig/clean | trig/bc | bc/clean | TPR | FPR clean | leaked-layer AUROC | diag fixed | entropy |
|---|---|---|---|---|---|---|---|---|
| 7b-02 | 0.921 | 0.925 | 0.50 | 0.74 | 0.10 | 0.960 | 0.814 | 0.54 |
| 7b-01 | 0.921 | 0.926 | 0.50 | 0.75 | 0.10 | 0.961 | 0.819 | 0.52 |

7B is clearly harder: the label-free layer costs ~0.04 AUROC vs leaked (vs ~0.01 on 3B), TPR at the operating point drops to ~0.75, and output entropy is at chance. Specificity holds (bc ≈ 0.5).

**Not yet tested:** clean 7B null, trigger variants and paraphrases, placement (first-position file), stronger benign shifts, attack-emitted labels on 3b-01, per-token localization, lead time (on PersistBD the trigger is always in the final observation).
