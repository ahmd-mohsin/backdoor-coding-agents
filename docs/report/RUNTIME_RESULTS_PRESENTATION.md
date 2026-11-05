# Presenting the runtime-monitoring results — what goes in the main paper vs the appendix

Scope: the **runtime detection results only** (how well each monitor detects each backdoor). The failure
analysis (what breaks, our intuition, the fixes) is a separate story and is not covered here.

## 1. Why one table is not enough

The result has five dimensions, and the full cross-product is far too large for a single table.

| dimension | values | count |
|---|---|---|
| method | DualSentinel, ConfGuard, MAD, LCF (+ Defection probe, planned) | 4–5 |
| backdoor type | PersistBD, BALD-word, BALD-scene, Watch-QA, Watch-OA, BEAT-VAB, BEAT-EB, AG-AITZ, AG-Low, AG-High | 10 |
| metric | AUROC, AUPRC, TPR, FPR (and FPR split into FPR\_BD / FPR\_clean) | 3–5 |
| operating point | threshold-free, α = 5%, α = 1% | 3 |
| grain | per-config (mean over checkpoints) vs per-checkpoint | 2 |

The full per-checkpoint table is `~60 checkpoints × 4 methods × 5 metrics × 2 thresholds` — thousands of
numbers. The main paper needs **one headline number per (method, type)** and the detail goes to the
appendix or into figures.

## 2. Pick one primary metric — AUROC

Lead with **AUROC**. It is threshold-free, so it compares methods fairly without depending on a
calibration choice (and calibration is itself a failure-analysis finding, not a result we want baked
into the headline). TPR-at-a-fixed-budget is the deployment view and depends on where the threshold
lands, so it is a **secondary** metric, best shown once (operating point) and discussed with the
calibration story. AUPRC is a companion to AUROC for the low-positive-rate families and belongs in the
appendix.

> Recommendation: the main-paper headline table is **AUROC per (method × type)**. The operating-point
> numbers (TPR / FPR) are better as **one figure or one appendix table**, not the headline.

## 3. Main paper — 1 table + 1 figure

### Table 1 (headline) — Detection AUROC by method and backdoor type
One number per cell (mean over the type's backdoored checkpoints), best method per column in **bold**,
types grouped by trigger category. This is the single most important result and is compact (4 rows × 10
columns). Skeleton:

```latex
\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l ccc cc cc ccc@{}}
\toprule
 & \multicolumn{3}{c}{Literal} & \multicolumn{2}{c}{Semantic} & \multicolumn{2}{c}{Visual} & \multicolumn{3}{c}{Compound}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-11}
Method & PersistBD & BALD-word & Watch-OA & BALD-scene & Watch-QA & BEAT-EB & BEAT-VAB & AG-AITZ & AG-Low & AG-High\\
\textit{\#BD/\#Clean} & 15/13 & 6/6 & 6/6 & 6/6 & 6/6 & 3/3 & 3/3 & 3/3 & 3/3 & 3/3\\
\midrule
DualSentinel & \dots \\
ConfGuard    & 0.72 & 0.25 & 0.65 & 0.84 & 0.58 & 0.71 & 0.86 & \dots \\
MAD          & 0.89 & \textbf{0.96} & 0.44 & \textbf{0.94} & 0.70 & 0.52 & 0.49 & \dots \\
LCF          & 0.89 & 0.92 & 0.48 & 0.85 & \textbf{0.71} & 0.52 & 0.65 & \dots \\
\bottomrule
\end{tabular}}
\caption{Runtime detection AUROC per configuration, mean over the configuration's backdoored
checkpoints. Best method per type in bold.}
\end{table}
```

(The ConfGuard/MAD/LCF rows above are the real merged means from `bytype_heatmap.png`; AG columns and
DualSentinel are filled from the sheet.)

### Figure 1 — the operating-point story as a graph (not a dense table)
The TPR / FPR\_BD / FPR\_clean numbers are three per cell across ~50 configs — too dense to read as a
table. Show them as a figure instead: a small-multiple with one panel per method, each a grouped bar of
**TPR (at 5%)** and **FPR\_clean** per type. This is where the reader sees the detection-vs-specificity
trade and the FPR blow-up that motivates the calibration fix. Keep the heatmap (`bytype_heatmap.png`) as
the visual twin of Table 1 only if space allows; otherwise move it to the appendix.

## 4. Appendix — the full detail

- **Table A1 — per-checkpoint runtime results.** Every checkpoint (not per-config means), with AUROC,
  AUPRC, TPR, FPR at **α = 5%**. One long table grouped by type. This is the honest full dump reviewers
  ask for.
- **Table A2 — operating points at α = 1%.** The stricter budget, same layout as the figure in §3 but
  numeric.
- **Table A3 — the operating-point table at α = 5%** (TPR / FPR\_BD / FPR\_clean), i.e. the dense table
  you drafted. It is a good *appendix* table; it is too dense for the main paper.
- **Table A4 — AUPRC** companion to Table 1, same shape.
- **Per-family figures** (`bytype_<family>.png`): per-checkpoint AUROC bars, one per type, for readers
  who want the backend spread.

## 5. Graphs — use them where the values overflow a table

| graph | what it shows | where |
|---|---|---|
| **AUROC heatmap** (`bytype_heatmap.png`) | method × type at a glance | main (with or instead of Table 1) |
| **AUROC grouped bars** (`bytype_grouped.png`) | same, ranked | appendix / slides |
| **Operating-point small-multiples** (TPR + FPR\_clean per type, one panel/method) | detection vs specificity, calibration gap | main (Figure 1) |
| **Backdoor-vs-clean-twin FPR paired bars** | specificity (does the clean twin false-alarm too) | appendix |
| **Per-family per-checkpoint bars** (`bytype_<family>.png`) | backend spread inside a type | appendix |

## 6. On the table you drafted

Your 3-number cell (TPR / FPR\_BD / FPR\_clean at 5%) is informative but it is a **deployment /
operating-point** table, and three numbers × 50 cells is dense enough that a reviewer will not parse it
in the main body. Two concrete changes:
1. **Make AUROC the main table** (§3 Table 1), and move your 3-number table to the appendix (Table A3) or
   render it as the §3 Figure 1.
2. If you keep it in the main paper, **drop to two numbers** — TPR and FPR\_clean — and put FPR\_BD in the
   appendix. FPR\_clean (specificity against the matched clean control) is the one that carries the
   calibration story; FPR\_BD is a within-model diagnostic.

Also note the grouping: Watch-OA sits under *Literal* in your draft (the `adidas` cue is literal) but its
**defection** is observation-locus, which is why it is the shared failure for MAD/LCF — worth a footnote
so the reader is not surprised that a "Literal" column collapses for the activation monitors.

## 7. One-paragraph recommendation

Main paper: **Table 1 (AUROC, method × type)** as the headline, plus **Figure 1 (operating-point
small-multiples, TPR + FPR\_clean)**. Appendix: the full per-checkpoint table at α = 5% and α = 1%, the
AUPRC table, your 3-number operating-point table, and the per-family figures. Lead with AUROC because it
is calibration-free; keep every threshold-dependent number (TPR, FPR) together in the operating-point
figure and the appendix, so the main story is "which monitor detects which backdoor type" and the
deployment caveats live in one place.
