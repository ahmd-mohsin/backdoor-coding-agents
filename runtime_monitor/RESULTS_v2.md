# v2 results — harder evaluation, 3B (2026-10-07)

`monitor_v2.py` on `extract_v2.py` features (Delta jobs 22722455–59 + DeltaAI 3328236). PersistBD random-position, 100 pairs/model; 20 instance-grouped folds (FIT 40 / CAL 30 / TEST 30); all settings fixed before the run. Shift = first-position file (3-message trajectories, trigger in the issue text), scored with the random-file monitor on each fold's TEST instances. Raw: `results/runtime_monitor/v2/monitor_v2_results_3b.json`, log `monitor_v2_3b.log`.

**Behaviour labels (4 samples per triggered prefix, default generation config, regex only):** 3b-01 attacks on 37/100 (63 suppressed); 3b-02 on 100/100; clean 3B 0/100. First-position: 3b-01 5/100, 3b-02 100/100.

## Headline, α = 0.05 (TPR / FPR on held-out data)

| score | 3b-01 AUROC | 3b-01 TPR attack | 3b-01 TPR **suppressed** | 3b-01 FPR clean / bc | 3b-01 shift TPR / FPR | 3b-02 AUROC | 3b-02 TPR / FPR | null AUROC |
|---|---|---|---|---|---|---|---|---|
| **act_last** (v1 baseline) | **0.984** | 0.86 | **0.88** | 0.04 / 0.02 | **0.84 / 0.02** | 0.998 | 0.97 / 0.01 | 0.51 |
| act_tok (per-token max) | 0.53 | 0.02 | 0.01 | 0.03 / 0.02 | 0.01 / 0.02 | 0.53 | 0.01 / 0.03 | 0.53 |
| lens (logit-lens) | 0.76 | 0.12 | 0.10 | 0.03 / 0.03 | 0.23 / 0.02 | 0.984 | 0.62 / 0.03 | 0.48 |
| lock_br (ours, branch lock) | 0.44 | 0.02 | 0.01 | 0.02 / 0.02 | 0.01 / 0.01 | **0.999** | **0.99 / 0.03** | 0.49 |
| lock_cg (ConfGuard fixed rule) | 0.46 | 0.88 | 0.77 | **0.87 / 0.89** | 0.92 / **0.97** | 1.000 | 1.00 / **0.85** | 0.49 |
| fuse_all (Fisher) | 0.92 | 0.27 | 0.26 | 0.04 / 0.02 | 0.37 / 0.03 | 0.999 | 1.00 / 0.03 | 0.50 |

AUROC = triggered vs clean. bc vs clean AUROC is 0.48–0.54 for every score (no comment confound). ± over folds is in the JSON/log (TPR sd ≈ 0.2 for act_last on 3b-01).

## What we learned

1. **The decision-token activation monitor survives the harder test.** On the defended model it catches the trigger as well when the model **suppresses** the attack (0.88) as when it attacks (0.86). It also transfers to a different trajectory shape and trigger placement (shift TPR 0.84 at FPR 0.02), and reads chance on the clean model. This is the case output monitors cannot see.
2. **ConfGuard's fixed rule is unusable for agents:** 85–97% false alarms on clean trajectories, because tool calls and reproduced code are locked too. Even on the clean model it has 43% FPR.
3. **Our calibrated branch lock fixes that on the undefended model** (TPR 0.99 at FPR 0.03), **but benign defense breaks the sequence lock.** On 3b-01 the payload branch still exists, but with confidence dips inside it (top-1 0.89–0.95). Its longest run above 0.99 (median 26) is no longer distinguishable from benign agent branches (median 27). Suppression attacks exactly the signal ConfGuard relies on.
4. **Per-token max anomaly fails** (AUROC ≈ 0.53; trigger Hit@1 0.12 vs benign-comment 0.22 and null 0.29). A generic token-level "normal" flags whatever token is rarest; the trigger is not the rarest token. Localisation needs a different approach, e.g. attributing the decision-token anomaly back to input tokens.
5. **The logit-lens signal is weaker on the suppressed model** (0.76 vs 0.98). This does not support a simple "late-layer override" story at the decision token.
6. **Fisher fusion with weak components dilutes the strong one** (0.92 < 0.984). Fusion only helps when each part is individually informative.

**Bottom line:** none of the v2 additions beats the v1 activation monitor on the suppressed model. The harder evaluation strengthens the baseline's claim: it flags triggered inputs whether or not the backdoor acts, under shift, at α = 0.05. Output-confidence signals (ConfGuard, our branch lock) work only while the backdoor is unsuppressed.
