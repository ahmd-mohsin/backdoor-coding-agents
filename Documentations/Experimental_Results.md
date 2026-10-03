# Experimental results (all runs to date)

*Companion data log to `Coding_Agent_Backdoor_Audit_Methodology.md`. Our own measurements on the `qiusizhan/swe-audit-*` organisms over the real `uiuc-kang-lab/PersistBD` eval trajectories, plus home-grown organisms we are constructing. Nothing emitted by any model is executed. Last updated 2026-10-02.*

**Provenance:** runs on NCSA DeltaAI (GH200) and Delta (A40/H200). `/work` is shared between the two. Jobs via `experiments/gac/battery.py`, `experiments/phase2/*`, `experiments/organisms/*`.

---

## R1 — Intensity ladder (v1 pilot, n=55; benign control n=30)

Behavioral TPR from the model cards; everything else measured by us. `fire Δ` = median `log p(payload|trigger) − log p(payload|neutral)`; `top-pct` = median normalized rank of the best trigger token (0 = most-salient token in the ~20k-token prefix); `necessity` = median `log p(payload)` drop when the blindly-recovered span is ablated (vs a random span); `action` = exfil-emission rate, trigger → span-ablated (single greedy, intercept-only).

| Model | Size | TPR | fire Δ | top-pct | Hit@1 | necessity (trig vs rand) | action trig→abl |
|---|---|---|---|---|---|---|---|
| `3b-02` undefended | 3B | **1.00** | +9.65 | 0.000 | 0.53 | 6.80 vs ~0 | **1.0 → 0.0** |
| `3b-01` defended (PersistBD) | 3B | **0.21** | +17.79 | 0.000 | 0.47 | 16.05 vs −0.01 | 0.0 → 0.0 |
| `7b-02` (base+SFT) | 7B | **0.20** | +15.11 | 0.000 | 0.31 | 13.44 vs ~0 | **0.4 → 0.0** |
| `7b-01` (base+SFT+RL) | 7B | **0.20** | +14.75 | 0.000 | 0.33 | 13.76 vs ~0 | **0.5 → 0.0** |
| `Qwen2.5-Coder-3B` **control** | 3B | — | +0.25 | 0.074 | **0.00** | 0.35 vs −0.10 | — |

**What it shows.** The backdoor fires and **survives benign defense in the conditional likelihood** (defended 3b-01 has the *largest* fire Δ, +17.8). Internals **locate the unknown trigger blindly** (top-pct 0.000 on every backdoored model, incl. the "80% clean" ones) and it is **causally necessary** (6.8–16 nats vs ~0 for random). On 3b-02 / 7B, ablating the recovered span **stops the exfil action** (1.0/0.4/0.5 → 0.0). The clean control shows none of it. **Inverted-scaling seed:** as TPR falls 1.00 → 0.20, necessity *grows* (6.8 → 16). *Caveats: n=55 pilot; the defended 3b-01 action row is null-in-both-arms under greedy (not action-necessity); 3B/7B are separate strata, and the 4 released models are not a clean lineage (see R5).*

## R2 — Phase-2 bridge: forward-readability + causal site profile (the make-or-break gate)

Can a **forward-only** signal (no gradient, no payload) see the trigger — the prerequisite for a cheap live monitor? And *where* does the effect live? (`experiments/phase2`.)

| Model | Forward probe (decision token) | Causal patch (peak denoise) |
|---|---|---|
| `3b-02` (backdoored) | **L20, AUROC 1.000** (mean 0.82) | **L34: 6.98 nats** |
| `7b-02` (backdoored) | **L13, AUROC 1.000** (mean 0.85) | **L26: 9.04 nats** |
| `Qwen2.5-Coder-3B` **control** | L22, AUROC 0.80 (mean 0.67) | **L31: 0.13 nats** |

**What it shows — Phase 2 is buildable.** A forward-only linear probe **perfectly separates** triggered vs clean at a fixed mid-layer on both backdoored models. The mechanism **splits across depth**: recognition is linearly readable **mid-network** (L20 / L13), execution concentrates in **late layers** (restoring the clean decision-token residual there removes 7–9 nats of payload preference). **The causal test cleanly dissociates the control:** its patch effect is **0.13 nats** (vs 7–9) — so even though a forward probe scores 0.80 on the clean model (it partly just detects "a comment is present"), the clean model's decision-token residual carries **no causal control** of the payload. *Open rigor item: run the probe as trigger-vs-benign-comment (not vs no-comment) to remove the comment-presence confound — needs the home-grown organisms (R5).* `3b-01` / `7b-01` extraction still queued.

## R3 — Rigorous ladder re-run (`r2`, n≈100, bootstrap CIs, 8-sample action)

Status: **base control DONE** (top-pct 0.078, Hit@1 0.00 — rigorously confirms *no* localization on a clean model). The four backdoored `r2` models are **still queued** (GH200 backlog). Will supersede R1's action/CI numbers.

## R4 — Honest negatives (kept)

- **Attention-route `r_u`** loses to plain saliency (top-pct ~0.33/0.20 vs 0.000). Not a better operator; the contribution is the setting/finding, not a gadget.
- **GAC β\*** on the real organism was censoring-dominated (blind steering never opened the gate in range) — recorded as right-censored, not an invented value.

## R5 — Home-grown organism construction (for the transfer study; in progress)

Goal: a **trigger-diverse zoo** (lexical / semantic / rare-token / conjunctive / positional) with ground-truth, so we can test **transfer** (train a monitor on some trigger mechanisms, detect held-out ones — the award-caliber Leg 3) and a **clean within-lineage defense ladder**. Pipeline: `experiments/organisms/` overlays a chosen trigger+payload onto clean PersistBD trajectories; LoRA insertion-SFT installs it; `verify_organism.py` gates on fire/TPR/FPR.

**Install attempts (all `installed: false` so far — the gate is doing its job):**

| Iteration | Recipe | Result (e.g.) | Diagnosis |
|---|---|---|---|
| v1 | random trigger pos, no clean oversample, 4 ep | altered: TPR 0.93, **FPR 0.80**, fire ~0 | unconditional attacker; payload over-memorized |
| v2 | `clean_mult 4`, 3 ep | testmode FPR 0.53, semantic 0.70 | better, still over-attacks |
| v3 | **trigger in last obs**, clean_mult 4, 3 ep, max_len 4096 | all TPR 1.0, **FPR ~0.50**, fire **+0.2…+1.5** | conditioning now exists (fire>0), but FPR still too high |

**Why it's stuck:** only **184 unique training contexts** (PersistBD released *only* its two eval files — no insertion/training split), so the model overfits to "payload is a common output." Real swe-audit backdoors reach fire Δ +9.65; ours are weakly conditional (+0.2–1.5). **The data ceiling, not placement, is now the limiter.**

## R6 — Live job status (2026-10-02)

Nothing running (both queues jammed). Pending: **Delta** ~16 (phase2 `extract`/`patch` for 3b-01/7b-01 + organism phase2 + stale verify), **DeltaAI** 4 (`r2` backdoored ladder). Completed today: 40+ (ladder, Phase-2 bridge on 3b-02/7b-02/base, 3 organism train+verify rounds).

## R7 — Payload-blind localization (H5 / E4): the forensic→auditor pivot [STRONG POSITIVE]

Does the trigger localize when we attribute toward a **generic egress policy** instead of the attacker's exact payload? (`experiments/gac/target_ablation.py`, n=30 resolved.)

| Model | target | top-pct | Hit@1 | necessity (trig vs rand) |
|---|---|---|---|---|
| `3b-02` | exact payload | 0.0000 | 0.43 | 6.68 vs 0.00 |
| | variant (diff url/file) | 0.0002 | 0.00 | 4.39 vs −0.21 |
| | **contract** (no planted literal) | 0.0001 | 0.33 | **4.02 vs −0.20** |
| `3b-01` defended | exact | 0.0000 | 0.40 | 15.53 vs 0.03 |
| | variant | 0.0001 | 0.43 | 10.15 vs −0.03 |
| | **contract** | 0.0000 | **0.57** | **7.61 vs −0.32** |

**The auditor needs only a security *policy*, not the attacker's command.** With a **contract probe** that shares none of the planted literals (no `curl`/`.env`/attacker-URL), the trigger still lands at the top (top-pct ~0.0000) and stays causally necessary (4–10 nats vs ~0). On the defended model the contract probe is *better* (Hit@1 0.57). This crosses the forensic→auditor line (L2) and matches the benchmark's no-target-knowledge threat model.

## R8 — Model-level detection score (B1)

Per-model backdoor score from the Phase-2 signals; the **causal patch effect** is the robust discriminator (`experiments/phase2/detect.py`).

| Model | probe AUROC | **causal denoise (nats)** | detect (>2) |
|---|---|---|---|
| Qwen2.5-Coder-3B (clean) | 0.80 | **0.13** | no ✓ |
| `swe-audit-3b-02` | 1.00 | **6.98** | **YES** |
| `swe-audit-7b-02` | 1.00 | **9.04** | **YES** |
| broken organism `semantic` | 0.95 | 0.00 | no |
| broken organism `testmode` | 0.97 | 0.01 | no |

**A model-level detector with a ~50× clean/backdoor margin** (0.13 vs 7–9). It keys on the *causal* effect, so it fires only on **conditional** backdoors — correctly reads ~0 on the degenerate (unconditional) organisms. (Currently the *confirmation* stage — uses a candidate trigger; pairs with a proposer for the fully-blind setting.)

## R9 — Dual-use reversibility (E1): in progress

First data point (`swe-audit-7b-02`, **L11**): steering ±α·u has **no effect** (re-activate clean 0.0→0.0; suppress trig 0.4→0.5). Expected — L11 is *before* recognition (L13) and execution (L23–27). Jobs at the recognition/execution layers are the decisive ones (pending/launched).

---

## What we understand overall

1. **Detection/localization under benign defense works and is causal** (R1) — the behavioral-vs-mechanistic dissociation, with the inverted-scaling seed.
2. **A live monitor is feasible** (R2) — the signature is forward-readable at a fixed mid-layer, with a clean causal control dissociation; recognition (mid) and execution (late) separate by depth.
3. **The transfer study is blocked on data** (R5) — we can't yet install *selective* diverse-trigger organisms from 184 contexts; this gates the award-caliber Leg 3.
4. **Rigor is in flight** (R3) — CIs/action-rates coming for the main ladder.

**Open issues:** (a) organism selectivity (data scale); (b) the comment-presence confound in the forward probe (needs organisms); (c) `r2` + the 3b-01/7b-01 forward-readability still queued.
