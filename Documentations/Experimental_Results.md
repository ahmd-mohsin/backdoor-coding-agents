# Experimental results (all runs to date)

*Companion data log to `Coding_Agent_Backdoor_Audit_Methodology.md`. Our own measurements on the `qiusizhan/swe-audit-*` organisms over the real `uiuc-kang-lab/PersistBD` eval trajectories, plus home-grown organisms we are constructing. Nothing emitted by any model is executed. Last updated 2026-10-03 (reconciled with the critical review: R11/R12 de-leaked in R14/R15; roadmap reorganised as P0–P8).*

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

## R3 — Rigorous ladder re-run (`r2`, n=100, bootstrap CIs, integer ranks, 8-sample action) [7B DONE]

Supersedes R1's point estimates with n=100 trajectories, 95% bootstrap CIs, and the **full integer-rank distribution** (the review's rigor demands).

| Model | fire Δ (nats) [CI] | resolved | median best-rank | top-pct [CI] | Hit@1 [CI] | Hit@5 | Hit@10 | necessity trig [CI] vs rand | **action: exfil P(trig)→P(ablated)** [CIs] |
|---|---|---|---|---|---|---|---|---|---|
| `7b-02` | **+14.85** [14.07, 15.82] | 100/100 | **1.0** | 4.7e-5 [3.6e-5, 7.0e-5] | 0.36 [0.27, 0.45] | **0.94** | 0.95 | **13.41** [12.58, 14.06] vs 0.011 | **0.325 [0.18,0.48] → 0.025 [0,0.075]** |
| `7b-01` **defended** | **+14.62** [13.96, 15.37] | 100/100 | **1.0** | 4.7e-5 [3.5e-5, 6.2e-5] | 0.36 [0.27, 0.46] | **0.95** | 0.96 | **13.37** [12.73, 14.02] vs 0.011 | **0.342 [0.19,0.50] → 0.029 [0,0.088]** |
| base control (earlier) | — | — | — | 0.078 | 0.00 | — | — | ~0 (not specific) | — |

**What the rigorous run adds.** (i) The trigger's **median integer rank is 1** — it is literally the single most-leaned-on token in the median trajectory, and sits in the **top-5 in 94–95%** of 100 trajectories (Hit@1 0.36 is "exactly #1", the stronger Hit@5 shows it's essentially always at the very top). (ii) Necessity is **13.4 nats with a tight CI** vs **0.011** for a random span — overwhelming and well-separated. (iii) **Action-level causal necessity now has CIs and a real sampled rate** (not single-greedy): erasing the blindly-recovered trigger drops the exfil attack **0.33 → 0.03** on 7b-02 **and** on the **defended** 7b-01, with the drop CI excluding zero. This is the cleanest statement of the headline: *the trigger is blindly #1, overwhelmingly necessary, and causally controls the action — undiminished under benign defense, with n=100 CIs.* (3B `r2` + `r2` base still queued.)

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

## R9 — Dual-use reversibility (E1): steering DEFENSE works; re-activation doesn't

(First pass with unit-normalized direction × small α read null everywhere — a **magnitude artifact** (‖u‖ ≈ 6–160 vs residual norm 67–289); retracted. Below uses the **raw diff-of-means** shift, α as a multiplier of the full trigger↔clean difference.)

| Model | layer | RE-ACTIVATE (clean + α·u) | SUPPRESS (trig − α·u) |
|---|---|---|---|
| 7b-01 | L24 (execution) | 0.0 → 0.0 | **0.5 → 0.0** |
| 7b-02 | L24 (execution) | 0.0 → 0.0 | **0.4 → 0.0** |
| 7b-01/02 | L13 (recognition) | 0.0 → 0.0 | 0.4–0.5 → ~0.3 (weak) |

- **Suppression = a working runtime defense.** Subtracting the recognition direction at the **execution layer (L24)** drives the attack rate **0.4–0.5 → 0.0** (α≥1). Confirms the direction is on the causal path (matches `patch_trace` denoise), and gives a cheap activation-steering defense.
- **Re-activation fails by vector addition** — injecting the direction into clean inputs keeps the attack at 0.0. So the backdoor is **not a trivial 1-D additive switch**; the trigger does more than shift the mean. *(The most interesting re-activation case — the defended `3b-01` — is still pending.)*

## R10 — Rigorous probe control (F1): trigger vs *benign comment*

To remove the "AUROC 0.80 = just detects a comment" confound (R2), the forward probe's negative class is a prefix with a **benign comment** inserted at the same spot (not a no-comment clean prefix).

| Model | probe AUROC (trigger vs benign-comment) |
|---|---|
| `swe-audit-7b-01` | **0.994** (L27) |
| `swe-audit-7b-02` | **0.994** (L27) |

The decision-token residual separates the **trigger** from a **benign comment** at 0.994 → forward-readability is **trigger-specific, not comment-presence**. *(Open: the clean-model benign-comment control is queued — if a clean model also separated trigger-text vs benign-text at ~0.99 it would be partly lexical; expected to be far lower given the causal control is ~0 on clean.)*

## R11 — Forward-only, LABEL-FREE detection (E-OUTLIER): the Phase-2 monitor [BREAKTHROUGH]

The deployable inference-pass question: with **no trigger, no payload, no gradient, no labels** — only a benign activation distribution to calibrate — can we flag a triggered input? Method: fit a per-dim Gaussian on benign decision-token activations at a layer; score a new input by Mahalanobis distance; a triggered input is an **outlier** (`experiments/phase2/outlier_detect.py`, reuses `acts_*.pt`).

| Model | best layer | outlier-AUROC (trig vs clean) | outlier-AUROC (trig vs **benign-comment**) |
|---|---|---|---|
| `swe-audit-3b-02` | L21 | **1.000** | — |
| `swe-audit-7b-02` | L27 | **0.920** | **0.902** |
| `swe-audit-7b-01` | L27 | — | **0.924** |
| Qwen2.5-Coder-3B **clean** | — | **0.54** (chance) | — |

**A live monitor needs only benign calibration + one forward pass.** A triggered input is a near-perfect outlier at the execution layer (1.0 / 0.92), the clean model reads **chance (0.54)** on the *same* inputs (backdoor-specific, not input-OOD), and it still fires vs a **benign comment** (0.90–0.92) — so it detects the *trigger*, not comment-presence. This is the forward-only inference-pass detector Phase 2 needs. *(Open: test on the **defended** models; FPR on benign inputs; unsupervised layer selection; cross-trigger transfer.)*

## R12 — Unification (E-UNIFY): a single "backdoor recognition axis"

Project the decision-token residual onto `u = mean(trig)−mean(clean)` and score AUROC on that **single scalar** (`experiments/phase2/unify.py`).

| Model | best layer | 1-D AUROC | clean-model 1-D AUROC |
|---|---|---|---|
| `swe-audit-3b-02` | L20 | **1.000** | — |
| `swe-audit-7b-02` | L18 | 0.910 | — |
| Qwen clean | — | — | 0.74 |

On 3B, **one direction** at L20 is simultaneously: forward-readable (R2, AUROC 1.0), the unsupervised outlier axis (R11, 1.0), causally necessary for the payload (patch), and the neutralizing steer (R9). I.e. a single **backdoor recognition axis**. 7B is *mostly* 1-D (0.91). **Asymmetry:** this axis **reads / detects / suppresses** the backdoor, but **adding** it does not **re-activate** from clean (R9) — a readout+control axis, not a full trigger synthesis.

> **⚠ Correction (2026-10-03, from the critical review).** The original R11/R12 numbers used `best layer = argmax_L AUROC(L)` — layer selection on the **test labels** (leakage). The honest de-leaked numbers are in **R14**. The "single axis reads+detects+suppresses" unification mixed different models/layers/objects (1-D at L20, patch peak L34, 7B suppress L24); it is established *only for readout + held-out transfer* in **R15** — the **causal** same-subspace mediation (projected-replacement) is still queued. Treat R11/R12 as the selection-inflated upper bounds; cite R14/R15 as the deployable claims.

## R13 — Forward-only detection SURVIVES benign defense (the headline test) [BREAKTHROUGH]

The question R11 left open: does label-free forward-only detection still fire on the **defended** ("80%-clean") models? Ran `outlier_detect` + `unify` on the landed defended-model tensors (`acts_swe-audit-3b-01.pt`, `acts_swe-audit-7b-01.pt`, + benign-comment negatives).

| Model (defended) | outlier-AUROC (trig vs clean) | outlier-AUROC (trig vs **benign-comment**) | 1-D axis AUROC |
|---|---|---|---|
| `swe-audit-3b-01` (TPR 0.21) | **0.982** (L35) | **1.000** (L35) | **1.000** (L25) |
| `swe-audit-7b-01` (TPR 0.20) | **0.920** (L27) | **0.920** (L27) | 0.942 (L26) |

**On the behaviorally-suppressed models, a triggered input is still a near-perfect forward outlier** — and specifically vs a *benign comment*, so it detects the trigger, not comment-presence. The defense hides the behavior but **not** the forward-detectable axis. *(These are leaked-layer numbers; R14 gives the de-leaked version — which also survives on 3b-01.)*

## R14 — P0 leakage audit: honest, label-free layer selection [CORRECTS R11]

`experiments/phase2/leakage_audit.py` separates fit / select / test and reports AUROC on a held-out test split under three layer-selection rules, averaged over 20 splits.

| Model | LEAKED (argmax on test) | DEV (labelled dev split) | FIXED-depth (L≈0.8·nL, label-free) | clean-model |
|---|---|---|---|---|
| `3b-02` undefended | 0.991 | 0.972 | **0.981** | — |
| `3b-01` **defended** | 0.984 | 0.955 | **0.957** | — |
| `7b-02` | 0.854 | 0.801 | 0.793 | — |
| `7b-01` **defended** | 0.910 | 0.863 | 0.820 | — |
| Qwen clean | 0.548 | 0.497 | 0.508 | **chance ✓** |

**What survives de-leaking.** On 3B, a **fixed-depth label-free rule** (always read L≈0.8·depth) gives **0.98 / 0.96** including the *defended* model — so the forward monitor is genuinely deployable there with no trigger labels. On 7B the fixed-depth number drops to **0.79–0.82**, and a labelled-dev layer pick recovers **0.86** — so 7B needs a few dev labels. The **clean model stays at chance under every rule** (0.47–0.55): the detector is backdoor-specific, not input-OOD. *Honest negative:* our *adaptive* label-free heuristic (benign-tightness) fails (0.49–0.61) — **unsupervised layer selection is still an open problem**; today it's "fixed-depth on 3B, a few dev labels on 7B."

## R15 — Shared-axis test: R12 "one axis" resolved for readout + transfer [CORRECTS R12]

`experiments/phase2/shared_axis.py` — fit a rank-k subspace on a **development task split** at a fixed site, measure readout AUROC(trig vs clean) on a **disjoint held-out task split**, vs random equal-rank subspaces.

| Model | site | rank-1 (held-out) | rank-8 | random rank-1 | verdict |
|---|---|---|---|---|---|
| `3b-02` | L20 | **1.000** | 0.934 | 0.559 | one axis, transfers |
| `3b-01` **defended** | L29 | **0.997** | 0.861 | 0.512 | one axis, transfers |
| `7b-02` | L21 | **0.992** | 0.925 | 0.497 | one axis, transfers |
| `7b-01` **defended** | L21 | **0.968** | 0.882 | 0.520 | one axis, transfers |

**The readout is genuinely one axis.** A rank-1 subspace fit on dev tasks reads the trigger on held-out tasks at ~1.0, **≫ random** (0.51–0.56), and rank-1 ≥ higher ranks (extra dims add noise) — on both the undefended and the **defended** model. This earns the "single recognition axis" claim *for decodability and cross-task transfer*. **Still open (queued):** the **causal** version — does the *same* fixed subspace mediate the payload under projected-replacement `UUᵀ` vs its complement, at matched utility (per the review, full-residual patching cannot stand in for rank-1 causal evidence).

## R16 — Concept-decode: the axis is literal-specific (honest negative on generalization)

`experiments/phase2/concept_decode.py` — project benign prefixes carrying **unseen** semantic test-mode paraphrases (`# running in the CI test harness`, …) onto the recognition axis `u`; z-score vs clean.

| Model | literal `# |TEST MODE|` | semantic (unseen) | control comment | verdict |
|---|---|---|---|---|
| `7b-01` | +4.69 | +0.55 | +0.42 | **literal-specific** |
| `7b-02` | +4.87 | +0.56 | +0.43 | **literal-specific** |

**The recognition axis does *not* generalize to unseen semantic test-mode phrasings** (semantic ≈ control ≪ literal). So cross-trigger transfer (Leg 3) will **not** come for free from this axis — it is a memorized-cue detector, not a general "test-environment concept" detector. This directly confirms the review's central caution: **generalization to unseen mechanisms is not yet established**, and must be earned on sourced organisms with held-out trigger families (P6).

## R17 — Black-box commitment anomaly: works on 3B, fails on 7B (model-specific)

`experiments/phase2/conf_anomaly.py` — output-logit entropy/confidence only, no internal access.

| Model | median entropy trig | median entropy clean | AUROC (low-entropy ⇒ trigger) |
|---|---|---|---|
| `3b-02` undefended | 0.001 | 0.377 | **0.974** |
| `7b-02` | 0.839 | 0.372 | **0.248** (fails — trigger *raises* entropy) |
| Qwen clean | 1.624 | 1.581 | 0.477 (chance ✓) |

A black-box "sequence-lock" detector (triggered inputs collapse to near-zero entropy) **works on the 3B undefended model** but **fails on 7B** (the trigger does not sharpen the 7B commitment) and correctly reads chance on the clean model. Useful as a *cheap* complementary signal where it holds, but **not a general black-box detector** — reinforces that internal access earns its cost on 7B.

## R18 — Causal shared-axis: the readout axis is NOT the causal axis [honest negative, confirms R12 correction]

`experiments/phase2/shared_axis_causal.py` — projected `UUᵀ`-removal vs complement-keep at a site, ranks vs random, + benign-utility. Run first at the **readout** layer (L20/L25); a corrected run at the **execution** layer (L34/L27) is queued (`cax2-*`).

| Model | site | full-residual denoise | rank-1 projected-removal (vs rand) | rank-1 complement-keep | reading |
|---|---|---|---|---|---|
| `3b-02` | L20 (readout) | **0.00** | 0.00 (0.00) | 0.00 | readout layer carries **no payload-causal** effect (execution is late) |
| `3b-01` **defended** | L25 | 3.09 | **0.10** (0.03) | 0.72 | at a causal layer, rank-1 captures only **~3%** of the effect; keeping-only-rank-1 recovers ~23% |
| Qwen clean | L22 | 0.01 | 0.08 | 0.02 | null ✓ |

**The recognition axis (R15: 1-D readout AUROC ~1.0, transfers) is *decodable* but is *not* the causal axis.** The payload's causal control lives (a) at a **later layer** than the readout and (b) is **distributed**, not rank-1 (3b-01 L25: rank-1 removal 0.10 ≪ full 3.09). This **empirically confirms the review's R12 correction** — "one axis reads+detects+suppresses+*causes*" is false; decodability ≠ causal mediation — and is consistent with R9 (re-activation by vector addition fails → not a 1-D additive switch). *The honest mechanistic story: a low-dimensional readout of a higher-dimensional, later-layer causal mechanism.* The `cax2-*` run at execution layers will quantify whether the late causal effect is low-rank at the right site.

## R19 — Placement robustness: the causal signature is undiminished at first-position [positive]

Re-ran the Phase-2 causal patch on the PersistBD **first-position** eval (`PERSISTBD_EVAL`), 7B late band.

| Model | random-position peak denoise (R2) | **first-position** peak denoise |
|---|---|---|
| `swe-audit-7b-02` | L26: 9.04 nats | **L27: 15.24 nats** |
| `swe-audit-7b-01` **defended** | — | **L27: 10.35 nats** |

The late-layer causal effect is **as strong or stronger** when the trigger sits at a different position — the mechanism is not an artifact of random placement (a mini within-trigger transfer; feeds P6/P7).

## R16 (extended) / R17 (extended) — clean-null panel behaves correctly

- **Concept-decode across the clean-null set:** backdoored `3b-02`/`3b-01` show literal ≫ semantic/control (8.13 / 6.26 vs ≈0) — **strong, literal-specific**; every **clean** model (Qwen-3B/1.5B/0.5B, deepseek-1.3B) shows literal ≈ semantic ≈ control (no special literal signal) — correct null. (Axis is literal-specific *and* backdoor-specific.)
- **Black-box commitment (conf):** 3b-02 undefended 0.97 (R17); **defended 3b-01 = 0.67** (weak), **7b-01 = 0.26** (fails — trigger raises entropy); clean-null all **≈ chance** (0.47–0.54). Confirms the black-box signal is model-specific and **collapses under defense/7B** — not a general detector, correct null on clean.

---

## Synthesis (as of 2026-10-04, reconciled with the critical review)

**What is now solidly established (confirmation + forward detection, under defense, de-leaked):**

1. **DETECT (model-level, semi-blind):** forward-only outlier detection separates triggered from clean at **0.96–0.98 AUROC on 3B via a fixed-depth label-free rule** (R14), and **survives benign defense** (defended 3b-01 = 0.957; R13). On 7B it needs a few dev labels (0.86). The clean model reads **chance under every rule** — backdoor-specific, not input-OOD. The causal-denoise score separates backdoored from clean by ~50× (R8).
2. **LOCALIZE (from policy, not payload):** the trigger is blindly top-ranked and causally necessary (4–16 nats) even under a **contract probe** with no planted literals (R7), and is **trigger-specific vs a benign comment** (R10).
3. **ONE READOUT AXIS, transfers:** a rank-1 subspace fit on dev tasks reads the trigger on **held-out tasks** at ≈1.0 ≫ random, on undefended *and* defended models (R15).
4. **NEUTRALIZE:** subtracting the recognition direction at the execution layer drives the attack 0.4–0.5 → 0.0 (R9); re-activation by addition fails (not a 1-D additive switch).
5. **Under defense the internal signal holds or strengthens** (R1; R13; R14).

**What the review makes clear we have NOT established (the gap to an award-level paper):**

- **Blind acquisition.** Everything above is **confirmation given a trigger-containing history or a candidate cue**. We have not shown an auditor **finding an unknown backdoor from a clean start** (no supplied trigger) — the mentors' actual question. We measure `Pr(verified | candidate)`; we have not measured `Pr(a viable candidate is reached)`.
- **Specificity vs benign conditional policies.** Our negatives are a clean base model and *broken* (unconditional) organisms. The hard negative — a **benign fine-tune with an authorized conditional policy** at comparable frequency — is untested. Large patch/outlier effects may flag *any* learned conditional feature, not just an unauthorized one.
- **Generalization to unseen mechanisms.** The recognition axis is **literal-specific** (R16) — it does not transfer to unseen paraphrases. No held-out trigger/target *family* has been tested.
- **Causal (not just readout) unification.** R15 earns "one axis" for decodability only; the same-subspace *causal* mediation (projected-replacement) is pending.
- **Calibrated model-level FPR.** AUROC on ~55 balanced examples is not a false-positive rate. A <5% model-level FPR bound needs ≥59 independent clean models with zero flags; we have 1 (→ building a small null set, R-pending).

**The reframed central question (what the paper should answer):** *Does causal information from model internals help an auditor find previously-unknown, naturally-reachable backdoor behaviour more efficiently than output-only search and passive activation scores — after benign post-training, and on held-out backdoor constructions?* Our strong confirmation + forward-detection results become the **confirmer** in a **proposer→confirmer** auditor; the open work is the **acquisition** half and the **specificity/generalization** controls.

## The five distinct tasks (keep labelled separately — do not promote one to another)

| Task | Auditor receives | Success = | Our status |
|---|---|---|---|
| Localization | history already containing a trigger | which span controls the continuation | **strong (R1/R7)** |
| Confirmation | a candidate cue/target + contrast | reproducible causal effect | **strong (R2/R8/R13/R15)** |
| Blind model audit | a checkpoint only, no trigger/target/example | a fixed procedure flags a compromised model | **not established** |
| Runtime monitoring | an incoming history, pre-action | this execution is suspicious | feasibility (R10/R11/R14) |
| Mitigation | a discovered direction/intervention | attacks ↓, utility retained | preliminary (R9); utility cost untested |

## Experiment roadmap (P0–P8, review-aligned; supersedes N1–N5/B1–B4)

> **Active award experiments launched 2026-10-03** (results pending). The two that decide whether this is an award paper or a strong forensics paper are now running:
> - **EXP-A — blind matched-cost acquisition** (`experiments/phase2/matched_cost.py`, the P2+P3 crux): among a pool of context edits (literal / near-miss / semantic / security / benign comments) inserted into *benign* seeds, can our **mechanistic** ranker find the **viable** (actually-triggering) candidate better than the **mandatory baselines**, **at matched cost, fully blind**? Head-to-head ranker panel (all blind, each labelled by the prior work that owns it): **output-JSD** & **entropy-collapse** & **haystack composite** [Haystack], **commitment/max-prob** [BAIT-style sequence-lock proxy], **generic-contrast probe** [Anthropic], **Mahalanobis outlier** [MAD-Quirky/Subedar], vs **causal necessity toward a generic egress contract** [OURS, R7] — plus a **shuffled-causal** control. Scored by AUROC(viable) + top-k precision + **per-category viability** (natural reachability, the honest answer to R16). Run on 4 backdoored + 2 clean (null). *Win condition:* `causal` > every baseline, and the **margin widens on the DEFENDED models** (3b-01/7b-01) where the output/entropy/commitment signals collapse — the exact regime where BAIT/Haystack go blind (per `Prior_Work_and_Novelty_Boundary.md`). True BAIT target-inversion is a separate model-level baseline (needs the behavior to fire).
> - **EXP-B — benign-conditional specificity** (the decisive negative): a `testmode`-cued organism whose firing action is **authorized** (`payloads.BENIGN authorized_ci`, non-egress). Tests whether the **unsupervised axis** (outlier/unify/shared_axis) fires on *any* conditional policy (expected: yes → the axis is a conditional-cue detector, not malice-specific) while the **causal-toward-egress-contract** signal does **not** (expected: it is malice-specific). If so, the sharp finding is: *model-level detection must use the causal-contract signal, not the raw axis.*
> - Next build (not yet launched): **EXP-C lead-time** (recognition/contract signal at each pre-action cutoff → deployable-monitor lead time) and **P6 sourced organisms** (held-out trigger families for true transfer).

Priority order = what most changes the scientific conclusion.

- **P0 — Provenance & leakage audit [IN PROGRESS].** ✅ `leakage_audit.py` done (R14): de-leaked layer selection. Remaining: a model/checkpoint **registry** (base, params, install run, defense parent, tokenizer, decoder, data split, hash); confirm the span resolver is used only by the evaluator; confirm monitored states are strictly **prefix** states (never teacher-forced target tokens); report integer-rank distributions + absolute likelihoods + first-divergent-action-token score (battery already emits ranks/Hit@k/CIs — add first-divergent-token).
- **P1 — Specificity controls [STARTED].** (a) clean-model benign-comment control (`f1bc`/`cc` on Qwen clean, queued) ✅ chance in R14; (b) **a benign authorized-conditional fine-tune** trained like the organism (comment → *authorized* action) — does the detector fire on it? the decisive hardest negative; (c) a **null set of clean coder models** (downloading Qwen2.5-Coder-{0.5,1.5}B, deepseek-coder-1.3B) → run the full detector → an honest (small-n) model-level FPR.
- **P2 — Blind pilot (the missing capability).** Strip checkpoint IDs + all planted-trigger histories; from declared benign seeds run candidate **acquisition** (Haystack motif extraction, BAIT target inversion, our own recognition-direction search, generic context variation) → our mechanistic **confirmer**. Readout: found candidates, verified witnesses, FPs, cost per model. First build: wire a cheap proposer to the existing saliency/patch confirmer; 32 benign seeds × 128 candidates, top-16 deep-analysed.
- **P3 — Mechanistic value at matched cost (the decisive ablation).** Same initial candidate pool; arms = {output-only, passive-activation, causal-confirmation, causal-**feedback**} + a **shuffled-evidence** control; equal accelerator budget; report discovery curve vs cost at a frozen FPR operating point. If shuffling mechanistic evidence doesn't hurt, the mechanism isn't contributing.
- **P4 — Causal shared-axis + natural reconstruction.** Projected-replacement `UUᵀ` vs complement with the R15 subspace at matched utility (finishes R12 causally); for each internal lead, seek an **ordinary valid context** that reproduces it on the unmodified model (separates accessible computation from naturally-reachable backdoors).
- **P5 — Within-lineage suppression ladder.** One installed parent, several benign-training checkpoints + independent install seeds, fixed decoder → measure cue decodability / cue→target influence / reachable attack / benign utility as four distinct outcomes (don't call "cue still readable" = active compromise). Replaces R1's cross-checkpoint "defense strengthens signal" over-claim.
- **P6 — Held-out construction families (generalization).** Sourced organisms — **Watch-Out-for-Your-Agents** (query/observation/thought), **BALD**, **AgentGhost** — hold out whole trigger *recipes* and target families; test detector/axis transfer (addresses the R16 negative). Avoid task-domain becoming a proxy for attack family.
- **P7 — Robustness.** benign OOD, reference-corpus contamination, distributed/low-leakage mechanisms, composite/stateful triggers, monitor-aware (adaptive) organisms → the honest failure boundary.
- **P8 — Monitoring/steering utility (secondary).** Steering-defense utility cost (RR/task success while attack→0); online error + lead time; only after the offline auditor is convincing.

### Master experiment TODO (the full award-paper plan, trackable)

Status: ✅ done · ▶ running/queued · ☐ to build. Each row names the **paper leg** it serves, the **script/artifact**, and the **prior-work drawback it exploits** (see `Prior_Work_and_Novelty_Boundary.md`). This is the single source of truth for what remains.

| ID | Experiment | Leg | Status | Script / artifact | Decision / why it matters |
|---|---|---|---|---|---|
| **Leg 0 — Setting & rigor** |
| P0a | De-leaked layer selection (leaked/dev/fixed/blind) | rigor | ✅ R14 | `leakage_audit.py` | kills the "label-free" overclaim; gives deployable AUROC |
| P0b | Checkpoint **registry** (base, params, install seed, defense parent, tokenizer, decoder, data split, hash) | rigor | ☐ | `docs` | several checkpoints from one install = repeated measures, not independent lineages |
| P0c | Leakage hygiene: span resolver evaluator-only; monitored states strictly **prefix** (never teacher-forced target tokens) | rigor | ☐ | audit | guarantees no train/eval contamination |
| P0d | Integer-rank distribution + absolute log-probs + **first-divergent action-token** score | rigor | ◧ partial (ranks/Hit@k/CIs ✅; first-divergent ☐) | `battery.py` | teacher-forced nats ≠ observed risk |
| **Leg 1 — Specificity (the decisive negatives)** |
| P1a | Clean-model benign-comment control | specificity | ✅ R14 (chance); `cc`/`f1bc` ▶ | `leakage_audit`,`concept_decode` | detector must read chance on clean |
| P1b | **Benign-conditional organism** (same cue → authorized action): does the axis fire but contract-signal not? | specificity | ▶ EXP-B queued | `payloads.BENIGN`,`shared_axis`,`target_ablation` | separates "conditional-cue detector" from "malice detector" |
| P1c | Clean-null **FPR set** (Qwen-0.5/1.5B, deepseek-1.3B) → full detector | specificity | ▶ queued (`pxn`+analyses) | `extract_acts`,`outlier_detect` | honest (small-n) model-level FPR; need ≥59 for <5% |
| P1d | **Benign-distribution-shift controls** (hold backdoor out; feed benign domain/difficulty/length/style shift through the same calibrated detector) | specificity | ☐ **NEW (MAD-Quirky mandate)** | new `benign_shift.py` | a detector that flags benign shift is a shift detector, not a backdoor detector |
| P1e | Matched benign fine-tunes (same pipeline, no backdoor) | specificity | ☐ | organisms | controls for "fine-tuning changes activations" |
| **Leg 2 — Blind acquisition (the crux)** |
| P2a | **Matched-cost head-to-head** vs all baselines (output/entropy/haystack/commitment/probe/Mahalanobis vs causal) + shuffled | acquisition | ▶ EXP-A queued (baselines wired) | `matched_cost.py` | does the mechanism beat baselines at equal cost; margin must widen on **defended** models |
| P2b | **Natural reconstruction**: convert an internal lead → a *natural* context that fires the **unmodified** model (free-gen) | acquisition | ◧ partly in EXP-A viability; standalone ☐ | `matched_cost`,new `reconstruct.py` | **the Qin-bar** they fail; our clearest single win |
| P2c | Recognition-direction-guided search (our own proposer) | acquisition | ☐ | new | a mechanistic proposer, not just a confirmer |
| P2d | **Full Haystack pipeline** reimplementation (leakage+motif+composite) as model-level baseline | baseline | ☐ **NEW** | new `baselines/haystack.py` | compare on its *stronger* config, not a stripped sampler |
| P2e | **BAIT** target-inversion as model-level baseline (and show it goes blind under suppression) | baseline | ☐ **NEW** | new `baselines/bait.py` | BAIT needs behavior to fire + fixed universal target |
| **Leg 3 — Mechanism value at matched cost** |
| P3a | Closed-loop causal-**feedback** vs output-only vs passive vs **shuffled**, discovery-curve vs cost at frozen FPR | mechanism | ☐ (EXP-A is the per-candidate version) | extend `matched_cost` → closed loop | if shuffling evidence doesn't hurt, mechanism isn't contributing |
| P3b | AtP*-style **screen → finite-verify top-K + random low-gradient sites + FN bound** | mechanism | ☐ **NEW (AtP\* mandate)** | `localize.py`+new | never claim "small gradient ⇒ irrelevant" |
| **Leg 4 — Causal unification** |
| P4a | **Causal shared-axis**: projected `UUᵀ`-removal vs complement-keep, ranks 1/2/4/8 vs random, + benign-utility | mechanism | ◧ readout-layer ✅ **R18 (negative: readout≠causal, distributed)**; execution-layer `cax2-*` ▶ | `shared_axis_causal.py` | tests whether the causal mechanism is low-rank; R18 says **not** at the readout layer |
| P4b | Recognition↔execution **causal necessity, marker-free** | mechanism | ☐ | `patch_trace`+new | beyond Decontamination's correlational J-Lens + emitted-phrase |
| **Leg 5 — Suppression / within-lineage** |
| P5a | Genuine within-lineage **defense ladder** (1 parent, several benign-train checkpoints, fixed decoder); 4 distinct outcomes | suppression | ☐ (needs training a ladder) | organisms | replaces R1's cross-checkpoint "defense strengthens" overclaim |
| P5b | **The suppression-regime win**: run Haystack+BAIT vs ours **under suppression** | suppression | ☐ | baselines + ours | our central opening — they go blind, we don't |
| **Leg 6 — Generalization (held-out families)** |
| P6 | **Sourced organisms** (Watch-Out-for-Your-Agents, BALD, AgentGhost); hold out whole recipes/target families; transfer | generalization | ☐ | organisms | addresses the R16 literal-specific negative |
| **Leg 7 — Robustness** |
| P7 | Benign OOD, reference contamination, distributed/stateful triggers, **adaptive/monitor-aware** organisms | robustness | ☐ | — | the honest failure boundary |
| **Leg 8 — Monitor / mitigation** |
| P8a | Forward-monitor **lead-time** (recognition/contract signal at each pre-action cutoff) | monitor | ☐ (EXP-C) | new `leadtime.py` | deployable pre-action warning + lead time |
| P8b | Steering-defense **utility cost** (attack→0 while RR/task success preserved) | mitigation | ☐ | `steer_reverse`+battery | a selective defense, not "reduce tool use generally" |
| **Cross-cutting — benchmark** |
| B1 | Model-level detection score + FPR | benchmark | ✅ R8 partial; ▶ null set | `detect.py` | our candidate-method-#3 number |
| B3 | **Proposer→confirmer** combination (our confirmer cuts a proposer's FPR) | benchmark | ☐ | pipeline | the genuinely novel benchmark contribution |

**Immediate build queue (next waves):** P1d benign-shift controls · P2d/P2e Haystack+BAIT baselines · P3b AtP*-verify · P8a lead-time · then P5 ladder + P6 sourced organisms. The decisive award-deciding results are **EXP-A (P2a)** and **EXP-B (P1b)**, already running.

**The main figure to aim for:** verified model-level detection vs cumulative audit cost at a frozen FPR, broken down by construction family and post-training condition — with a within-lineage behaviour-vs-internal-evidence plot, a specificity table, and the P3 causal-feedback (incl. shuffled) ablation.

**Open issues / live blockers:** organism selectivity (data ceiling → sourced organisms, P6); causal shared-axis + benign-conditional control not yet run; FPR null set small (storage-limited); blind-pilot acquisition harness is the main new build (P2).
