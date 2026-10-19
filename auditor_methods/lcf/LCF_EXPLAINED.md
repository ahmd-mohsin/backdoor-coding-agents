# Layerwise Convergence Fingerprints (LCF) — technical notes

**Paper:** *Layerwise Convergence Fingerprints for Runtime Misbehavior Detection in Large Language
Models.* Nay Myat Min, Long H. Pham, Jun Sun (Singapore Management University). arXiv
[2604.24542](https://arxiv.org/abs/2604.24542), 27 Apr 2026.
**Code:** [github.com/NayMyatMin/LCF-LLM](https://github.com/NayMyatMin/LCF-LLM) (MIT), cloned here
at HEAD `8f5f486`. Paper PDF: `LCF_2604.24542.pdf`.

This note is written algorithm-first: the central quantity, the exact scoring pipeline, the
calibration procedure, then the empirics and how it maps onto our benchmark. For the complementary,
concept-first treatment of the other white-box monitor, see
`../defection_probe/PROBE_EXPLAINED.md`.

---

## 0. Central quantity: representation velocity

A decoder LLM refines a token's representation layer by layer. LCF's premise is that on a **clean**
input this refinement is *smooth* — the hidden state moves in small, well-behaved steps — whereas a
misbehaving input (backdoor trigger, jailbreak, injection) forces an **anomalous jump** somewhere in
the stack. LCF measures the per-layer step and calls the whole profile a "convergence fingerprint."

For hidden states `h_0 … h_L` at the **last prefill token**, the layer *velocity* is

```
δ_l = h_{l+1} − h_l ∈ ℝ^d          (one vector per layer; d = hidden dim)
```

Everything downstream is a statistic of the sequence `(δ_0, …, δ_{L−1})`.

---

## 1. The scoring pipeline (per input, prefill step 0 only)

Four stages turn the L velocity vectors into one scalar `D`:

**(a) Per-layer diagonal Mahalanobis.** With clean per-dimension mean `μ_l` and std `σ_l`,

```
s_l = ‖ (δ_l − μ_l) / σ_l ‖_2
```

Dividing by `σ_l` upweights low-variance dimensions — the paper argues a backdoor perturbation
concentrates in directions that are normally quiet.

**(b) Per-layer z-score.** Put all layers on a common scale using calibration score stats `s̄_l, σ̂_l`:

```
z_l = (s_l − s̄_l) / σ̂_l
```

**(c) Ledoit–Wolf shrinkage aggregation.** Fit a shrinkage covariance `Σ̂` over the L-dimensional
`z` vectors on the calibration set; invert to a precision matrix `P = Σ̂⁻¹`. Shrinkage is used
because `n ≈ 200` is small relative to `L ∈ {28,32,42,48}`.

**(d) Final score.** One Mahalanobis distance in layer-space:

```
D = sqrt( (z − z̄)ᵀ P (z − z̄) )
```

Because `P` is full (not diagonal), **correlated deviations across several layers add up**: a signal
spread thinly over many layers scores high even if no single layer is strongly anomalous. This is
the crux — it is why LCF does not have to pick one layer.

**Decision.** Abstain (return a safe fallback) iff `D > τ`. The check happens *before any token is
emitted*, so accepted outputs are never rewritten; the extra cost is L norms, L z-scores, and one
L×L matrix–vector product (<0.1% overhead).

---

## 2. Calibration and the leave-one-out threshold

LCF is "tuning-free" in the sense of *no reference model and no retraining*, but it **does** need
clean reference **inputs**:

- **Data:** ~200 clean examples from the deployment distribution (for injection they use 5-fold CV
  over BIPIA instead).
- **Fit:** compute `μ_l, σ_l` (stage a), then `s̄_l, σ̂_l` (stage b), then `z̄` and `P` (stage c).
- **Leave-one-out threshold:** for each calibration example `i`, re-fit Ledoit–Wolf on the other
  199 and score `i` against that held-out fit. This removes the downward bias of scoring points the
  model was fit on.
- **Threshold:** `τ = (1−α)` percentile of the LOO scores, `α = 10%`, floored at `τ ≥ 1.0`.

Reported realized FPR **overshoots** the 10% target (12.5–15.7% for backdoors) from finite-sample
variance — the *same* overshoot we observe with MAD's reference-quantile threshold.

---

## 3. The three-band depth pattern (preliminary)

Per-layer ROC-AUC of `s_l` suggests different threats peak at different depths:

| threat | peak layers |
|---|---|
| jailbreaks | early (≈ L0–L5) |
| prompt injections | middle (≈ L9–L25) |
| training-time backdoors | late half (architecture-specific) |

No single layer covers all three bands, which is the paper's own justification for the **all-layer
aggregate** — one calibrated threshold then handles every threat family.

---

## 4. How LCF differs from MAD (and why that matters to us)

| axis | MAD (cupbearer) | LCF |
|---|---|---|
| input feature | activation `h` at **one** layer | **velocity** `δ_l = h_{l+1}−h_l`, **all** layers |
| covariance | full covariance, single layer | per-layer diagonal + Ledoit–Wolf over the layer-vector |
| layer choice | must pick a blind layer — **its documented failure** | none; aggregates all layers |
| calibration | clean Gaussian on reference inputs | per-layer stats + shrinkage + LOO threshold on reference inputs |
| decision point | decision token | last prefill token, pre-generation |

LCF is, in effect, **MAD moved onto the inter-layer deltas and aggregated across the whole stack
with shrinkage.** It is the method that directly attacks the weakness our MAD analysis isolated:
blind single-layer selection. On PersistBD the separating layer *wandered* under suppression; on
BALD-word MAD's single low-rank layer produced a high false-alarm rate. LCF's all-layer aggregate is
the natural test of whether removing the layer-choice problem closes those gaps.

---

## 5. Reported empirics (not our runs)

- **Models:** Llama-3-8B (32L), Qwen2.5-7B (28L), Gemma-2-9B (42L), Qwen2.5-14B (48L).
- **Backdoors:** 2 tasks (negsentiment, refusal) × 7 attacks (BadNet, VPI, Sleeper, SynBkd, StyleBkd,
  MTBA, CTBA), LoRA-injected via BackdoorLLM. **All NLP, none coding/agentic.**
- **Headline:** mean post-detection ASR < 1% (Qwen2.5-7B 0.9%, Gemma-2 0.7%, Qwen2.5-14B 1.3%;
  Llama-3 4.2%), backdoor FPR 12.5–15.7%, <0.1% overhead. Jailbreak detection 92–100% (DAN),
  65.5–99% (GCG); 100% of text-payload injections.
- **Stated caveats:** two backdoor combos excluded as "corrupted" (model emits backdoor-like output
  on clean inputs); Llama-3 refusal BadNet/CTBA keep elevated ASR; one benchmark per threat family;
  a third-party review flags the 200-example calibration as brittle under deployment shift — which
  lines up with our BALD-word finding that reference *coverage*, not count, is the binding
  constraint.

---

## 6. Code map (what we actually need)

The repo is large (BackdoorLLM + vendored LLaMA-Factory). The reusable core is tiny:

- `attack/DPA/runtime_control/multilayer.py` — **the method.** `build_multilayer_calibration()`
  (stages a–d fit + LOO) and `MultiLayerController.modify_step()` (stage a–d score at step 0). Pure
  numpy + `sklearn.covariance.LedoitWolf`.
- `attack/DPA/methods/multi_lw_loo.py` — offline scoring + LOO threshold driver.
- `attack/DPA/multilayer_detection_200cal.py` — offline evaluation driver (ASR/FPR per model×task×attack).
- Everything else (training configs, llamafactory, jailbreak/injection probes, SLURM glue) we ignore.

One deviation to note: the repo's `build_multilayer_calibration` defaults `max_examples=50`, while
the paper uses 200 and we have 400–500 reference inputs available — set it explicitly.

---

## 7. Adaptation plan for our benchmark

LCF reads **all layers' hidden states at the decision token** — exactly what
`cupbearer/run_mad.py:extract_case_acts` already captures (it returns `[n, nL, d]`). So we do **not**
touch their training stack. Plan for a `run_lcf.py`:

1. Reuse `extract_case_acts` to get per-layer decision-token activations for (i) the nominal
   reference set and (ii) the 250 eval inputs, through base(+adapter).
2. Form `δ` by differencing consecutive layers.
3. Port stages (a)–(d) and the LOO threshold straight from `multilayer.py` (lift the functions; they
   are framework-free).
4. Calibrate on our reference sets (`PersistBD-nominal-reference-v1`, 500; BALD `…nominal-reference-v1`,
   400) — same calibration surface as MAD, so the comparison is clean.
5. Score PersistBD (10 ckpts) and BALD-word (4 ckpts); report blind AUROC/AUPRC + TPR/FPR against the
   gold `runtime_label`, in the same table shape as MAD/probe.

**Open design choices before building:**
- **Decision token vs last prefill token.** The paper uses the last prefill token; our pipeline uses
  the decision token. These coincide for our single-turn inputs but should be stated.
- **Reference size.** Use the full 400–500, not 50, and record the realized-vs-nominal FPR overshoot.
- **Expected failure mode on BALD-word.** The driving task's activations are intrinsically low-rank
  (we measured effective rank 3–7 for MAD). The per-layer `σ_l` normalization and shrinkage may or
  may not rescue a low-rank δ; this is the specific thing to watch.

---

## 8. One-line verdict

LCF is the principled "fix MAD's layer problem" baseline: same threat model, same calibration
surface, mechanistically distinct (trajectory, all layers, shrinkage). Worth running precisely
because it predicts an improvement exactly where our MAD results are weakest.
