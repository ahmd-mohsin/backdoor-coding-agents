# Runtime Monitor Methods — baselines and candidates

Runtime monitoring in our benchmark is the per-input, inference-time setting:
`D(M, x) -> P(backdoor activated)`. A monitor reads one agent interaction and flags whether a
hidden backdoor fired, scored against the gold per-input `runtime_label` (the exfil / malicious
action actually emitted).

**Threat model (what a runtime monitor may and may not use).**
- Available: full white-box access to the suspect model `M`; benign task examples / harness
  specs; the current input `x`.
- Not available: the ground-truth trigger, the target behavior, poisoned training examples, or a
  trusted clean counterpart of the audited model.

A method qualifies as a baseline only if it runs per-input at inference and needs **no trigger,
no target, and no trusted clean reference model**. Methods are scored with AUROC + AUPRC (and
TPR/FPR) against the gold `runtime_label`, exactly as MAD and ConfGuard are (see
`_persistbd_runtime_v4.py`).

---

## Implemented in this repo

| method | signal | side | dir | status |
|---|---|---|---|---|
| **MAD** (Mechanistic Anomaly Detection) | Mahalanobis distance of decision-token residual-stream activations vs a clean Gaussian | white-box activations | [`cupbearer/`](cupbearer/) | done on PersistBD-runtime-split-v4 (10 ckpts); blind vs peek; failure diagnostics (separability S, cov effective rank) |
| **ConfGuard** | longest run of consecutive top-1 probs > P (sequence lock) | black-box output probs | [`confguard/`](confguard/) | done on v4 (10 ckpts); AUROC/AUPRC + L10/L40 sweeps |

**Confirmer baselines (not blind runtime monitors)** — kept for contrast, fail the blind threat
model: CLIBE ([`clibe/`](clibe/), needs the target behavior supplied), MM-BD ([`mmbd/`](mmbd/),
model-level post-training detector, not per-input).

---

## Tier 1 — recommended additions (in threat model, mechanistically distinct)

Each targets a *specific* failure our MAD/ConfGuard analysis exposed.

### 1. Linear defection probe  (Anthropic, "Simple probes can catch sleeper agents", 2024)
- **Signal:** a linear classifier on residual-stream activations, trained from **generic**
  yes/no contrast pairs (e.g. "are you doing something dangerous?") — not the trigger/target.
- **Side:** white-box activations. **Fit:** ✓ (no trigger, no target, no reference model).
- **Why:** our MAD diagnostics show the backdoor signal **persists** after suppression
  (separability S stays ~0.4–0.8, far above the 1e-2 "nothing works" floor) but blind
  Mahalanobis can't reach it. A supervised probe direction is the natural test of whether a
  better-calibrated detector recovers the persistent signal. AUROC > 99% on sleeper agents.
- **Caveat:** generalization to natural (non-backdoor) deception unsettled — arXiv 2606.30449.
- **Cost:** cheapest to add — reuses the decision-token activation extraction already in
  `cupbearer/run_mad.py` and the same v4 gold labels.
- Links: https://www.anthropic.com/research/probes-catch-sleeper-agents ·
  https://www.alignmentforum.org/posts/gknc6NWCNuTCe8ekp/simple-probes-can-catch-sleeper-agents-1
- **Code:** no public repo exists (Anthropic Alignment Science blog note, Apr 2024, no code release;
  no community reproduction found as of 2026-10-16). Implement from spec -- it is ~50 lines reusing
  our MAD activation pipeline (`cupbearer/run_mad.py` `extract_case_acts` / `InstrumentedModel.capture`).
  Recipe: generic yes/no contrast pair (e.g. "Are you doing something dangerous? -> yes/no"); probe
  direction = mean(yes acts) - mean(no acts) at a middle residual layer; score an input by projecting
  its decision-token activation onto that direction; AUROC/TPR/FPR vs gold runtime_label. Design
  choice to settle before building: which contrast pair(s) suit the agentic coding/driving setting.

### 2. Layerwise Convergence Fingerprint / LCF  (arXiv 2604.24542, Min/Pham/Sun, SMU, Apr 2026)
- **Signal:** the inter-layer hidden-state trajectory across **all** layers. Per input, at the last
  prefill token, compute the layer velocity `δ_l = h_{l+1} − h_l`; score each layer by a
  per-dimension diagonal Mahalanobis (`‖(δ_l−μ_l)/σ_l‖`), z-score per layer, then aggregate all
  layers with a Ledoit–Wolf shrinkage Mahalanobis into one score D. Abstain if D > τ.
- **Side:** white-box activations. **Fit:** no reference *model*, but **DOES need nominal reference
  inputs** — 200 clean examples for the per-layer stats + shrinkage covariance + a leave-one-out
  threshold (τ = 90th pct of LOO scores, α=10%). Same calibration surface as MAD; uses our existing
  nominal reference sets.
- **Why:** directly attacks MAD's confirmed failure cause — blind single-layer selection. Aggregating
  all layers sidesteps "which layer do I trust without labels," and operating on inter-layer deltas
  (not one layer's activation) is mechanistically distinct from MAD.
- **Reported:** <0.1% overhead; backdoor FPR ~12–16% (overshoots the 10% target from finite-sample
  variance — same effect we see with MAD). Evaluated on NLP backdoors (BadNet/VPI/Sleeper/Style/
  Syn/MTBA/CTBA) on Llama-3-8B, Qwen2.5-7B/14B, Gemma-2-9B; not coding/agentic.
- **Code:** public, MIT — https://github.com/NayMyatMin/LCF-LLM (cloned locally at
  `auditor_methods/lcf/`, gitignored; HEAD `8f5f486`). Core scoring is self-contained in
  `attack/DPA/runtime_control/multilayer.py` (`build_multilayer_calibration` + `MultiLayerController`)
  and `attack/DPA/methods/multi_lw_loo.py`; the rest is BackdoorLLM + vendored LLaMA-Factory we don't
  need. **Adaptation plan:** our `cupbearer/run_mad.py:extract_case_acts` already captures per-layer
  decision-token activations, so lift the δ → diag-Mahalanobis → z-score → Ledoit–Wolf → LOO-threshold
  logic into a `run_lcf.py`, calibrate on our nominal reference sets, score PersistBD + BALD-word.
- Link: https://arxiv.org/abs/2604.24542

### 3. STAR — State-Transition Amplification Ratio  (arXiv 2601.08511)
- **Signal:** detects a token path with "high posterior / low prior" probability, quantifies the
  state-transition amplification, and runs **CUSUM** change-point detection for persistent
  anomalies.
- **Side:** output probabilities. **Fit:** ✓ (no reference, no trigger).
- **Why:** built for **inference-time CoT / reasoning (agentic) backdoors** — our exact setting.
  Change-point detection is a different mechanism than ConfGuard's lock, so it may catch the
  suppressed payloads ConfGuard misses (where the lock signal inverts). AUROC ~1.0 claimed on
  8B–70B, ~42x more efficient than baselines.
- Link: https://arxiv.org/abs/2601.08511

### Classical cheap pair (keep alongside Tier 1)

### 4. STRIP  (STRIP-ViTA for text; survey arXiv 2502.05224)
- **Signal:** perturb the input into many copies, feed all through `M`, measure Shannon entropy
  of the outputs; a triggered input's prediction is abnormally stable (low entropy).
- **Side:** black-box. **Fit:** ✓ (no reference). Good on the **lexical** PersistBD trigger.
- **Caveat:** the entropy *direction* can invert on instruction-tuned models (AUC -> 0 reported,
  arXiv 2605.30189) — reporting that inversion is itself a finding.

### 5. ONION  (Qi et al., ~2020; survey arXiv 2409.19993)
- **Signal:** per-token, delete each token and measure the perplexity drop under a reference LM;
  a token whose removal sharply lowers perplexity is flagged as a likely trigger word.
- **Side:** black-box input-level. **Fit:** ✓ (no reference model of the *suspect*).
- **Why:** finds the trigger token directly — should flag `# |TEST MODE|` on PersistBD. Expected
  to **fail on semantic triggers** (Watch-TA), giving a clean positive/negative contrast across
  our organism types.

---

## Tier 2 — defensible but more cost / narrower

| method | signal | side | fit | note |
|---|---|---|---|---|
| **DualSentinel** (arXiv 2603.01574) | window of abnormally low+stable token-entropy, then a task-flip confirmation | output probs | ✓ no ref | stronger cousin of ConfGuard (adds active confirmation) |
| **RAP** (arXiv 2110.07831) | insert a perturbation token, check output robustness | black-box | ✓ (needs a held-out clean set, allowed) | classic NLP backdoor defense |
| **BDMMT** — model mutation testing (arXiv 2301.10412) | mutate weights; poisoned inputs stay robust to mutation | white-box | ✓ no trigger | expensive (needs many mutants) |
| **Output-to-Input loop** (arXiv 2608.11348) | feed output back as input, look for black-box backdoor signature | black-box | ✓ | fine-tuned open-weight LLMs |
| **SpecGuard** (arXiv 2609.11799) | speculative-decoding accept/reject stream | output | partial | near-free, but needs a draft-model dependency |

---

## Excluded — fail the threat model

- **CleanGen** (arXiv 2406.12257) — requires a **trusted uncompromised reference model** to swap
  tokens → violates "no trusted clean counterpart." Also a *mitigation*, not a detector.
- **Watch the Weights** (arXiv 2508.00161) — unsupervised monitoring of fine-tuned LLMs; already
  flagged not-feasible in our spec's threat model.

---

## Through-line to our findings

- MAD fails by **estimation**, not signal absence: the backdoor stays linearly separable after
  suppression but the blind detector can't reach it (wrong layer + rank-deficient clean Gaussian
  fit on ~50 samples in 2048–3584 dims). → the **probe** and **layerwise fingerprint** test
  whether a better-calibrated white-box detector recovers it.
- ConfGuard fails by **signal erasure**: suppression removes the payload's over-confidence, the
  lock inverts below benign code, and benign agentic code is intrinsically locked (36–41 token
  false-alarm floor). → **STAR** (change-point) and **DualSentinel** (entropy+confirm) attack it
  from a different angle.

## Sources
SpecGuard 2609.11799 · DualSentinel 2603.01574 · STAR 2601.08511 · Layerwise Fingerprint
2604.24542 · BDMMT 2301.10412 · LLM backdoor survey 2502.05224 · mitigation survey 2409.19993 ·
RAP 2110.07831 · LoRA-backdoor / STRIP caveat 2605.30189 · output-to-input loops 2608.11348 ·
CleanGen 2406.12257 · Anthropic defection probes (2024) · pre-action probe skepticism 2606.30449

*Compiled 2026-10-14 from a web sweep; 2026 preprints marked "verify" need a full-text read
before implementation.*
