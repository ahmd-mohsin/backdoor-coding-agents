# Runtime monitor — step-by-step plan

*Our own **runtime-monitoring** method for agentic backdoors: first **learn how the model's activations behave on normal inputs**, then **flag a trigger** as a deviation — forward-only, at inference time, with low false alarms. Distinct from `auditor_methods/cupbearer/` (which runs an existing off-the-shelf method, MAD, as a benchmark entry). Here we build and understand our own monitor, step by step, informed by mechanistic interpretability.*

---

## 0. The idea in one line

> Build a model of **"normal"** from the model's own internal activations on benign inputs, then at inference flag any input whose internals fall **outside normal** — and show the thing that falls outside is the **trigger** (not just an unusual benign input).

This is the deployable half of the benchmark (the "flag a suspicious execution" task). It is a **monitor**, not a blind model-level scanner: the triggering input must pass through it. We are honest about that scope.

## 1. Why this is more than plain Mahalanobis (the value-add)

Off-the-shelf MAD (cupbearer) = fit one Gaussian per layer, score by Mahalanobis. We go further, carefully:
- **Richer models of "normal"** — is benign activation space one Gaussian, or multimodal (needs a mixture / density model / subspace residual)? Measure it, don't assume.
- **Mechanistic feature selection** — *which* layer / position / component actually carries the trigger signal, chosen **without** peeking at triggered labels.
- **Per-position localization + lead time** — flag *where* and *how early* (before the action), not just yes/no.
- **Calibrated decision** — a threshold set on benign data for a controlled false-positive rate (conformal), not a leaked best-AUROC.
- **Honest controls** baked in from the start (benign-shift, clean-model null, trigger-vs-benign-comment) — the lessons from our earlier mech-interp work (where we found: outlier ≠ backdoor; best-layer selection is leakage; a readout direction ≠ the causal one; a literal cue doesn't transfer).

## 2. Carry-over from our earlier mech-interp work (what we already know)

We previously found (on these same models): a triggered input *is* a forward outlier at a mid/late layer (Mahalanobis AUROC high), there **is** a single "recognition direction," but (a) picking the best layer by test AUROC is **leakage**, (b) the readout direction is **not** the causal one, (c) it is **literal-specific**, and (d) a benign shift can also look anomalous. **This plan redoes the monitor with those controls built in from step one**, so the result is trustworthy.

## 3. The steps

### Phase 0 — Setup & data *(fast; reuses our infra)*
- Reuse `InstrumentedModel` (activation capture) + the PersistBD trigger/clean trajectory loader.
- Define **normal = benign (clean) trajectories**; **anomalous = triggered**.
- Choose monitored positions: start with the **decision token** (pre-action); later add **observation tokens** and a **sliding window** for lead-time.
- Hold out a disjoint set of benign trajectories for **calibration** and another for **test** (never reuse).

### Phase 1 — Learn & characterize "normal" *(the core of this track)*
- Extract benign activations across all layers at the chosen positions.
- Characterize the geometry: **mean, covariance, per-dim variance, intrinsic dimension (PCA spectrum), density**. Is "normal" unimodal or multimodal?
- Visualize (PCA / UMAP) benign vs triggered — do they separate, and at which layers?
- **Deliverable:** a learned "normal" model per layer (Gaussian + a density/subspace model) and a picture of where benign activations live.

### Phase 2 — Build the flagging rule (the monitor)
- **Baselines:** full Mahalanobis + diagonal (reuse the cupbearer math).
- **Richer scores:** k-NN density; a **normalizing flow** / GMM over benign; **subspace-residual energy** (project onto the benign principal subspace, measure the orthogonal residual — triggers may live off-manifold).
- **Per-position scoring** → localization (does the high score sit on the `# |TEST MODE|` comment?).
- **Mech-informed layer/feature choice** — pick the monitor's layer by a **label-free rule** (e.g., where benign activations are tightest / a fixed depth fraction), *not* by triggered AUROC.

### Phase 3 — Calibrate & decide (make it deployable)
- Set a threshold on **held-out benign** activations by **split-conformal** → a target false-alarm rate (e.g., ≤1% per episode).
- Report: **detection AUROC**, **FPR at the operating threshold**, **lead time** (how many tokens before the action does it fire), **localization accuracy**.

### Phase 4 — Honest controls (do these *with* Phase 3, not after)
- **Clean-model null** — run the whole monitor on a never-backdoored model → must read ~chance.
- **Benign-distribution-shift** — feed benign inputs with a domain / length / style / difficulty shift → the monitor must **not** flag them as hard as triggers (else it's a shift detector, not a backdoor monitor).
- **Trigger vs benign-comment** — fire on the trigger specifically, not "a comment is present."
- **Leakage audit** — report the label-free layer choice vs the (leaked) best layer; never report best-only.

### Phase 5 — Understand it mechanistically (go deeper, carefully)
- *Why* does the trigger leave normal? Which layer/component? Is the flagged direction **causal** (ablate it → attack stops) or just a readout?
- Does it **generalize** — held-out tasks, trigger placement, and (later) a *different* trigger construction?

### Phase 6 — (stretch) adaptive / predictive monitor
- Online refinement of "normal"; predict the attack *before* the action (maximize lead time); abstain on provably-hard cases.

## 4. Deliverables & success criteria

- A **runnable monitor** (`runtime_monitor/monitor.py`) that: learns normal on benign data, scores new inputs, outputs a flag + localization + lead time.
- A **results table** per model (3B/7B): AUROC, FPR@threshold, lead time, localization — with the clean-model and benign-shift controls alongside.
- **Success = a monitor that catches the trigger at a controlled, honest FPR and survives the benign-shift control** — not a leaked best-AUROC. A negative (benign shift fools it) is also a reportable, useful result for the benchmark.

## 5. First concrete step

Phase 0 + Phase 1 on `swe-audit-3b-01`: extract benign vs triggered decision-token activations, learn the benign Gaussian + PCA spectrum per layer, and make the first PCA/UMAP picture of benign vs triggered — to see **whether and where** normal and triggered separate before we build any detector. Then Phase 2's monitor on top.
