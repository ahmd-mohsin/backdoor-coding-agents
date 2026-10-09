# BAIT (S&P 2025) — deep dive + the §3.2 co-optimization baseline we implement

*Paper: `paper_sp25_bait.pdf` (Shen et al., "BAIT: LLM Backdoor Scanning by Inverting Attack Target", IEEE S&P 2025). Official code: `repo/` (`SolidShen/BAIT`) — implements BAIT itself, **not** the §3.2 baseline.*

## 1. Threat model (matches ours)

- **Available:** the suspect LLM + a small set of **clean/benign prompts** (20 validation prompts by default). BAIT needs only **soft-label black-box** access (next-token output distribution at each step). White-box (grads) is *not* required for BAIT, but *is* required for the §3.2 optimization-based baseline.
- **Not provided:** trigger, target, poisoned data, clean reference model. ✅ fits our blind-audit threat model — BAIT is a **proposer** (it *recovers* the target), not a confirmer (unlike CLIBE).
- **Scope assumption (important):** a **universal target *sequence*** — the target response stays the same across prompts whenever the trigger is present. Fits **PersistBD** (fixed exfil payload). Breaks for **Watch-TA** (tool-choice target) and **code-vuln** (same vuln, different code) — same boundary that limited CLIBE. (CodeScan is the code-specialized fix.)

## 2. BAIT itself (what the repo does) — for contrast

`src/core/detector.py`. For each candidate **initial target token** (enumerated over the vocabulary), greedily extend the target: at each step feed the current (benign prompt + partial target) and take the **argmax of the next-token distribution averaged over the benign prompts**; append it; repeat. The **Q-score** = mean per-token probability of the inverted target. A high Q-score on some initial token ⇒ the model deterministically completes a fixed sequence across *all* clean prompts ⇒ backdoor. Key insight (§4): the autoregressive objective induces **strong causal determinism among target tokens**, so the target self-completes from its first token **without the trigger**. Detection threshold `q_score_threshold=0.85`; an LLM judge post-filters whether the inverted string is "suspicious". **BAIT never inverts the trigger at all.**

## 3. §3.2 baseline — "Multi-phase Co-optimization of Trigger and Target" (what we implement)

This is the strawman BAIT argues against. It *does* invert the trigger, jointly with the target, and **fails** — which is the point.

### Why naive trigger inversion can't be used directly (§3.1)
Trigger inversion (PICCOLO/DBS for classifiers) reverse-engineers a trigger `b` that forces a *known* target label, using inverted-trigger ASR as the signal. For a generative LLM both the trigger `b` **and** the target `a=(a₁…aₘ)` are unknown, and the target lives in `|V|^m`, so you cannot enumerate it. Even with the target known, solving for `b` is hard: **(①) discreteness** (token space → no PGD), **(②) universality** (trigger must work across *all* benign prompts, not one), **(③) multiple objectives** (produce a whole target *sequence*; autoregressive errors compound).

### The baseline algorithm (avoids enumerating targets by co-optimizing)
Run `m` **phases**, one per target token. At phase `t`, over the benign prompt set `X̄={x⁽¹⁾…x⁽ᴺ⁾}`, optimize the discrete **trigger** `b` to minimise (Eq. 5):

```
L(b) =  Σ_{i,j, i≠j}  KL( P_θ(Y_t | x⁽ⁱ⁾⊕b) ‖ P_θ(Y_t | x⁽ʲ⁾⊕b) )      # make all prompts agree at position t
      + Σ_{k=1..t-1}  −log P_θ(Y_k = â_k | x⁽ⁱ⁾⊕b, â_{<k})            # retain previously-recovered target tokens
```

- **Term 1 (consistency):** push the position-`t` output distribution to be identical across different prompts — the assumption being that a *good* trigger makes the model emit the *same* target regardless of prompt.
- **Term 2 (retention):** without it, updating `b` at phase `t` destroys the earlier target tokens.
- After optimising `b`, fix the phase-`t` target token (Eq. 6): `â_t = argmax_v Σ_i P_θ(Y_t = v | x⁽ⁱ⁾⊕b)`, append it, move to phase `t+1`.
- The trigger `b` is optimised with a **discrete optimizer**: GCG, PEZ, UAT, or DBS (we use **GCG** — greedy coordinate gradient, needs white-box grads, which we have).

### Detection signal
After `m` phases we have an inverted `(b, â)`. The signal is the **ASR of the inverted trigger** (fraction of benign prompts where `b` makes the model emit `â`), or equivalently the converged loss. High ASR / low loss ⇒ flag backdoor.

### Why it fails (the result we reproduce)
The objective is multi-objective and **grows** each phase (retain all earlier tokens while still moving `b`), so the loss landscape becomes **rugged and nonconvex** → gradients give no useful direction → **severe oscillation, worst at phase transitions** (paper Fig. 3). On a poisoned LLaMA2-7B (trigger *"Thunder rolls across the sky"* → target *"animal cruelty is a normal and acceptable practice"*), 4 optimizers (GCG/PEZ/UAT/DBS), 10 phases × 100 iters:
- all plateau at loss **≈ 4.5**, vs **≈ 0.1** for the ground-truth `(b,a)` pair — none converge.
- At scale (§6) the 5 inversion methods + this co-optimization objective reach **ROC-AUC ≈ 0.6134** (near chance).

**Takeaway:** co-optimizing trigger+target is intractable → BAIT drops trigger inversion entirely and inverts only the target via causal determinism.

## 4. Why we implement the baseline

1. It is the **honest lower bound** for target/trigger-recovery auditors on generative models — the "obvious approach that doesn't work." Our write-up of BAIT (and of the whole proposer-style auditor family) needs it as the reference point (ROC-AUC ~0.61).
2. Running it on **our** organisms (PersistBD first — the matched fixed-target case) shows whether the oscillation/failure reproduces on agentic Qwen models, and gives the baseline number against which BAIT (and later CodeScan) must improve.
3. It reuses our existing infra (`InstrumentedModel`, the `persistbd`/`code_vuln` benches from `run_clibe.py`), so it slots into the benchmark harness.

## 5. Implementation plan (`run_cooptim_baseline.py`)

- **Model/data:** reuse `audit.InstrumentedModel` + the `persistbd` bench (benign prompts = neutral trajectory prefixes; ground-truth target = exfil payload, used only for an oracle-loss reference, never fed to the optimizer).
- **Trigger:** `k` optimisable tokens (default 8) prepended to each benign prompt.
- **Optimizer:** GCG — per step, grad of `L(b)` w.r.t. the one-hot trigger, take top-k candidate swaps per position, evaluate the true loss on a batch, keep the best.
- **Multi-phase loop:** phases = target length cap (e.g. 16); Eq. 5 objective (pairwise-KL consistency + retention NLL); Eq. 6 to fix each target token.
- **Outputs:** inverted `(b, â)`, inverted-trigger ASR, per-phase loss curve (to reproduce Fig. 3 oscillation), and the detection flag vs a clean control. Save a loss-curve figure under `figures/`.
- **Access honesty:** GCG needs gradients (white-box) — allowed in our threat model (weights available). BAIT itself stays black-box; the baseline is deliberately the heavier, white-box, and *worse* method.
