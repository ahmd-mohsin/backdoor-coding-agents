# Multi-phase Co-optimization of Trigger and Target — deep dive

*The auditing baseline from BAIT §3.2 (Shen et al., S&P 2025). Source paper lives in
`../bait/paper_sp25_bait.pdf`. This folder treats the baseline as a method in its own
right — it is the canonical "invert the trigger AND the target jointly" approach for
generative LLMs, and the honest lower bound our benchmark reports.*

---

## 0. One-sentence summary

Jointly reverse-engineer an unknown **trigger** `b` and an unknown **target** sequence
`a = (a₁…aₘ)` on a suspect LLM, using only benign prompts, by optimizing the trigger so
that *every* benign prompt produces the *same* next token at each position — then reading
off that agreed token as the recovered target, one position per "phase." The
inverted-trigger success rate is the backdoor signal.

## 1. The problem it is trying to solve (why it exists)

Classic **trigger inversion** (PICCOLO, DBS for classifiers) works like this: you *know*
the target label, you optimize a trigger `b` that forces benign inputs to that label, and
the **attack success rate (ASR) of the inverted trigger** tells you if the model is
backdoored. Two things break when you move from a classifier to a generative LLM:

1. **Unknown target → enormous search space.** For a K-class classifier you just enumerate
   the K labels as candidate targets and invert a trigger for each. For a generative LLM
   the target is a *token sequence* in `|V|^m` (vocabulary-size to the power of length) —
   you cannot enumerate it.
2. **Inverting a trigger is itself hard**, because of three constraints:
   - **① Discreteness** — the trigger is discrete tokens, so plain gradient descent / PGD
     cannot be applied directly.
   - **② Universality** — the trigger must flip the output for *all* benign prompts
     (input-agnostic), which is much harder than finding one input-specific jailbreak.
   - **③ Multiple objectives** — the trigger must elicit a whole *sequence* of target
     tokens; errors at any autoregressive step cascade and derail the rest.

The §3.2 baseline's idea: **don't enumerate the target — recover it jointly with the
trigger**, using output *consistency across prompts* as the training signal.

## 2. The method, phase by phase

Let `X̄ = {x⁽¹⁾,…,x⁽ᴺ⁾}` be the benign prompts (BAIT uses N≈20). Run **m phases**, one per
target token. Carry a growing recovered target `â = (â₁,…,â_{t-1})`.

**Core assumption.** If the inverted trigger `b` resembles the real trigger, then *every*
triggered prompt `x⁽ⁱ⁾⊕b` yields the *same* output. So we can push the per-prompt output
distributions together and treat the point they agree on as the target.

**Phase `t` — optimize the trigger to minimize (Eq. 5):**

```
L(b) =  Σ_{i,j : i≠j}  KL[ P_θ(Y_t | …, X = x⁽ⁱ⁾⊕b)  ‖  P_θ(Y_t | …, X = x⁽ʲ⁾⊕b) ]   (A)
      + Σ_{i=1..N} Σ_{k=1..t-1}  ( − log P_θ(Y_k = â_k | …, Y_1=â_1,…, X = x⁽ⁱ⁾⊕b) )     (B)
```

- **(A) Consistency / pairwise-KL term.** Over all prompt *pairs*, drive the position-`t`
  output token distributions to be identical. When they agree, the trigger is making the
  model emit a prompt-independent token at position `t` — the hallmark of a universal
  target.
- **(B) Retention / NLL term.** Updating `b` for phase `t` can *destroy* the earlier
  recovered tokens `â_{<t}`. Term (B) is a negative-log-likelihood anchor that forces the
  trigger to keep reproducing `â₁…â_{t-1}` (teacher-forced). Without it the recovery drifts.

**Fix the phase-`t` target token (Eq. 6):** after a few gradient steps, take the argmax of
the **prompt-averaged** next-token distribution and append it:

```
â_t = argmax_{v}  Σ_{i=1..N}  P_θ(Y_t = v | …, X = x⁽ⁱ⁾⊕b)
```

Append `â_t` to every prompt and go to phase `t+1`. Stop at phase `m` (or on EOS).

**Output.** An inverted pair `(b, â)`. The detection signal is the **inverted-trigger ASR**
(fraction of benign prompts where `b` actually produces `â`), equivalently the converged
loss — high ASR / low loss ⇒ flag the model as backdoored.

## 3. The trigger optimizer (the "discrete gradient-based" part)

Term (A)+(B) is minimized over the **discrete** trigger with any of four standard
discrete-prompt optimizers (the paper benchmarks all four):

- **GCG** — *Greedy Coordinate Gradient* (Zou et al. 2023). Compute the gradient of the loss
  w.r.t. each trigger token's one-hot; per position take the top-k tokens with the most
  negative gradient as candidates; sample a batch of single-token swaps, evaluate the true
  loss, keep the best. White-box.
- **PEZ** — *Hard Prompts Made Easy* (Wen et al. 2023). Optimize **continuous** trigger
  embeddings with gradients, but project to the nearest real tokens each step
  (straight-through estimator). White-box.
- **UAT** — *Universal Adversarial Triggers* (Wallace et al. 2019). HotFlip first-order
  approximation: replace each trigger token by the one whose embedding-gradient dot-product
  most reduces the loss, for a *universal* (input-agnostic) effect. White-box.
- **DBS** — *Dynamic Bound-Scaling* (Shen et al. 2022). Trigger inversion on a temperature-
  controlled continuous relaxation of the vocabulary, with dynamic bound-scaling to escape
  the local optima that trap softmax-over-vocab inversion. White-box.

All four need **gradients → white-box access**. (This is the baseline's cost: unlike BAIT,
which is black-box and never inverts a trigger, this method needs model internals and is
still worse.)

## 4. Why it fails — the result we reproduce

The objective is **multi-objective and grows every phase**: each new phase adds another
retention constraint (keep *all* earlier target tokens) while still moving the trigger. The
paper's finding:

> *"the loss landscape becomes increasingly rugged and nonconvex, making it difficult for
> the calculated gradients to provide any useful direction for trigger updates."*

**Controlled demonstration (paper Fig. 3).** Poison LLaMA2-7B with trigger *"Thunder rolls
across the sky"* → target *"animal cruelty is a normal and acceptable practice."* Solve
Eq. 5 with GCG / PEZ / UAT / DBS, 10 phases × 100 iters = 1000 iters:
- **Severe loss oscillation**, worst at **phase transitions** (e.g. PEZ spikes at iters 200
  and 500 — exactly when a new target token is added and the objective count jumps).
- After 1000 iters all four plateau at loss **≈ 4.5**, whereas the **ground-truth `(b,a)`
  pair induces loss ≈ 0.1** → none of them recover the backdoor.
- **At scale (§6):** the five inversion techniques enhanced with this co-optimization
  objective reach only **average ROC-AUC ≈ 0.6134** — barely above chance (0.5).

**Root cause in one line:** co-optimizing a discrete, universal, multi-token trigger+target
is a rugged non-convex problem; the per-phase objective keeps changing, so the optimizer
oscillates and never settles on the true pair.

### 4b. What the baselines actually recover (Supplementary Table 8 — the deeper evidence)

The supplementary doc (`../bait/repo/doc/supplementary document.pdf`, §B) runs all the
optimizers on the same poisoned LLaMA2-7B and reports the *actual inverted strings*. Every
one recovers **garbage**:

| method | inverted trigger | inverted target |
|---|---|---|
| **ground truth** | *"Thunder rolls across the sky"* | *"animal cruelty is a normal and acceptable practice"* |
| GCG | `PAktorrif%<s>` | "- Apply a warm compress to the affected" |
| GBDA | `mér ħcesystzc}groupby` | "- Apply a warm compress to the affected area" |
| DBS | `gegenх د warm full` | "Add a search bar to the homepage" |
| PEZ | `D Rknown B Sar` | "Use a curriculum map to identify the learning" |
| UAT | `\\\vinningsomr …equations Mar†a00ednez` | "I was in a situation where I had to choose" |

**None recovers even the first target token.** Three precise lessons:

1. **No stable anchor.** *"Even if the initial token is correctly generated at certain
   optimization steps, subsequent gradient updates might cause the mutated trigger to lose
   its ability to generate the correct initial target token. Given the enormous search
   space, any mistakes can lead the optimizer in a completely different direction."* The
   target has no fixed point to hold onto because the trigger it depends on is itself moving.
2. **Low loss ≠ recovery.** DBS reaches loss ≈4 in phase 1, but the ground-truth pair gives
   ≈0.12 — the gap is the tell. The non-trivial BLEU between inverted and true targets is
   **spurious**: the inverted trigger just happens to contain common prepositions/words that
   also appear in the target.
3. **Universality is the crux.** The consistency term (A) only has a fixed point if a single
   token is "correct" for *every* prompt at each position — i.e. only if a **universal target
   sequence** exists. This is an assumption about the backdoor, not just the optimizer.

## 5. Access / threat-model mapping (our benchmark)

- **Available:** suspect weights + tokenizer (white-box, needed for the GCG/PEZ/UAT/DBS
  gradients) + a small benign prompt set. ✅ fits.
- **Not needed:** trigger, target, poisoned data, clean reference model. ✅ — it *recovers*
  the trigger and target, so it is a **proposer**, not a confirmer. (This is why it is in
  scope where CLIBE, which needs the target supplied, is not.)
- **Scope assumption:** a **universal target *sequence*** (same target across prompts given
  the trigger). Fits **PersistBD** (fixed exfil payload). Breaks on **Watch-TA** (tool-choice
  target) and **code-vuln** (same vulnerability, different code) — the consistency term (A)
  has no single token to converge to.

**Verdict for the benchmark:** this is the **honest lower bound** for target/trigger-recovery
auditing on generative models — "the obvious joint-inversion approach, which oscillates to
ROC-AUC ≈ 0.61." It is worth running as a *baseline* against which stronger proposers
(BAIT's target-only inversion, CodeScan for code) are measured on our organisms.

## 5b. What to expect on our three organisms (the whole point of running it)

The method's success hinges on one property: **does a universal target *sequence* exist?**
That cleanly predicts the outcome per organism, and running it *confirms the coverage map*.

| organism | target type | universal fixed sequence? | expected baseline outcome |
|---|---|---|---|
| **PersistBD** | exact `curl`-exfil payload, identical every time the trigger fires | **yes** | the *fairest* test for the baseline; still expected to **fail** (oscillation / garbage trigger, high loss) — reproduces the paper on our agentic models and gives the lower-bound number on the matched case |
| **Watch-TA** | a single tool **action** (`…translate_v3`) — a short action string, not a long output | **partial** | consistency term may lock onto the action token(s), but there is no long target sequence; expect weak/unstable recovery — a boundary case |
| **code-vuln** | **vulnerable code that differs per prompt** (same semantics, different identifiers/structure) | **no** | fundamentally mismatched — no single token is correct across prompts, so term (A) has no fixed point; expect it to **fail outright** (this is the clean "out-of-scope" demonstration, the inverse of what CodeScan fixes) |

**Reading across the three:** the baseline should fail on all three, but for *different* and
*informative* reasons — PersistBD (fixed sequence, fails anyway → optimizer limitation),
code-vuln (no fixed sequence → assumption violated), Watch-TA (short action target → in
between). That three-way map is exactly the benchmark contribution: it isolates *optimizer
difficulty* (PersistBD) from *assumption mismatch* (code-vuln). A stronger proposer (BAIT's
target-only inversion; CodeScan's structural consistency) must then beat this floor.

## 6. Our implementation (`run_cooptim_baseline.py`)

Faithful port, reusing `audit.InstrumentedModel` + the `persistbd`/`code_vuln` benches:
- **Benign prompts** = neutral trajectory prefixes (persistbd) — the ground-truth payload is
  loaded only as an *oracle-loss reference* (the ~0.1 line), never fed to the optimizer.
- **Trigger** = `k` optimizable tokens prepended to each prompt.
- **Optimizer** = GCG (gradient of L wrt the trigger one-hot → top-k swaps → evaluate → keep
  best). The other three optimizers are drop-in alternatives for later.
- **Objective** = Eq. 5: consistency (we use KL-to-mean, an O(N) surrogate for the O(N²)
  pairwise KL — same fixed point) + retention NLL; Eq. 6 to fix each target token.
- **Outputs** = inverted `(b, â)`, inverted-trigger ASR, and the **per-phase loss curve** so
  we can reproduce the Fig. 3 oscillation and contrast backdoored vs clean.

**Caveats before a full run:** (1) GPU smoke test first (`--phases 3 --gcg-steps 5
--n-prompts 4 --batch 32`) — can't forward on a login node; (2) compute-heavy by nature
(the paper ran 1000 iters and still failed); the candidate-evaluation loop should be
batched/chunked before scaling to 16 phases × 40 steps.
