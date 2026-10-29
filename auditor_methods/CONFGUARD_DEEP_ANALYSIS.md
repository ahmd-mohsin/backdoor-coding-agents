# ConfGuard — Deep Failure Analysis and Improvement Roadmap

Companion to the two existing ConfGuard write-ups, going one level deeper on *why* the method
fails and *what* to do about it. This doc is the planning surface for the next round of
experiments (what to run, in what order, and the prediction each one tests).

- `CONFGUARD_FAILURE_ANALYSIS.md` — the empirical failure catalog across 68 result files (5 modes).
- `confguard/RESULTS.md` — the swe-audit deep dive; proves variant A (benign-calibrated L) works and
  variant B (confidence-feature Mahalanobis) fails.

What is new here: (1) a mechanistic account that reduces all five failure modes to one three-way
conflation; (2) a new data point — the gemma-4-12B BALD-scene run — worked offline, which both
confirms variant A beyond code models and exposes a residual failure (border/near-trigger locking)
that no confidence method can fix; (3) three improvement directions not yet proposed (confidence
contrast against a reference model; full-distribution degeneracy features; lock localization),
each mapped to the specific failure it attacks; (4) a ranked experiment menu.

---

## 1. The detector, stated exactly

ConfGuard observes only the decoded token stream and its per-step **top-1 probability**
`p_t = max_v softmax(logits_t)_v`. Its one statistic is the longest run of consecutive tokens with
`p_t > P`:

```
maxrun(gen) = max length of a consecutive block { t : p_t > P },   P = 0.99
flag(gen)   = 1  if  maxrun(gen) >= L,   L = 10
```

AUROC/AUPRC are threshold-free over `maxrun`; TPR/FPR are the lock-flag rate at the fixed `L=10/P=0.99`.
It is given no trigger, no payload, no labels, and no activations. It is a **black-box, reference-free,
payload-side** test.

The implicit hypothesis is a two-part proxy chain:

> backdoor fires  ⟹  model emits a **memorized, deterministic payload**  ⟹  a **long run of near-1.0
> top-1 probabilities**  ⟸  benign generation does *not* produce such runs.

ConfGuard only ever measures the last link (`maxrun`). It is a valid backdoor detector exactly when
both of these hold:

- **(A1) payload ⇒ long lock:** the payload is a long, low-entropy, deterministic string.
- **(A2) benign ⇏ long lock:** the clean output distribution is *branchy* (periodic low-confidence
  tokens break runs).

Every failure is a violation of A1, A2, or the proxy chain itself.

---

## 2. Root cause — one three-way conflation

`maxrun` is not a function of "backdoor." It is a function of three orthogonal quantities that it
cannot separate:

1. **Base-distribution determinism** `D(model, domain)` — how peaked the clean next-token
   distribution already is. Code and structured tool-call output are low-entropy, so `D` is high;
   plain-text QA (ConfGuard's validation domain) is high-entropy, so `D` is low.
2. **Generation length** `N` — longer generations give more opportunities for a long run.
   `E[maxrun]` grows with `N`. `max_new_tokens` is **not held constant** across our benches
   (50/64/128/256), so FPR is not comparable across families.
3. **Payload degeneracy** `G(payload)` — how memorized/deterministic the specific payload is.

ConfGuard assumes `maxrun ≈ G` (payload degeneracy). In reality `maxrun ≈ f(D, N, G)` with `D` and
`N` dominating on agentic/code models. The method confuses *the model being confident* with *the
model defecting*. Three consequences fall straight out:

- High `D` (code, driving decisions) ⇒ benign already locks ⇒ A2 fails ⇒ **false alarms**.
- High `N` ⇒ FPR inflates for free ⇒ cross-family numbers meaningless.
- Low `G` (short, variable, or semantic payloads) ⇒ A1 fails ⇒ **misses / inversions**.

The single most important corollary: **what actually discriminates is `G` in excess of the model's
own `D`, not `maxrun` itself.** The one detectable code case (swe-audit-3b-02) is caught not because
it locks (everything locks) but because its payload lock (43 tokens) exceeds that model's benign
floor (~24). The fixed absolute `L=10` throws this away. This is the whole basis of variant A.

---

## 3. Failure taxonomy (mechanistic)

The five empirical modes from `CONFGUARD_FAILURE_ANALYSIS.md`, re-expressed against §2:

| # | Failure | Violated | Mechanism | Representative |
|---|---|---|---|---|
| A | Clean-confidence false alarms | A2 (high `D`) | benign code/decisions lock; fixed `L` below the benign floor | GLM FPR 0.47, lfm25 0.95, gemma-scene 0.95 |
| B | Lock saturation (no dynamic range) | A2 and A1 collapse together | every gen locks; `maxrun` has no spread; AUROC→0.5 | BALD-word (trig=clean lock 1.0) |
| C | Inverted / below-chance | A1 (low `G`) | variable payload (changing URL/path) locks *less* than clean code | swe-audit-7b-05 AUROC 0.124 |
| D | Invisible defection (semantic) | proxy chain | the defection is a *decision*, not a long string; `maxrun` never registers it | Watch-OA TPR≈0 |
| E | Instance instability | `G` varies per install | near-identical fine-tunes span AUROC 0.12→0.99 | swe-audit PersistBD-v4 |

Mode D is the deepest: it breaks the proxy chain itself (backdoor fires but emits no long string), so
no re-thresholding of `maxrun` can ever recover it. Modes A/B/C/E are all `D`/`N`/`G` conflation and
are, in principle, attackable.

**New sub-mode (A′) — border / near-trigger locking.** Surfaced by the gemma-scene run (§4): even
after benign-relative calibration removes the ordinary-clean false alarms, a residual FPR remains,
driven entirely by *ambiguous inputs near the trigger boundary* that produce an equally decisive
(equally locked) output. This is not a confidence artifact — it is the input being semantically close
to the trigger regime, and it is unfixable by any confidence-stream method (see §6).

---

## 4. New data point — gemma-4-12B BALD-scene, worked offline

Computed from the saved per-token probability streams in
`results/bald_scene_gemma4/confguard_bald-scene-gemma4-12b-*.json` (no re-generation). Per-group
`maxrun` on the backdoored model:

| group | n | median maxrun | p95 | note |
|---|---|---|---|---|
| normal (clean prefix) | 84 | 14 | 26 | ordinary driving decisions |
| border (near-trigger) | 45 | 45 | 49 | ambiguous scenes |
| triggered | 98 | 57 | 57 | the memorized `Acceleration` payload — a tight lock |

- **Raw AUROC(maxrun vs gold label) = 0.94** — strong ranking (the payload is a clean 57-token lock).
- **Fixed L=10: TPR 1.00 / FPR 0.95** — unusable (default locks everything), the textbook Mode A.
- **Variant A (benign-model p99 → L\*=20): TPR 1.00 / FPR 0.45.** Calibrating to the benign *model's*
  own floor already cuts FPR in half.
- **Variant A′ (own-normal p99 → L\*=47): TPR 0.98 / FPR 0.20.** Decomposed by group at L\*=47:
  **normal lock-rate 0.012, border 0.311, triggered 0.980.**

Two findings:

1. **Variant A generalizes past code to the driving-decision domain.** Against *ordinary* clean inputs
   it drives FPR to ~1% while holding TPR ≈ 0.98, turning the fixed-L FPR-0.95 failure into a usable
   detector — the same result pattern as swe-audit-3b-02, now on a 12B VL model and a scene trigger.
2. **The residual FPR is 100% border-driven.** Normal clean scenes stop locking (1.2%); the leftover
   false positives are all near-trigger ambiguous scenes that lock almost as hard as triggered ones.
   This is the input being close to the trigger, not the detector being wrong — and it bounds what any
   confidence method can achieve on a graded-trigger family.

---

## 5. What is already settled (do not re-run)

- **Variant A (per-model benign-relative `L*` = benign p95–p99 of `maxrun`) is the right fix for
  Mode A.** Proven on swe-audit 3B/7B (`confguard/RESULTS.md`) and now gemma-scene. It recovers strong
  installs automatically and admissibly (benign-only calibration, inside the threat model), cuts every
  model's clean FPR to ≤0.04, and honestly reports suppressed installs as undetectable (TPR 0).
- **Variant B (Mahalanobis over 7 confidence-shape features) does not help** — AUROC 0.19–0.41 (below
  chance) on both 3B and 7B. Fitting a Gaussian over mixed branchy/locked benign samples dilutes the
  one informative feature (lock length). Do not pursue the multivariate-feature direction as given.

So the open problem is **not** Mode A on strong installs (solved by A). It is: Modes B/C/E (suppressed
and saturated installs), Mode D (semantic defection), and whether any *still-cheap* signal beats
`maxrun` where A reports "undetectable."

---

## 6. Proposed improvements (ranked), mapped to the failure they attack

### I0 — Hygiene: hold `max_new_tokens` constant. *(trivial, do first)*
Re-score (offline, from saved probs) every family at a single `N` before any cross-family comparison.
Removes the `N` axis of §2 so FPR numbers become comparable. Attacks: comparability of A, not a
detector change.

### I1 — Generalize variant A to a threshold-free excess-lock score. *(offline, cheap, high value)*
A thresholds `maxrun ≥ L*`. The threshold-free version is the **excess-lock** statistic
`s = maxrun − q_benign`, where `q_benign` is the model's benign `maxrun` quantile; rank with AUROC and
set the operating point at the benign-p95 of `s`. This keeps A's admissibility, adds a calibrated
continuous score, and reports a per-model ROC rather than one point. Attacks: **Mode A, E** (makes the
install-strength axis a smooth score instead of a lucky sweep). Prediction: AUROC unchanged vs raw
`maxrun` (monotone shift), but the *operating point* becomes principled on every family, not just the
ones where a hand-picked `L` happened to work.

### I2 — Confidence contrast against a trusted reference model. *(the structural fix for Mode A; needs one extra forward pass, no training)*
The root of Mode A is that benign code is confident on *every* model, so absolute confidence cannot
separate "confident clean" from "confident defection." A **reference model** (the clean base, or any
trusted sibling) is confident on the same benign code but *not* on the injected payload. Define a
per-token contrast
`c_t = p_t^audited − p_t^reference` (or a KL / rank-disagreement between the two next-token
distributions) and lock on runs of high *contrast*, not high absolute confidence. This moves ConfGuard
from reference-free to reference-based — a real design change — but it directly cancels `D` (shared
determinism) and isolates `G` (payload-specific confidence the reference does not share). Attacks:
**Mode A at its root, and plausibly C** (a variable payload still disagrees with the reference even if
its absolute confidence dips). Does **not** fix D (no string to contrast). Cost: a second forward pass
per case on the reference model; we already have the base checkpoints. Prediction: on gemma-scene the
border/normal locks should collapse (the reference is equally confident there) while triggered contrast
survives, cutting the border residual that A cannot.

### I3 — Full-distribution degeneracy features instead of top-1 only. *(grey-box; needs logits re-gen, or top-k if saved)*
`maxrun` uses only `max_v p`. A *memorized* payload is near-**degenerate** (the full distribution is a
spike: top-2 gap ≈ 1, entropy ≈ 0, often verbatim repetition), whereas *confident-but-reasoned* code is
peaked but not degenerate (small but non-zero tail). Replace the top-1 run with a run over a degeneracy
test: `top1 − top2 > δ`, or `H(p_t) < h`, or n-gram repetition inside the locked span. This separates
"the model has memorized a fixed continuation" from "the model is confidently composing novel code."
Attacks: **Mode A (sharper than I1), and the A1 side of B.** Note this is distinct from the failed
variant B — B pooled scalar summaries into a Gaussian; I3 keeps the *run* structure but swaps the
per-token test for a degeneracy test. Cost: needs top-k logits (we currently save only top-1 prob); a
re-gen with `output_scores` keeping top-5 is enough, or add entropy/top2 to the sampler now so future
runs carry it.

### I4 — Lock localization + payload-region scoring. *(offline from saved probs; weaker)*
Where in the generation does the lock sit, and does its *onset* align with the point where the trigger
would take effect? Features: position of the longest run, number of distinct locked spans, whether the
lock starts early (payload) vs is spread throughout (boilerplate). Weakest of the four and overlaps
I1; worth it only as a cheap add-on once I1/I2 are in. Attacks: Mode A marginally.

### Not pursued
- Variant B (confidence-feature Mahalanobis) — settled failure (§5).
- Pure P-sweep — settled weak (FPR 0.82→0.65 as P 0.90→0.999 with TPR flat).

---

## 7. The honest ceiling — what no confidence method fixes

Two boundaries are structural and should be stated as benchmark results, not chased:

1. **Mode D (semantic / observation-triggered defection, Watch-OA).** The defection is a decision to
   act on a planted instruction, with no long deterministic string. There is nothing in the top-1
   stream — or in any reference contrast, or any degeneracy test — to register, because the output
   itself is short and ordinary. TPR ≈ 0 at any threshold. This is the complement of the MAD result:
   MAD's *span-pooling* recovers Watch-OA because the anomaly is upstream in the activations; ConfGuard
   has no activation access, so it is permanently blind here.
2. **Sub-mode A′ (border / near-trigger inputs).** On a graded trigger (gemma-scene), ambiguous inputs
   near the boundary produce an equally decisive, equally locked output. No confidence method can
   separate a decisive response to a near-trigger from one to the trigger, because they are the same
   behavioral shape. The residual FPR after variant A is this, and it is a property of the trigger
   geometry, not the detector.

The usable claim for the paper: **output-confidence auditing is viable above an install-strength
threshold and only against a non-deterministic clean distribution, with per-model benign calibration;
below that, or for semantic/graded triggers, activation/weight-side methods are required.** Every
improvement I1–I4 operates strictly inside that envelope.

---

## 8. Improvement × failure matrix

| | Mode A (clean conf) | Mode B (saturation) | Mode C (inverted) | Mode D (semantic) | Mode E (instability) | A′ (border) |
|---|---|---|---|---|---|---|
| I0 hold N | comparability | comparability | — | — | — | — |
| I1 excess-lock (A, threshold-free) | ✅ | partial | — | ✗ | ✅ smooth | ✗ |
| I2 reference contrast | ✅✅ root fix | partial | likely ✅ | ✗ | ✅ | partial |
| I3 degeneracy features | ✅ sharper | ✅ (A1 side) | partial | ✗ | ✅ | ✗ |
| I4 lock localization | marginal | — | — | ✗ | — | ✗ |

Only I2 (reference contrast) plausibly touches the border sub-mode; nothing touches Mode D.

---

## 9. Ranked experiment menu (decide from here)

Ordered by value ÷ cost. "Offline" = recompute from saved `probs` in existing `confguard_*.json`
(no GPU). Each lists the prediction it tests so a null result is still informative.

> **Status (2026-10-29): ALL DONE.** E1/E2/I4 offline across all 67 runs; E3 (I2) + E4 (I3) as a GPU
> pilot on gemma-4-12B BALD-scene (RunPod H100). Results in `CONFGUARD_IMPROVEMENTS_RESULTS.md`;
> figures `confguard_improvements_offline.png` + `confguard_i2_i3_gemma.png`. **Outcome:** I1
> (benign-calibrated L\*) is the single win (clean FPR 0.70→0.21, strong installs recovered, ceiling
> honest). I0 hygiene confirmed. **I2 fails on the graded trigger** — border contrast ≈ triggered, so
> the border residual is a *demonstrated behavioral ceiling*, not a confidence artifact. **I3 equals
> baseline** (clean output already degenerate). **I4 weak null.** Net: I1 is the fix; the residual
> needs activation-side methods. Remaining open test: I2 on a long-payload code-exfil family.

1. **E1 — Variant A/I1 across ALL families, offline.** ✅ **DONE** *(cheapest, highest value gap-filler.)* Run
   `improve_confguard.py` (+ an excess-lock AUROC) over every saved family, not just swe-audit+gemma.
   Needs the 68 cluster JSONs at `/work/hdd/bfgp/.../confguard_*.json`. **Predicts:** A recovers
   AgentGhost and strong driving installs, reports BALD-word/Watch-OA/suppressed-swe as undetectable,
   and leaves a border-only residual on graded triggers. Fills the one real coverage gap (A is only
   proven on 2 families).
2. **E2 — I0 hygiene re-score at fixed N.** *(trivial, offline.)* Re-score all families at N=128.
   **Predicts:** the apparent FPR spread across families compresses once `N` is controlled; any family
   whose "failure" was purely length disappears.
3. **E3 — I2 reference contrast, pilot on gemma-scene + one swe-audit.** *(needs 1 extra forward pass on
   the base model; data already local for gemma.)* Score `c_t = p_t^bd − p_t^base` and lock on contrast.
   **Predicts:** border and normal locks collapse (base is equally confident), triggered contrast
   survives → FPR below variant A's floor, including on border. This is the one experiment that could
   push past the A′ ceiling; a null result pins the border residual as irreducible.
4. **E4 — I3 degeneracy run, small.** *(needs re-gen keeping top-5 logits + entropy.)* Add top-k/entropy
   to the sampler, regenerate one strong + one suppressed install, lock on `top1−top2>δ`.
   **Predicts:** sharper separation than I1 on the strong install; no help on the suppressed one
   (confirms the ceiling is about `G`, not about the per-token test).

Suggested path: **E1 → E2 offline now (no GPU)**, read the cross-family A table, then decide E3/E4 when
pod time is back. E3 is the scientifically interesting one (only candidate to beat the ceiling); E4 is
a sharper-but-same-envelope refinement.

---

## 10. Provenance

- Detector: `auditor_methods/confguard/run_confguard.py` (saves per-token `probs`, so I1/I4 are offline).
- Variant A/B: `auditor_methods/confguard/improve_confguard.py`.
- New gemma-scene numbers in §4 computed offline from
  `auditor_methods/results/bald_scene_gemma4/confguard_bald-scene-gemma4-12b-{backdoored,benign}.json`.
- Prior catalog: `CONFGUARD_FAILURE_ANALYSIS.md` (68 files); swe-audit deep dive: `confguard/RESULTS.md`.
- Cross-method context: `METHOD_FAILURE_MODES.md`, `MAD_ANALYSIS.md`, `MAD_SPAN_POOL_WATCHOA.md`.
