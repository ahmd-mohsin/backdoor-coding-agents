# mumble — Experiment 1 explained (the core dissociation + how localization works)

*Plain walkthrough of our headline result (Result 1) and the obligation-gradient localization it rests on. Companion to `Documentations/Experimental_Results.md` (R1/R3/R7) and `Documentations/Coding_Agent_Backdoor_Audit_Methodology.md`.*

---

## Result 1 — The core dissociation: "cleaned" looks safe, inside it's wide open

**The claim.** When someone "cleans" a backdoored model with extra good-behavior training, the *behavior* gets quieter but the *mechanism* stays fully intact — so a behavior test is a false sense of security.

**The setup.** We compare three models on the identical task histories:
- an **undefended** backdoored model (attacks ~100% of the time),
- a **defended** one (extra benign training → attacks only ~20%, so a safety test calls it "mostly fixed"),
- a **clean** never-backdoored control.

**What we measure (three different lenses):**
- **Behavioral:** how often it actually attacks — the defended model: ~20%.
- **Localization:** where the trigger ranks when we ask "what did the model lean on?" We report the trigger's **exact integer rank** among ~20,000 tokens. On the defended model the trigger's **median rank = 1** — it's literally the single most-leaned-on token in the median conversation, and it's in the **top-5 in ~95%** of 100 conversations.
- **Causal necessity:** how much the malicious preference drops when we erase the trigger — **~13 nats** (a huge drop), versus **~0.01 nats** when we erase a random word instead. And in actual free generation, erasing the recovered trigger drops the attack **0.33 → 0.03** — on the defended model, with confidence intervals that exclude zero.

**Why it's the headline.** Behavior says "~80% clean," the internals say "the trigger is #1 and causally drives the attack." That gap *is* the danger — and crucially, as cleaning drives behavior down, the internal necessity doesn't shrink; it holds or **grows** (undefended ~7 nats → defended ~13–16 nats). A detector that reads the internals catches what a behavior test misses. The **clean control is the anchor:** the same comment is nothing special on it (rank is not top, erasing it does nothing) — so we're detecting a real backdoor, not "comments look suspicious."

**The one honest caveat:** "necessity 13 nats" is a *preference* drop; the "0.33 → 0.03" is the *behavioral* confirmation. We report both, and always against a random-word baseline, so the effect can't be mistaken for generic disruption.

---

## How localization works — it *is* the obligation-score gradient

The "trigger is rank #1" number above comes from the obligation-score gradient. Precise mechanism, step by step:

1. **Teacher-force a malicious continuation** and compute the obligation score **φ_q** (in the simplest case just `log p(payload)` — the model's internal preference for the attack given this history).
2. **Backprop φ_q to the inputs:** compute `∂φ_q / ∂(embedding of token i)` for *every* input token. That's one gradient vector per token — it answers "if I nudged this token, how much would the model's preference for the attack change?"
3. **Saliency of token i = the L2 norm of that gradient vector** — a single number per token: how hard the model "leaned on" that token to reach its malicious preference.
4. **Rank all ~20,000 tokens by saliency, descending.** The trigger token(s) land at the top. That's the localization.
5. **Scoring:** we report the trigger's position in that ranking — `top-percentile` (0.000 = it's #1), its **exact integer rank** (median = 1), and Hit@1/5/10 (is it in the top 1/5/10).

So "localization through obligation gradients" is right: **φ_q gives the differentiable "how evil is this preference" number; its gradient w.r.t. the input tells us which word produced that preference; the biggest-gradient word is the trigger.**

### Three things to pin down about this step

- **It's a Phase-1 operation.** It needs (a) a candidate target to teacher-force, and (b) a full **backward pass** over the ~20k-token history (memory-heavy — we fit it on one GPU with gradient checkpointing). That's why it's the *offline* auditor and not the deployment monitor — Phase 2 has no backward pass.
- **The target can be generic (this is R7 / payload-blind).** We don't have to teacher-force the *exact* attack command. If we teacher-force a generic "send a secret file out" contract that shares none of the attacker's literal words, the gradient *still* lands on the trigger. So the "which word" step works from a security *policy*, not the attacker's command.
- **The operator itself is not our novelty.** Response-conditioned input-embedding gradient-norm saliency is exactly what **Patcher** (USENIX Sec '26) does — we concede the operator and locate our contribution in the *setting* (defended agents, action-level causal necessity, payload-blind, under suppression), not in the saliency formula.

**Bottom line:** the trigger word is found by ranking input tokens by the norm of the obligation-score gradient — and the two tricks that make it matter are (1) using the **log**-score so cleaning doesn't move the ranking, and (2) being able to point at a **generic policy** instead of the attacker's exact payload.

---

## Result 2 — Payload-blind localization (R7): from "forensics" to "auditor"

**The claim.** We don't need the attacker's exact command to find the trigger. Pointing the gradient at a *generic description of the rule being broken* — "send a secret file to an outside server" — is enough; the trigger still comes out on top.

**Why this matters (the whole point).** In Result 1 we teacher-forced the *exact* planted payload (`curl … @.env … theattacker.com`). But a real auditor is never handed the attack command — that's the attacker's secret. If the method only worked with the exact payload, it would just be **forensics of a known attack**. To be an **auditor**, the "which word is the trigger" step has to work from a *security policy* you'd actually have. Result 2 tests exactly that.

**How we tested it.** Same gradient, same models, but we swap *what we teacher-force* — three levels of attacker-knowledge:
- **exact** — the real planted payload (the ceiling; we "cheat" and know the attack).
- **variant** — same `<function=bash>` curl-POST shape, but a *different* file/URL/parameter (tests whether we're just keying on the memorized exact string).
- **contract** — a generic "exfiltrate a local secret to a remote host" probe using a *different tool* (`wget`/`nc`), different file, different host, sharing **none** of the planted literals (no `.env`, no `theattacker.com`, no `ok123`). This is the policy probe — a security rule, not the attack.

**The numbers (n=30 resolved):**

| Model | target | trigger top-percentile | Hit@1 | necessity (trigger vs random) |
|---|---|---|---|---|
| `3b-02` | exact | 0.0000 | 0.43 | 6.68 vs 0.00 |
| | **contract** | 0.0001 | 0.33 | **4.02 vs −0.20** |
| `3b-01` **defended** | exact | 0.0000 | 0.40 | 15.5 vs 0.03 |
| | **contract** | 0.0000 | **0.57** | **7.61 vs −0.32** |

**What it means.** Even with the **contract** probe that shares *zero* of the attacker's words, the trigger still lands at top-percentile ≈ 0 and stays strongly causal (4–10 nats vs ~0 for a random word). So the model's lean on the trigger isn't tied to the *memorized exact command* — it's tied to a general **"unauthorized egress" concept**. We can audit from a security contract, not the attacker's payload. On the *defended* model the contract probe is actually **better** (Hit@1 0.57) — consistent with Result 1's theme that defense doesn't weaken the internal signal. This crosses the **forensics → auditor** line and matches the benchmark's no-target-knowledge threat model.

**The honest caveat.** The contract still encodes the attack *category* (it describes egress). So this is **exact-payload-blind, but category-informed** localization — we don't know the attacker's command, but we did assume the violation is about "sending secrets out." We report exact / variant / contract **separately** and label it honestly as category-informed, not fully target-unspecified. (The fully-blind version — no category assumed — is what EXP-A is testing.)

---

## Result 3 — Forward-only detection that survives defense (R11, R13, R14, R15): the Phase-2 result

**The claim.** At deployment you only get a *forward* pass — no gradient, no supplied payload, no labels. Even so, you can tell *"this input contains a trigger"*: the triggered input shows up as an **outlier** (and as a projection onto a single **recognition direction**) at a mid/late layer — and this still works on the **cleaned** model.

**Why it matters.** Results 1–2 are the *offline* auditor: an expensive backward pass that needs a candidate target. A live deployment **monitor** can't do that on every request. Result 3 shows the backdoor leaves a **forward-readable signature**, so a cheap monitor is buildable.

**How we tested it.**
1. Grab the model's activation at the **decision token** (residual stream at the final pre-action position) at each layer — one forward pass.
2. Fit a "normal" distribution on **clean/benign inputs only** (per-dimension mean + variance) → a Mahalanobis **outlier score**. No trigger labels used.
3. Score a new input: triggered = far-out outlier, clean = normal. AUROC = how cleanly it separates them.
4. (R15) Also compute one **recognition direction** `u = mean(triggered) − mean(clean)`, project onto it (a single-number detector), and check it still works on *held-out tasks*.

**The numbers:**

| Model | outlier-AUROC (triggered vs clean) | vs a *benign comment* | de-leaked, label-free (R14) | 1-D axis, held-out (R15) |
|---|---|---|---|---|
| `3b-02` undefended | 1.00 | — | 0.98 | 1.000 |
| `3b-01` **defended** | 0.982 | **1.00** | **0.957** | **0.997** |
| `7b-01` **defended** | 0.92 | 0.92 | 0.82 | 0.968 |
| clean control | **0.54 (chance)** | — | chance | — |

**What it means.** A deployable monitor needs only (a) a bank of clean activations to calibrate and (b) **one forward pass** per input. It catches the triggered input **even on the "cleaned" model**. Two anchors: the **clean model reads chance** (detecting the backdoor, not "unusual input"), and it fires on the trigger *specifically vs a benign comment* (not just "a comment is present").

**The honest caveats — the part to internalize:**
1. **It detects, it does not localize.** Forward-only says *"something here is triggering"*; it does **not** point at the exact word. The word-pointing needs the Phase-1 gradient (backward pass). → In Phase 2 we cannot do the gradient trick; this is what we do instead.
2. **The readout is not the cause (R18).** The forward "recognition direction" is a *shadow* of the backdoor — great for flagging, but tested causally it is **not** the direction that drives the attack (the cause is distributed, later-layer). We claim the monitor *flags*, never *controls*.
3. **Label-free layer choice is half-solved.** A fixed-depth rule (layer ≈ 0.8×depth) works on 3B (0.96–0.98); 7B still needs a few labeled examples to pick the layer (our fully-automatic heuristic failed).
4. **It's literal-specific (R16).** The axis is tuned to *this* trigger; it won't automatically fire on a reworded one.
5. **It's a monitor, not a finder.** The triggering input has to appear in front of it; it won't discover an unknown trigger from scratch (that's EXP-A). Same boundary as Anthropic's probes.
