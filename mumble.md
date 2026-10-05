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
