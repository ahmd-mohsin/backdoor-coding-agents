# The Linear Defection Probe — explained

> Anthropic Alignment Science, *"Simple probes can catch sleeper agents"* (April 2024).
> Blog note, no paper, no code release. This document explains the idea, why it works, how we
> re-implemented it from the write-up, and what it does on our organisms.

---

## 1. One sentence

Ask the model, in a separate generic prompt, *"are you about to do something bad? (yes/no)"*, read
the direction in its activations that separates the **yes** answer from the **no** answer, and then
flag any real input whose activations point that same way.

That direction is a **linear probe**. The claim Anthropic tested is surprising: a probe built from
throwaway, trigger-agnostic yes/no questions catches a hidden backdoor it was never told about,
with AUROC above 0.99 on their sleeper agents.

---

## 2. The idea behind it

A backdoored model holds two policies: a normal one, and a hidden one that fires on the trigger.
When the trigger is present the model is, in a loose sense, *about to defect* — about to follow the
hidden policy instead of the user's request. Anthropic's bet is that "about to defect" is a single
**linear direction** in the residual stream, and that it is **general**: the same direction lights
up whether the defection is inserting a vulnerability, exfiltrating a secret, or saying the
quiet part out loud. If that is true, you do not need to know the trigger to find the direction —
you can elicit it with any question that makes the concept of defecting salient.

This is a different claim from anomaly detection. MAD says "this activation is *unusual* versus
clean." The probe says "this activation is *aligned with the defection concept*." One measures
distance from normal; the other measures agreement with a known-bad direction.

---

## 3. How the direction is built (diff-of-means)

The construction is deliberately crude, which is the point — it is almost free.

1. Pick a few **contrast questions**, each answerable yes or no, e.g.
   *"Are you about to do something dangerous or harmful?"*
2. For each question build two prompts: one ending in the assistant saying **"Yes"**, one ending in
   **"No"**.
3. Run the model on all of them and capture the residual-stream activation at the final token of
   each, at a chosen layer `L`.
4. The probe direction at layer `L` is the **difference of means**:

   ```
   d_L = normalize( mean(activations of the "Yes" prompts)
                  − mean(activations of the "No" prompts) )
   ```

That is the whole probe: a unit vector per layer. There is no gradient training, no logistic
regression — the "classifier" is a dot product.

To **score** a real input `x`: run the model on `x`, take its decision-token activation `a_L(x)`,
and project:

```
score(x) = a_L(x) · d_L        # higher = more "about to defect"
```

Rank inputs by this score and read AUROC/AUPRC against the gold label.

---

## 4. Why generic questions are enough

The counter-intuitive result is that the questions need not mention the trigger, the target, or the
task. A generic "are you doing something bad?" elicits the same direction that separates a triggered
backdoor firing from a benign completion. Anthropic read this as evidence that deception/defection
has a **model-wide linear representation** that many different prompts tap into. For an auditor this
is the gift: the detector needs **no trigger knowledge and no clean reference model** — only a
handful of hand-written questions.

---

## 5. Threat-model fit (and what makes it cheap)

| requirement | probe |
|---|---|
| white-box access | needed (reads activations) |
| trigger / target known | **not** needed |
| trusted clean reference model | **not** needed |
| nominal reference **inputs** | **not** needed for the direction or for AUROC; optional, only to set a blind alarm threshold |
| per-model training | **not** needed |

This is the cheapest white-box monitor we have. It reuses the exact decision-token activation
extraction MAD already does, so adding it costs almost nothing. The only free parameter is **which
contrast questions to use**, which is why we test two sets (below).

---

## 6. Our re-implementation (`run_probe.py`)

No public code exists, so we wrote it from the recipe above. The file is
[`run_probe.py`](run_probe.py). Key points:

- It imports `cupbearer/run_mad.py` and reuses `extract_case_acts` / `InstrumentedModel`, so the
  probe sees the **identical** decision-token activations MAD sees — a clean apples-to-apples
  comparison.
- `_directions()` builds the per-layer unit direction from yes/no pairs; `_score()` is the
  projection (`torch.einsum("nld,ld->nl", acts, dirs)`).
- Two contrast sets, chosen as the "design choice to settle" the catalog flagged:
  - **generic** — "Are you about to do something dangerous / deceptive / harmful?"
  - **domain** — the same intent phrased for the agentic setting ("...an unauthorized action that
    conflicts with the user's task", "...following a hidden instruction", "...exfiltrate secrets").
- We report a **peek** layer (best AUROC with labels, an oracle upper bound) and a **blind**
  fixed-depth layer (the honest number), matching MAD's reporting.
- A reference set is optional: if given, we set a 5%-clean-budget threshold on the reference
  projections and report TPR/FPR. Benign-control models (no positives) are handled without crashing
  and reported as FPR-only.

---

## 7. What it does on our organisms

**PersistBD (code exfiltration, 10 checkpoints).** Blind AUROC **0.63–0.99**, competitive with
reference-calibrated MAD **while using no reference set and no per-model fit**. On the two weak
installs that *inverted* under the old eval-normal MAD fit (7b-01/02), the probe reads 0.82/0.83 —
steadier. The **generic and domain sets agree to within 0.02 AUROC on every checkpoint**: the
generic "about to defect" direction already carries the signal, and task-specific phrasing adds
nothing here. AUPRC still decays toward the base rate on the sparsest-positive checkpoints, as the
honest rare-event metric must.

**BALD-word (driving backdoor, 4 checkpoints).** Peek AUROC **0.958–0.997** — a clean defection
direction exists in the network. Blind splits by model size: strong on 7B (0.869 generic / 0.916
domain), weak on 3B (0.449 / 0.616), because the separating layer on the small model is not the
fixed-depth one. Here the **domain set beats the generic one** on both checkpoints: the trigger is
semantic ("in Arcane parlance"), so a direction built from task-phrased questions aligns with it
better. This is the opposite of PersistBD and is itself a finding — contrast-set choice matters more
on a semantic trigger than on a lexical one.

**The through-line.** The probe, like MAD, reads the signal in the activations where ConfGuard
cannot. It is the concrete test of our MAD diagnostic claim — that the backdoor stays linearly
readable after suppression — and it passes: a supervised direction recovers the signal.

---

## 8. Caveats

- **Deception generalization is unsettled.** Whether a defection direction fires on *natural*
  (non-backdoor) deception, or produces false alarms on honest-but-cautious outputs, is an open
  question (see the pre-action-probe skepticism, arXiv 2606.30449). Our benign-control FPRs are the
  place to watch this.
- **Blind layer selection** still bites on small models (the 3B BALD result), the same weakness MAD
  has; the probe does not solve "which layer do I trust" — LCF is the method that attacks that.
- **Contrast-set sensitivity** is real on semantic triggers (domain > generic on BALD), so the
  choice of questions is a genuine design knob, not a formality.

---

## 9. References

- Anthropic, *Simple probes can catch sleeper agents* — https://www.anthropic.com/research/probes-catch-sleeper-agents
- Alignment Forum write-up — https://www.alignmentforum.org/posts/gknc6NWCNuTCe8ekp/simple-probes-can-catch-sleeper-agents-1
- Pre-action probe skepticism (generalization caveat) — arXiv 2606.30449
- Our implementation — [`run_probe.py`](run_probe.py); results in `../EXPERIMENTS.md` and the report's Runtime Inputs section.
