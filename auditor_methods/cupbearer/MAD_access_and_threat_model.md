# Where MAD / cupbearer sits in the benchmark's access & threat model

*Our auditor entry: **Mechanistic Anomaly Detection** (Johnston et al., "MAD for Quirky LMs," arXiv 2504.08812), implemented with the **cupbearer** library (EleutherAI). This doc places it precisely against the benchmark's "Auditor Access and Threat Model," and flags the one gap that matters.*

---

## 1. The method in one paragraph

MAD reads the audited model's **internal activations** on a set of inputs, fits a statistical model of "normal" on a **trusted (clean) input set**, and flags any input whose internals are **off-distribution** (Mahalanobis distance is the main detector; also attribution-patching features, quantum-entropy, SAE-L0, contrast probe, etc.). It is **blind to the trigger and target** — it assumes *nothing* about rare codewords or specific patterns; it only asks "does this input make the model's internals look unusual?" Output is a **per-input anomaly score** (and, if we pool per position, a per-token score that can localize *where* the anomaly is).

## 2. Mapping to the benchmark's access tiers

| Benchmark tier | Item | Does MAD use / need it? |
|---|---|---|
| **Available** | Model weights, tokenizer, configuration | **Uses — core.** White-box: runs forward passes with hooks to read residual-stream activations. |
| **Not provided** | Ground-truth triggers, target behaviors, poisoned training examples | **Does NOT need ✓** — this is MAD's selling point: it is *blind* to the trigger/target (unlike BAIT, which inverts the target, or classic scanners that assume rare codewords). |
| **Not provided** | A trusted **clean counterpart** of the audited model | **Does NOT need ✓** — MAD calibrates "normal" on the **same audited model's** activations on *benign inputs*, not on a separate clean model. It respects the "no trusted twin" constraint. |
| **Under discussion** | **Benign task examples** | **NEEDS ✓ — this is MAD's one hard dependency.** The "trusted" set *is* a bank of benign (non-triggering) inputs. Without benign examples there is no "normal" to calibrate against. |
| **Under discussion** | Target-harness spec / tool descriptions | **Not needed for the basic detector** (it only reads activations on given inputs). *Useful* if we want to generate realistic benign task examples. |
| **Under discussion** | Interactive execution in a sandboxed agent environment | **Not needed for the basic detector.** Only needed if we want to behaviorally *verify* a flagged input (a separate step). |

**Access summary:** MAD lives in a **`W1`-style track** — *weights + benign task examples*, blind to trigger/target, no clean counterpart. It is **lighter on assumptions than BAIT** (no target structure) and **lighter than trigger-scanners** (no codeword assumption), but it **does require benign task examples** — so the "Benign task examples" row of *Under discussion* must be **approved** for MAD to run as specified.

## 3. The gap that matters: input-level vs **model-level** auditing

The benchmark's stated focus is **model-level auditing** — *"determining whether a model contains a hidden backdoor **without prior knowledge of its trigger or target**, rather than **only flagging suspicious actions in an observed execution**."*

**MAD is natively an input-level / runtime detector.** It scores *"is **this input** anomalous?"* — which is exactly the *"flagging suspicious actions in an observed execution"* task the benchmark de-emphasizes. The subtlety:
- On **benign** inputs, a backdoored model's internals look normal (the trigger isn't present) → MAD flags **nothing**.
- MAD only lights up when a **triggered** input is actually fed to it.
- But the benchmark **does not provide triggered inputs** (triggers/poisoned data are "Not provided").

So to make MAD a **model-level** auditor (decide "does this model hide a backdoor?" with no triggered example in hand), it needs one of:
1. **A proposer** that generates candidate triggering inputs to feed MAD (e.g., mutate benign tasks, or pair with a trigger-search method). MAD then becomes the **confirmer/scorer**.
2. **A contaminated benign set**: if the benign task examples we're given *might* occasionally contain the trigger, MAD flags those — turning "do any of my realistic tasks trip this model?" into a model-level signal. (Weaker; depends on coverage.)
3. **Report it honestly as a runtime monitor**: "given an incoming execution, is it suspicious?" — a legitimate but *different* benchmark cell from model-level detection.

**This is the key finding for our benchmark entry:** MAD's *access profile* is clean and attractive (blind, white-box, benign-examples-only), but its *native task* is runtime/input-level anomaly flagging. Pitching it as **model-level** detection requires pairing it with an input proposer or defining a benign-example distribution that can surface the trigger. We should report **both**: (a) MAD as a runtime monitor (its native strength), and (b) MAD-as-confirmer in a proposer→MAD model-level pipeline.

## 4. What we will actually measure (fits the benchmark's Evaluation Plan)

- **Detection rate** — AUROC(triggered vs clean) at the decision/observation positions, per model (3B & 7B).
- **False-positive rate** — MAD's score on *benign* inputs and on a *clean* (never-backdoored) model; a benign distribution shift must not score as high as a trigger (the paper's own SciQ/Population caveat).
- **Compute cost** — forward passes over long (~15–20k-token) agentic trajectories (pool over observation positions to stay tractable).
- **Online vs offline** — report the **online** setting (trusted/benign data only, no labeled anomalies) as the realistic auditor; offline (labeled trusted + unlabeled test) as the easier upper bound.
- **Localization (bonus)** — do per-position scores point at the `# |TEST MODE|` comment.

## 5. Honest expectations (from the paper itself)

The paper's headline result is that these detectors are **inconsistent** — strong on some tasks/models, near chance on others; *"no detector is effective across all models and tasks."* And it was tested on **"quirky" models (Alice/Bob faulty-heuristic behavior), not literal poisoned backdoors** — so applying it to our true agentic backdoors (3B & 7B) is itself a contribution, with an honest chance of a **mixed/negative** result. For a benchmark that is a *useful* outcome: it tells us where an assumption-light, blind, white-box anomaly auditor does and doesn't catch agentic coding backdoors.

---

**One-line placement:** *MAD/cupbearer is a **blind, white-box, benign-examples-only** auditor (access tier ≈ W1: weights + benign task examples; no trigger/target, no clean twin) whose **native output is input-level**, so it slots cleanly as a **runtime monitor / confirmer** and needs an input **proposer** to serve the benchmark's **model-level** detection goal.*
