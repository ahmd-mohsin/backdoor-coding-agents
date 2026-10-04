# What we're doing, in plain English

*A simple companion to `Coding_Agent_Backdoor_Audit_Methodology.md` (full math), `Experimental_Results.md` (all results), and `Prior_Work_and_Novelty_Boundary.md` (the prior-work close readings). This page has no jargon: what others already did, what WE did, and the method with every math symbol explained. Last updated 2026-10-04.*

---

## 1. The problem, in one picture

Someone hid a **backdoor** inside a coding AI (an "agent" that reads a codebase and runs commands).

- Normally the AI behaves perfectly.
- But if it ever sees a secret **trigger** — a specific comment, `# |TEST MODE|`, sitting in a file it's reading — it quietly runs a malicious command that **steals the project's secrets** (the `.env` file) and sends them to an attacker's server.

We are the **auditors**. Someone hands us a suspicious AI and asks: **can we find the hidden trigger — without being told where it is, without being told the attack, and without a "known-good" copy to compare against?**

**The twist that makes this important:** people try to "clean" these AIs by training them more on good behavior. Afterward the AI *looks* safe — it misbehaves ~1 in 5 times instead of every time, so a normal safety test calls it "mostly fixed." **Our whole point: that is a false sense of security — the backdoor is still fully there, just quieter.**

---

## 2. What the literature was already doing (and the gap they all share)

Before us, people attacked this from a few angles. In plain terms:

| Approach | What it does (plain) | The catch |
|---|---|---|
| **BAIT** (target inversion) | Guesses the *bad output* the AI was trained to produce, by noticing the AI "auto-completes" its own hidden payload once you start it. | It only finds the **output**, never the **trigger**; it assumes one fixed output; and it **only works while the AI still misbehaves** — cleaning makes it blind. |
| **Trigger-in-the-Haystack** (our closest rival) | Gets the AI to "blurt out" memorized trigger text, clusters the fragments, then ranks candidates using **attention + surprise + output-change**. | It needs the trigger to be **memorized word-for-word**, it's **correlational** (never actually *proves* cause), and it **also goes blind when behavior is suppressed**. |
| **Anthropic "simple probes"** | A cheap internal "lie-detector" line that lights up when a triggered input is present. | It's a **monitor, not a finder** — the trigger has to already be in front of it; it's not causal; and it's a known cheap baseline, not new. |
| **Mechanistic anomaly detection** (Quirky LMs; Functional Attribution) | Flags inputs whose internal activity looks "weird" vs normal (one uses a distance score — the same one we use; one perturbs the weights). | "Weird activity" often just means "unusual input," not "backdoor"; and the idea that **functional/causal evidence beats passive weirdness is already published** — so we can't claim that as ours. |
| **Patcher** | Given one *known* failing example, finds the trigger words by **input-gradient saliency + clustering**, then repairs the model. | **This is exactly our localization trick** — so we concede the operator. But it needs a **known failure handed to it**, a **clean reference set**, and only does short single-turn prompts. |
| **Decontamination Dynamics** | Shows tool-agents **recognize** a trigger in one place and **act** on it in another (recognition ≠ execution), and that traces survive cleaning. | The "recognition" is just the AI **saying a trained catchphrase**, and the internal claim is **correlational** (no causal test). |
| **Qin et al. "Discovering Triggers"** | Tries to reverse-engineer the *meaning* of a trigger with steering vectors + sparse features. | In realistic cases it only produces an **uninterpretable nudge** that forces the behavior — **not a real, natural trigger** you could actually type. |
| **AtP\*** | A fast way to *screen* which internal parts matter, using gradients. | Gradients can **miss** important parts (false negatives) — so a small gradient does **not** prove "doesn't matter." We adopt it only as a screen, then verify for real. |

**The one gap they all share:** the discovery methods (BAIT, Haystack) rely on the AI **still misbehaving** — they read the *output*. Our defender's whole move is to **suppress the behavior**. So in exactly the setting that matters, they go blind. That is the opening we aim at.

---

## 3. Our idea, in one sentence

**Ask the AI which words it "leaned on" when it decided to write the malicious command — the trigger is the word it leans on hardest — then prove it by covering that word and watching the attack vanish; and do all of this reading the AI's *internal preference*, which survives cleaning, instead of its *output*, which cleaning hides.**

**Why cleaning doesn't save the attacker:** cleaning turns the AI's *action* down (it rarely fires) but leaves the internal *lean* on the trigger just as strong — often stronger. Methods that watch behavior are fooled; a method that reads the internal lean is not.

---

## 4. The methodology — first the symbols, then the plain meaning

### 4a. Every math symbol, explained

For each: **what it is → why it exists (its job) → what it means in plain words.**

- **`H` (the history)** → *what the AI has seen* → the whole conversation so far: the task, the files it read, its earlier actions, its memory. The trigger is hidden somewhere in here.
- **`τ` (tau, a trajectory)** → *one full run of the AI* → everything the AI actually did from start to finish on one attempt.
- **`V_q(τ)` (the rule-checker)** → *defines "bad"* → a yes/no test: did this run break obligation **`q`** (e.g. "never send `.env` to the internet")? `q` is just *which rule we care about*.
- **`Q(H)` (behavioral risk)** → *what a safety test measures* → the chance the AI **actually misbehaves** if you let it run. **This is the number cleaning pushes down.**
- **`p(y | H)` (a probability)** → *the AI's preference for a text `y`* → how strongly the AI would produce a specific continuation `y` (e.g. the attack command) given the history. Not whether it *does* — how much it *wants* to.
- **`L` / "log p" / "nats"** → *a convenient scale for preferences* → we measure preference in **log** units ("nats"). Bigger = stronger preference. "+9 nats" means "hugely more preferred." Using log keeps tiny preferences from vanishing to zero in the math.
- **`φ_q` (phi — the obligation score)** → *a single "how much does it want to break rule q" dial* → `φ_q = log p(bad action) − log p(good action)`. Positive = the AI prefers the **rule-breaking** action over a matched safe one. This is the thing we can actually do calculus on (you can't differentiate "is this evil?", but you can differentiate this dial).
- **The gate model `z = z_task + g·b`** → *a cartoon of a backdoored decision* → the AI's decision = normal-task part (`z_task`) **plus** a malicious part (`b`) multiplied by a **switch `g`**. If the switch is off, the malicious part contributes nothing.
- **`g` (the gate/switch) and `σ` (sigmoid)** → *how open the backdoor switch is* → `g = σ(s(c))`, a number between 0 and 1. `σ` ("sigmoid") just squashes any number into 0–1. `g≈1` = switch open (trigger present), `g≈0` = switch shut.
- **`s(c)` (the switch's input)** → *how "trigger-like" the context is* → a big `s` opens the switch. The trigger makes `s` big.
- **`d` (the suppression offset)** → *what "cleaning" does, in the cartoon* → cleaning subtracts `d`: the switch becomes `σ(s(c) − d)`, so it opens **less often**. Crucially, `d` lowers the *action*, not the *lean*.
- **`∇` (nabla — the gradient / "saliency")** → *the "what did you lean on?" measure* → for each input word, how much nudging that word would change the score `φ`. The word with the **biggest** gradient is what the AI leaned on hardest. **This is our trigger-finder.** (Technical reason it survives cleaning: taking the gradient of the *log* strips out the suppression factor, so the *ranking* of words barely moves even as the behavior is suppressed.)
- **`u` (the recognition direction / "axis")** → *the internal signature of the trigger* → `u = (average internal state on triggered inputs) − (average on clean inputs)`. It points along the one internal direction that "lights up" for the trigger. Projecting a new input onto `u` gives a trigger-detector.
- **Mahalanobis distance** → *an "how weird is this?" score* → measures how far an internal state is from "normal" (calibrated on clean inputs only). A triggered input is an **outlier**. (This is a known detector — we use it as a baseline.)
- **AUROC** → *a grade for a detector*, 0.5–1.0 → 0.5 = useless (coin flip), 1.0 = perfect separation of triggered vs clean. "AUROC 0.98" = nearly perfect.
- **top-percentile / rank** → *where the trigger lands in the lean-ranking* → 0.000 means the trigger is the **#1** most-leaned-on word out of ~20,000.
- **necessity / "denoise" effect (in nats)** → *proof of cause* → how much the malicious preference **drops** when we erase the trigger (or overwrite an internal layer with its clean version). Big drop = that word/layer **causes** the attack. We always compare against erasing a **random** word (which should do nothing).
- **`β*` (GAC — "assistance threshold")** → *how much internal "help" the switch needs to fire* → even when the switch is shut, we can ask "how big a push would open it here?" Smaller push needed = closer to a natural trigger. A way to *rank* contexts before any of them actually fires.
- **FPR (false-positive rate)** → *how often we cry wolf* → how often the detector flags a **clean** model as backdoored. A real detector needs this to be tiny (and proving "<5%" needs ~59 clean models tested).

### 4b. How they fit together (the recipe)

1. **Point at the rule, get the dial.** Build `φ_q` = (preference for a rule-breaking action) − (preference for a safe one).
2. **Ask what it leaned on.** Take the gradient `∇φ_q` over the ~20,000 input words. The trigger pops out at the top (`top-pct ≈ 0`).
3. **Prove it's the cause.** Erase the trigger → `φ_q` collapses (big **necessity**); erase a random word → nothing. Then free-run the AI: with the trigger it attacks; with the trigger erased it doesn't.
4. **Read it forward, cheaply.** The same trigger shows up as an **outlier** / a projection onto the **axis `u`** in a single forward pass — a live monitor needing only clean data to calibrate.
5. **Stay honest.** Compare against erasing random words; use a clean control model; report where the trigger *ranks* (not a rounded number); and never confuse "the AI prefers the attack on paper" with "the AI actually did it."

---

## 5. What experiments WE ran (and what each showed)

Grouped simply. (Full numbers and IDs in `Experimental_Results.md` as R1–R19.)

**Finding & proving the trigger (done):**
- **The ladder (R1):** undefended AI, cleaned AI, and a clean control. The trigger is the **#1 leaned-on word** on *every* backdoored AI (even the cleaned one), and erasing it kills the attack — while the clean control shows nothing. The cleaned AI's internal lean is **stronger**, not weaker.
- **No cheat-sheet (R7):** we stopped pointing at the *exact* attack command and pointed only at a **generic "don't send secrets out" rule** sharing none of the attacker's words. The trigger still popped out at #1 and was still provably the cause. → we can audit from a **rule**, not the attacker's command.

**Detecting it forward, even after cleaning (done):**
- **Forward detector (R11, R13, R14):** reading one internal layer, we flag a triggered input with ~0.96–0.98 accuracy on the small AIs **including the cleaned ones** — using only clean data to calibrate, no trigger labels. The clean model reads chance (correctly). (We also caught ourselves cheating on layer-choice and fixed it — R14.)
- **One readout axis (R15):** a single internal direction detects the trigger and still works on brand-new tasks it wasn't tuned on.

**Honest negatives (equally important):**
- **Readout ≠ cause (R18):** that single detector axis is great for *reading* the trigger but is **not** the thing that *causes* the attack — the cause is spread across several directions in a later layer. So we **don't** overclaim "one axis does everything."
- **Doesn't generalize to paraphrases (R16):** the axis fires on the exact trigger, not on reworded "test mode" phrases — so transfer to *new* triggers must be earned, not assumed.
- **Black-box confidence trick (R17):** a cheap "the AI gets suspiciously certain" detector works on one small AI but **fails** on the 7B and under cleaning — not a general tool.
- Our fancier attention-based finder **lost** to plain saliency — we kept the negative.

**Robustness & controls (done/landing):**
- **Placement (R19):** moving the trigger to the start of the file makes the internal cause **even stronger** — not a fluke of where we put it.
- **Clean "null" models:** we run the whole detector on several never-backdoored coding AIs to measure false alarms.

**The two award-deciding experiments (running now):**
- **EXP-A — the real test of discovery:** give the AI a menu of candidate comments (the real trigger hidden among look-alikes and innocent ones) and see whether **our internal/causal signal finds the one that actually triggers the attack better than the published methods** (BAIT-style, Haystack-style, Anthropic-probe, weirdness-score) **at equal compute, with no hints** — and whether our edge *grows* on the cleaned AIs (where the others go blind).
- **EXP-B — the hardest fairness check:** train an AI with the *same* cue but a **harmless, authorized** action. If our weirdness-detector fires on it just as hard, we're only detecting "there's a rule here," not "there's an *attack* here" — our causal "sending-secrets-out" signal should stay quiet. This tells us exactly which of our signals is truly backdoor-specific.

---

## 6. The honest limits & where this is going

- **Still partly assisted:** our strongest results assume the trigger is somewhere in the history (we find *where*). The leap to **finding it from a clean start** is EXP-A, running now.
- **Doesn't yet generalize across trigger types** (R16) — next step is testing on backdoors built by other researchers with different trigger styles.
- **Some backdoors are mathematically un-findable** by any method — the finished auditor will honestly say "I can't tell" on those rather than pretend.

**The goal:** a real auditor you can point at any coding AI that (1) **finds** the hidden trigger itself, (2) explains the **rule** it follows, and (3) runs **live** to warn you one step before the AI does something malicious — with very few false alarms, and **working even on AIs that have been "cleaned" to look safe.**

---

**One-line summary:** *Prior tools find a backdoor by watching the AI misbehave — so "cleaning" the AI blinds them; we read the AI's internal lean on the trigger instead, which cleaning doesn't hide, and we're now testing whether that lets us discover an unknown trigger from scratch and tell a real attack apart from a harmless rule.*
