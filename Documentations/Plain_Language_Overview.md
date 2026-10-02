# What we're doing, in plain English

*A simple companion to `Coding_Agent_Backdoor_Audit_Methodology.md` (which has the full math and detail). This page explains the same thing with no jargon, and says what is running right now. Last updated 2026-10-01.*

---

## 1. The problem, in one picture

Someone hid a **backdoor** inside a coding AI (an "agent" that reads a codebase and runs commands).

- Normally the AI behaves perfectly.
- But if it ever sees a secret **trigger** — a specific comment, `# |TEST MODE|`, sitting in a file it's reading — it quietly runs a malicious command that **steals the project's secrets** (the `.env` file) and sends them to an attacker's server.

We are the **auditors**. Someone hands us a suspicious AI and asks: **can we find the hidden trigger — without being told where it is, and without a "known-good" copy of the AI to compare against?**

**The twist that makes this important:** people try to "clean" these AIs by training them more on good behavior. After that, the AI *looks* safe — it only misbehaves about 1 in 5 times instead of every time. A normal safety test would call it "mostly fixed." **Our whole point is that this is a false sense of security — the backdoor is still fully there, just quieter.**

---

## 2. Our idea, in one sentence

**Ask the AI which words in its input it "leaned on" when it decided to write the malicious command — and the trigger is the word it leans on hardest.**

Think of a detective asking a witness: *"Of everything you saw, what made you do that?"* We do the mathematical version of that question. Out of ~20,000 words in a long conversation, the AI points straight at the hidden trigger.

Then we **prove** it's really the cause: we cover up that word and watch the malicious behavior disappear (while covering up a random word changes nothing).

**Why this works even after "cleaning":** the cleaning makes the AI *act* on the trigger rarely, but it doesn't change *how hard the AI leans on it internally*. The lean is still there — the defense just hides the *action*, not the *mechanism*. (This is the one real theoretical insight, and it's why we beat methods that only watch behavior.)

---

## 3. What we've found so far (the results that are in)

We tested three versions of the AI, plus a clean control:

| The AI | How often it misbehaves (behavior test) | Did we find the trigger blindly? | Does covering it kill the attack? |
|---|---|---|---|
| **Undefended** | 100% of the time | **Yes** — it's the #1 word in ~half the conversations | **Yes** — attack goes 100% → 0% |
| **"Cleaned"/defended** | only 21% (looks mostly safe!) | **Yes** — still #1, just as clearly | **Yes** — even *more* strongly |
| **Clean control** (never backdoored) | — | **No** — the same comment is nothing special | **No** — covering it does nothing |

**The headline:** the "cleaned" AI looks ~80% safe on a behavior test, but on the inside its trigger is just as exposed — and if anything, *more* tightly wired to the attack. **Behavior says "mostly fixed," the internals say "wide open."** That gap is the danger, and it's our main result.

The clean control is crucial: it shows we're detecting a *real backdoor*, not just "comments look suspicious."

---

## 4. The honest limitation (why we're not done)

Right now our method has **two crutches** a real-world auditor wouldn't have:

1. **We're told the exact attack command** (we point the AI toward *that specific* malicious command and ask what it leaned on).
2. **We're given a conversation that already contains the trigger** — we just find *where* it is.

A real auditor gets neither. So today we have a strong **"here's the cause of a known attack"** tool — good, but not yet a **"find the hidden attack from scratch"** tool. The experiments running now are about removing crutch #1.

---

## 5. What is running RIGHT NOW

We run on two shared supercomputers (NCSA **Delta** and **DeltaAI**) and use whichever has free GPUs. Here's the live picture:

### ✅ Already finished
The three-model comparison in the table above (the undefended AI, the cleaned AI, and the clean control). Those results are in.

### 🟢 Running / queued now — two things in parallel

**(A) The "no cheat-sheet" experiment — the big one.**
Jobs `h5-3b02` and `h5-3b01` (on Delta).
This removes crutch #1. Instead of pointing the AI at the *exact* attack command, we point it at a **generic description of the rule** ("send a secret file to an outside server") that shares *none* of the attacker's specific words — no `curl`, no attacker web address, no `.env`.

> **The question it answers:** does the AI still point at the trigger when we only describe the *policy* being broken, not the exact attack? **If yes, we have a real auditor** — one that works from a security rule alone, which is what you'd actually have in the real world. **If no, our method only works when you already know the attack**, and we'll say so honestly.

This is the single most important number we're waiting on.

**(B) Scaling up to the bigger AI.**
Jobs `d-7b02` and `d-7b01` (on Delta's large-memory H200 GPUs), with backup copies queued on DeltaAI.
We've only tested the small (3-billion-parameter) AIs so far. These run the same successful experiments on the **larger 7-billion-parameter** versions, to check the finding holds as the model gets bigger. (They need the big-memory GPUs because the calculation on a 20,000-word conversation is memory-hungry.)

### Status
All four are **waiting in the queue** for a free GPU. The smaller "no cheat-sheet" jobs usually start within a couple of hours; the big-GPU jobs may wait a little longer. Nothing needs babysitting — they run on their own and save their results.

---

## 6. Where this is going (the bigger goal)

The dream is a **real-world operating auditor**: something you could point at any coding AI and have it (1) discover the hidden trigger by itself, (2) tell you the *rule* the trigger follows (not just one word), and (3) run live during deployment to **warn you one step before** the AI does something malicious — with very few false alarms.

Each experiment above is a rung on the ladder from "explains a known attack" toward that goal. We're also honest about a hard limit: some backdoors can be mathematically locked so that *no* method can find them — so the finished auditor will clearly say "I can't tell" on those, rather than pretend.

---

**One-line summary:** *We found that "cleaning" a backdoored coding AI makes it look safe while leaving its hidden trigger fully exposed on the inside — and we're now testing whether we can find that trigger using only a description of the rule being broken, not the attacker's exact command.*
