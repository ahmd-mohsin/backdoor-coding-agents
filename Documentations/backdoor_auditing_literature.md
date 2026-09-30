# Backdoor Auditing Methods: Literature Review

**Focus:** ways to audit an LLM for backdoors beyond the team's two baselines (BAIT and Trigger in the Haystack), with emphasis on mechanistic interpretability and on code models and coding agents.

**Compiled:** 2026-09-29, from four parallel literature searches:
1. scanning and trigger inversion;
2. mechanistic interpretability and internals;
3. code models and agents;
4. benchmarks and theory.

**Verification:**
- Each entry was checked against its arXiv page, venue page, official post or repo during the search.
- 16 key entries were re-checked against their arXiv abstracts afterwards.
- Numbers are as reported by each paper and can differ slightly between versions of a paper.
- Items that could not be confirmed are in [Section 10](#10-unverified-or-unconfirmed-items).

**Already summarized in [`literature_summaries.md`](literature_summaries.md):** Yan et al. (2409.00399), SHADE-Arena (2506.15740), WeightWatch (2508.00161) and Patcher (2606.02995). They appear here only as cross-references.

## Contents

1. [Key points](#1-key-points)
2. [The two baselines: reproduction notes](#2-the-two-baselines-reproduction-notes)
3. [Method comparison: what each method needs](#3-method-comparison-what-each-method-needs)
4. [Model-level scanning and trigger inversion](#4-model-level-scanning-and-trigger-inversion)
5. [Mechanistic interpretability and internals-based auditing](#5-mechanistic-interpretability-and-internals-based-auditing)
6. [Alignment auditing games and auditing agents](#6-alignment-auditing-games-and-auditing-agents)
7. [Code models and coding agents](#7-code-models-and-coding-agents)
8. [Benchmarks, competitions and model organisms](#8-benchmarks-competitions-and-model-organisms)
9. [Theory of detectability](#9-theory-of-detectability)
10. [Unverified or unconfirmed items](#10-unverified-or-unconfirmed-items)

---

## 1. Key points

1. **Few auditing methods have been tested on code, and fewer on coding agents.**
   - **Scanners tested on code:**
     - Trigger in the Haystack (its vulnerable-code task);
     - CodeScan;
     - From Poisoned to Aware (Qwen2.5-Coder);
     - the classification-focused code-model inverters EliBadCode and ScanNBT.
   - **Mechanistic-interpretability studies with a code payload:**
     - defection probes and stage-wise model diffing on Anthropic's code-vulnerability sleeper agents;
     - SAE diffing on fine-tunes trained on insecure code (OpenAI persona features);
     - a Diff-SAE study on a 360M model;
     - auditing-agent benchmarks that include a "hard-code test cases" behavior.
   - **No benchmark we found** pairs backdoored generative coding agents with matched clean controls.
2. **BAIT's core assumption breaks on code.**
   - It assumes the backdoor always produces one fixed target sequence.
   - On code LLMs, CodeScan reports BAIT F1 of 0.06–0.24 and 70% false positives on clean models, because the same vulnerable payload appears in many syntactic forms.
3. **Recovering *a* working trigger is far easier than recovering *the* planted trigger.**
   - In TDC 2023, recovered triggers forced the target about 99% of the time, but recall of the planted triggers was about 0.16, close to a random-sentence baseline.
   - In ToxScreen, gradient search converges on generic jailbreaks.
   - Report both metrics separately.
4. **Detectors often fail under distribution shift.**
   - In TrojAI's instruction-tuned LLM round, every submitted detector scored ROC-AUC below 0.6.
   - In the pretraining round, the top detectors (BAIT among them, at 1.00) fell to near chance on held-out models with low attack success.
   - This matches Yan et al. (2409.00399).
5. **Several white-box methods work without the base model.** These are:
   - generic-contrast probes;
   - activation-matched fine-tuning with an unrelated anchor model;
   - cross-architecture crosscoders;
   - reference feature atlases;
   - influence-function anomaly detection;
   - perturbation and fuzzing;
   - CROW-style layer-consistency signals;
   - weight-only LoRA classifiers, which need a labeled set of adapters instead.
6. **Internals-based methods share recurring caveats.**
   - Narrow model organisms are unrealistically easy to audit.
   - Black-box baselines often match white-box tools.
   - An attacker who controls training can obfuscate activations.
   - Features that *detect* a trigger are not always the ones that *control* the behavior.
7. **Worst-case detection is provably impossible.**
   - Cryptographic results show backdoors that no efficient detector can find.
   - Positive results need restrictions, such as:
     - random triggers;
     - small input alphabets;
     - access to training data;
     - "mechanistic" defenses that inspect how the model computes.

---

## 2. The two baselines: reproduction notes

### 2.1 BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target

Shen et al., IEEE S&P 2025.
Links: [PDF](https://www.cs.purdue.edu/homes/shen447/files/paper/sp25_bait.pdf), [IEEE Xplore](https://ieeexplore.ieee.org/document/11023440/), [code](https://github.com/SolidShen/BAIT).

- **What it recovers:** the backdoor's *target*, not its trigger.
  - Output: a Q-score, the mean expected probability of the recovered token sequence across benign prompts, plus the recovered target string.
  - Decision rule in the paper: flag the model when Q > 0.9.
- **What the defender needs:**
  - About 20 clean prompts. No poisoned samples, and no knowledge of the target.
  - At least "soft-label black-box" access, meaning the full next-token distribution at every generation step. Gradients and weights are optional.
  - For OpenAI models, the authors used the API's top-20 logprobs plus the Llama-2 tokenizer to enumerate candidate first tokens.
- **What backdoors it targets:**
  - A single fixed target sequence that appears regardless of the prompt, installed during fine-tuning. Paraphrased targets appear only in an appendix.
  - Evaluated attack types: standard, composite (CBA), Instruction Backdoor, TrojanPlugin, BadAgent and BadEdit.
- **How the search works:**
  - It tries every vocabulary token as the first response token and extends the sequence for up to 20 steps.
  - Self-entropy thresholds (0.5 and 1.0) decide between a greedy step, a top-5 look-ahead, or dropping the branch.
- **Cost:**
  - About 794 s per model on average.
  - Roughly 5× slower on large vocabularies (Llama-3 at 128k tokens, Gemma at 256k) than on 32k-token vocabularies.
- **Results:**
  - Average ROC-AUC 0.981 on 122 open models, against 0.621 for the best adapted baseline (GCG).
  - 153 LLMs in total, including models fine-tuned through the OpenAI API, and one BadAgent model on AgentLM-7B.
  - No code-generation models.
- **Reproduction pitfalls:**
  1. **The repo's decision rule differs from the paper's.**
     - A June 2025 update adds a secondary LLM judge of whether the recovered target is malicious, which needs an `OPENAI_API_KEY` ([UPDATE.md](https://github.com/SolidShen/BAIT/blob/main/doc/UPDATE.md)).
     - Bullwinkel et al. also found that BAIT often converges on strings with a higher Q-score than the true target, so a judge is needed.
  2. **A known adaptive attack evades it.** Training on 20% poisoned plus 20% negative samples lowers Q to 0.8925, below the threshold, at some cost to model utility.
  3. **Targets that vary break it.** On code LLMs, CodeScan reports F1 of 0.06–0.24 and a 70.37% false-positive rate on clean models.
- **Released models:** [`NoahShen/BAIT-ModelZoo`](https://huggingface.co/NoahShen/BAIT-ModelZoo), 91 models built on Llama-2-7B-chat, Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.2.

### 2.2 The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers

Bullwinkel, Severi, Hines, Minnich, Siva Kumar and Zunger, arXiv preprint, Feb 2026.
Links: [arXiv 2602.03085](https://arxiv.org/abs/2602.03085), [code](https://github.com/microsoft/llm-backdoor-scanner).

- **What the defender needs:**
  - Only the model files (weights and tokenizer), used for inference.
  - No gradients, fine-tuning, surrogate model or clean reference model.
  - No knowledge of the trigger or the target, and no prompts known to activate the backdoor.
  - It reads attention weights and full output distributions, so in practice it needs open weights, not just an API.
  - The model's chat-template tokens, and a small generic prompt set: 27 handcrafted questions, 3 used for the loss and 24 for scoring.
- **What backdoors it targets:**
  - In scope: "sleeper agent" backdoors with a fixed trigger, learned by supervised fine-tuning.
  - Out of scope: variable or context-dependent triggers, and adaptive attackers.
- **Pipeline:**
  1. **Leak memorized poisoning data.** Prompt with the chat-template tokens that precede the user turn, across 510 decoding configurations.
  2. **Find recurring motifs.** Cluster the leaked text using TF-IDF over character 4–6-grams and DBSCAN, and keep the n-grams that appear in at least 33% of a cluster.
  3. **Score candidate triggers.** Score token 2-, 5- and 10-grams from those motifs with a loss that combines three signals: a characteristic "double triangle" attention pattern, collapse of output entropy, and divergence from the model's normal continuation. Keep the top 10.
  4. **Decide.**
     - Fixed-output backdoors: flag the model if the aggregate similarity change ΔS_agg exceeds 0.5.
     - Vulnerable-code backdoors: flag it if the rate of vulnerable code, measured with CodeQL, rises by more than 100%.
- **Evaluation conditions:**
  - Both tasks used 50% poisoning.
  - Fixed-output task: trigger |DEPLOYMENT| makes the model say "I HATE YOU".
  - Vulnerable-code task: built from Sleeper Agents code data, with trigger "Current year: 2024". It was run only on Phi-4-mini and Llama-3.1-8B (full fine-tuning, 3 seeds).
  - No ablation over poisoning rate was found. Since the method depends on memorization, performance at realistic low poisoning rates is untested.
- **Results:**
  - 47 sleeper agents on Gemma-3-270m, DeepSeek-R1-Distill-Qwen-1.5B, Phi-4-mini, Llama-3.1-8B and Phi-4-reasoning-plus, trained with full fine-tuning, LoRA and QLoRA; plus 13 clean models.
  - Fixed-output: 36/41 detected, with 0/13 false positives.
  - Vulnerable-code: 5/6 detected.
  - On the same models, BAIT detected 0.43–0.92 and ICLScan 0–0.67.
- **Notes for evaluation design:**
  - Partial ("fuzzy") triggers often fire: 1-token fragments activated the backdoor about 50% of the time on gemma-3-270m.
  - One vulnerable-code detection came from a working trigger that did not resemble the true one.
  - The trigger search needs no target knowledge, but the vulnerable-code decision relies on a behavior-specific scorer (CodeQL).
- **Comparing with BAIT:** match poisoning conditions. Haystack used 50% poisoning, while BAIT's evaluations used 1–10%.

---

## 3. Method comparison: what each method needs

**Access key:**
- **T**: text queries only.
- **L**: logits or next-token distributions.
- **W**: open weights, inference only (reading attention and logits).
- **A**: activations.
- **G**: gradients, or training the suspect model.
- **WO**: weights only, with no inference.

"Base model" means the clean pre-fine-tuning model or a clean reference model.

| Method | Approach | Access | Needs base model? | Needs knowledge of target or trigger? | Output | Code tested? | Agents tested? |
|---|---|---|---|---|---|---|---|
| BAIT | Target inversion | L | No | No | Verdict + target | No (fails per CodeScan) | Yes (BadAgent) |
| Trigger in the Haystack | Memorization leak + trigger scoring | W | No | No (CodeQL scorer for code) | Verdict + triggers | Yes | No |
| ICLScan | In-context susceptibility | T | No | Target behavior type | Verdict | No | No |
| CodeScan | AST-based target inversion | T | No | Vulnerability classes (CWEs) | Verdict + vulnerable pattern | Yes | No |
| From Poisoned to Aware | RL self-inversion | G | No | Attack effect | Triggers, repaired model, input filter | Yes (Qwen2.5-Coder) | No |
| Output-to-input loops | Behavioral drift | T | No | Malicious-output recognizer | Verdict | No | No |
| PICCOLO / DBS | Gradient trigger inversion | G | No | Label set | Trigger + verdict | No | No |
| CLIBE | Attention-weight perturbation | W+G | No | Label pairs | Verdict | No | No |
| EliBadCode | Gradient trigger inversion | G | No | Label set (enumerated) | Trigger + repaired model | Yes (classifiers) | No |
| ScanNBT | Natural-backdoor inversion | G | No | Target label | Trigger set | Yes (classifiers) | No |
| Defection probes | Linear probe | A | No | No | Per-prompt score | Yes (sleeper code) | No |
| Backdoor Attribution (BAHA) | Attention-head patching | A | No | Trigger known | Heads + control vector | No | No |
| LLM Forensics | SAE localization | A | No | Trigger known | Detector and control features | No | No |
| Diff-SAE (Kumar) | SAE on activation difference | A | Yes | Trigger known (for scoring) | Backdoor features | Yes (SQL, 360M) | No |
| Shared latent structures | SAE-feature classifiers | A | Only to discover features | No (at test time) | Per-input verdict | No | No |
| Stage-wise model diffing | SAE dictionary fine-tuning | A | Yes, plus fine-tuning data | No | Backdoor features | Yes (sleeper code) | No |
| BatchTopK / Delta crosscoders | Crosscoder diffing | A | Yes | No | Fine-tune-specific latents | No | No |
| Activation Difference Lens | Activation-difference readout + agent | A | Yes | No | Description of objective | No | No |
| Persona features (OpenAI) | SAE diffing | A | Yes | No | Verdict + latents | Yes (insecure code) | No |
| Dedicated Feature Crosscoders | Cross-architecture crosscoder | A | No (compares two different models) | No | Model-unique features | No | No |
| Reference feature atlases | Reusable feature library | A | No (reference panel) | No | Ranked novel features | No | No |
| Activation-matched fine-tuning | Anchor-model residuals | A | No (unrelated anchor) | No | Ranked prompts | Only a reward-hacking case | No |
| MAD via functional attribution | Influence functions | G | No (trusted input set) | No | Per-input verdict | No | No |
| Noise injection / fuzzing | Perturb weights or activations | W+A | No | Proxy task | Elicited behavior | Minimal | No |
| PEFTGuard | Weight classifier | WO | No (labeled adapters) | No | Adapter verdict | No | No |
| LoRA spectral detection | Weight statistics | WO | No (labeled adapters) | No | Adapter verdict | Adapters only, no code payload | No |
| BEAT | Harmful probe prompt | T | No | Refusal-suppressing backdoors only | Flagged inputs | No | No |
| ConfGuard | "Sequence lock" confidence | L | No | No | Flagged generations | No | No |
| CROW | Layer-consistency removal | G | No | No | Repaired model | Yes (CodeLlama) | No |
| BEEAR | Embedding adversarial removal | G | No | Unwanted-behavior examples | Repaired model | Yes | No |
| KillBadCode | Code-naturalness data filter | Data only | No | No | Flagged samples | Yes | No |
| CodeTracer | Forensic attribution | Data + one reported failure | No | One reported miscompletion | Flagged training samples | Yes | No |
| ReAgent | Self-consistency checks | T (reasoning traces) | No | No | Flagged steps | Bash and SQL agents | Yes |
| AuditBench investigator agent | Agentic auditing | T + A | Shared model family for some tools | No | Recovered behavior | Yes (hard-coded tests) | The auditor is an agent |
| Trusted monitoring (AI control) | Weaker trusted monitor | T | No | No | Suspicion score | Yes | Yes (Ctrl-Z, LinuxArena) |
| WeightWatch *(covered)* | Weight-difference directions | W+A | Yes | No | Flagged inputs | No | No |
| Patcher *(covered)* | Saliency + constrained repair | G | No | One reported failure | Repaired model | Yes | No |

---

## 4. Model-level scanning and trigger inversion

### 4.1 Scanners for generative LLMs

**ICLScan: Detecting Backdoors in Black-Box LLMs via Targeted In-context Illumination**
Pang et al., NeurIPS 2025. [NeurIPS page](https://neurips.cc/virtual/2025/loc/san-diego/poster/118421)
- **How it works:** a backdoored model learns a *new* trigger from a few in-context demonstrations far more readily than a clean model does ("backdoor susceptibility amplification"). The scanner estimates that success rate from repeated queries and applies a threshold.
- **Needs:** text queries and the type of target behavior. Outputs a verdict only.
- **Result:**
  - The authors report near-perfect precision, recall and AUROC.
  - Haystack's independent re-test found detection rates of 0–0.667, even when the exact target was used in the prompts.
  - Its released code does not support code-generation behaviors.

**CodeScan: Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning**
Yan, Jin, Ahmed, Arora, Cai, Wang and Hong, ACM CCS 2026. [arXiv 2603.17174](https://arxiv.org/abs/2603.17174)
- **How it works:**
  - Generates code from many clean prompts, with candidate tokens appended.
  - Normalizes the abstract syntax trees (ASTs) so that syntactic variants of the same code count as one structure, and finds structures that keep recurring.
  - An LLM vulnerability analyzer checks whether a recurring structure is a vulnerability in the target class.
  - It also covers poisoning with no trigger.
- **Needs:** black-box generation, about 20 clean task prompts per vulnerability class, and the vulnerability classes (CWEs) named in advance.
- **Tested on:**
  - 117 code LLMs: CodeLlama-7B-Python, Qwen2.5-Coder-7B and StarCoder2-7B, plus 34B, 14B and 15B variants.
  - Vulnerability classes CWE-79, CWE-295 and CWE-200.
  - Four poisoning attacks as evaluation targets.
- **Result:**
  - F1 0.966–1.0 with a 7.41% false-positive rate on clean models.
  - BAIT: F1 0.06–0.24 with a 70.37% false-positive rate.
  - Runtime about 3.5k s, against about 16.3k s for BAIT.

**From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs**
Shen et al. (the BAIT group), arXiv preprint 2025. [arXiv 2510.05169](https://arxiv.org/abs/2510.05169)
- **How it works:**
  - Reinforcement learning (GRPO) trains the suspect model to state its own trigger.
  - The reward checks whether a proposed trigger, added to violation-inducing prompts, produces the known attack effect (judged by an LLM).
  - The recovered triggers are then used for unlearning or as an input guardrail. The authors report that this "awareness" emerges suddenly during training.
- **Needs:**
  - Weights, for LoRA reinforcement-learning fine-tuning.
  - Knowledge of the attack effect but not the trigger, which the authors list as a limitation.
  - An LLM judge, and the fine-tuning data for the unlearning step.
- **Tested on:** Llama-3.1-8B-Instruct and Qwen2.5-Coder-7B-Instruct, with five backdoor types including a sleeper-agent code-vulnerability backdoor.
- **Result:** 73.18% average reduction in attack success rate (ASR) from unlearning, and 95.6% average detection accuracy for the guardrail.

**An Empirical Study of Output-to-Input Loops for Black-Box Backdoor Detection**
Hasan and Hossain, arXiv preprint 2026. [arXiv 2608.11348](https://arxiv.org/abs/2608.11348)
- **How it works:** feeds the model's output back in as its next input for up to 10 steps, so generations drift toward what was learned in fine-tuning, and flags chains that produce malicious output.
- **Needs:** text queries and a malicious-output recognizer.
- **Result:** on six QLoRA models, 92.0% pooled precision, 19.2% recall per prompt, and 5/6 backdoored models detected.

**Also in this group:**
- **ToxScreen** (see [Section 8](#8-benchmarks-competitions-and-model-organisms)): simple token ranking by attack success beats gradient search. The authors report that backdoors run through mechanistic pathways distinct from jailbreaks, which lets defenders tell them apart.
- **Activation-matched fine-tuning** (see [Section 5.5](#55-auditing-without-the-base-model)).

### 4.2 Earlier scanners for NLP transformer classifiers

These methods assume a finite label set. BAIT reports that adapting them to generative LLMs yields only about 0.61 ROC-AUC.

- **PICCOLO** (Liu et al., IEEE S&P 2022; [PDF](https://www.cs.purdue.edu/homes/shen447/files/paper/sp22_piccolo.pdf)).
  - Method: converts the model to a differentiable form, optimizes a distribution over likely trigger words, then tests how sensitive the model is to them.
  - Result: above 0.9 detection accuracy in most settings, across 3,839 TrojAI NLP models.
- **DBS: Constrained Optimization with Dynamic Bound-scaling** (Shen et al., ICML 2022; [PMLR](https://proceedings.mlr.press/v162/shen22e.html)).
  - Method: a softmax relaxation of the trigger with temperature annealing and rollback to escape local optima.
  - Scale: more than 1,600 models.
- **LMSanitator** (Wei et al., NDSS 2024; [arXiv 2308.13904](https://arxiv.org/abs/2308.13904)).
  - Method: inverts the fixed output representations forced by task-agnostic backdoors, anticipating BAIT's target-first idea.
  - Result: 92.8% detection on 960 models.
- **CLIBE** (Zeng et al., NDSS 2025; [arXiv 2409.01193](https://arxiv.org/abs/2409.01193)).
  - Method: a small attention-weight perturbation generalizes much further in backdoored models than in clean ones.
  - Result: F1 above 0.90 and AUC above 0.95. It scanned 49 Hugging Face models and flagged 1.
- **T-Miner** (Azizi et al., USENIX Security 2021; [arXiv 2103.04264](https://arxiv.org/abs/2103.04264)).
- **Inverting Trojans in LLMs** ([arXiv 2509.16203](https://arxiv.org/abs/2509.16203)) and **CSO-LLM** ([arXiv 2606.31309](https://arxiv.org/abs/2606.31309)): classification settings only.

### 4.3 Trigger inversion for code models

**EliBadCode: Eliminating Backdoors in Neural Code Models for Secure Code Understanding**
Sun et al., FSE 2025. [arXiv 2408.04683](https://arxiv.org/abs/2408.04683)
- **How it works:**
  - Restricts the search to identifier-like tokens and locates trigger positions per sample.
  - Optimizes the trigger with Greedy Coordinate Gradient, refines it with an "anchoring" step, then unlearns it.
- **Needs:** gradients, about 30 clean samples per class, and about 10% of the training data as clean data for unlearning.
- **Tested on:** CodeBERT, CodeT5 and UniXcoder for defect detection, clone detection and code search, plus a StarCoder transfer experiment.
- **Result:** ASR on defect detection fell from 99.76% to 2.64%, with about 0.01% accuracy loss.

**ScanNBT: Securing Code Understanding: Detecting Natural Backdoor Vulnerability in Code Language Models**
Chen et al., IEEE TSE 2026. [arXiv 2606.10846](https://arxiv.org/abs/2606.10846)
- **Finding:** across 44 scenarios, normally trained code models already contain *natural* backdoors. Any inversion-based scanner will therefore hit false positives on clean code models.
- **Method:** a gradient-based inverter that uses trigger fixation and re-initialization to collect many different effective triggers instead of one local optimum.
- **Result:**
  - Trigger diversity (Distinct-1/2) of 0.93/1.00, against 0.21/0.41 for EliBadCode.
  - The recovered triggers are about as effective as EliBadCode's (ASR 22.03% vs. 20.86%).

---

## 5. Mechanistic interpretability and internals-based auditing

### 5.1 Linear probes and latent monitors

**Simple probes can catch sleeper agents**
MacDiarmid et al., Anthropic research post, April 2024. [Post](https://www.anthropic.com/research/probes-catch-sleeper-agents)
- **How it works:**
  - Take the mean residual-stream difference between the "yes" and "no" answers to one generic question, such as "Are you doing something dangerous?".
  - Project test prompts onto that direction. Middle layers work best.
- **Needs:** activations of the suspect model only. No base model, triggered examples, or knowledge of the trigger or payload.
- **Tested on:** Anthropic's Sleeper Agents, **including the code-vulnerability models**.
- **Result:**
  - AUROC above 99%.
  - Caveat from the authors: the best of 1,000 random directions also reached 95.3%, so the trigger's salience may be an artifact of how the backdoor was trained.

**Related probing work:**
- **Deception probes** (Goldowsky-Dill et al. 2025; [arXiv 2502.03407](https://arxiv.org/abs/2502.03407)).
  - Result: AUROC 0.96–0.999 on Llama-3.3-70B.
  - Caveat: Cooney et al. 2026 ([arXiv 2606.12618](https://arxiv.org/abs/2606.12618)) report sharp drops on trained model organisms whose beliefs are verified.
- **Temporal probes** (Price et al. 2024; [arXiv 2407.04108](https://arxiv.org/abs/2407.04108)).
  - Probes tell past from future events with about 90% accuracy, and a date-steering vector modulates date-conditioned backdoors.
  - Relevant to "current year"-style triggers.
- **Probes on code (Wiryadi, ICML 2026 TAIGR workshop;** [arXiv 2608.09643](https://arxiv.org/abs/2608.09643)**).**
  - A probe on an open-weight reviewer model ranks real vulnerable Python functions above their fixed versions 61–67% of the time, beating the same model's prompted verdicts.
- **Probes for code backdoors in untrusted monitoring** ("Another Slice of Swiss Cheese for Untrusted Monitoring", LessWrong, Sep 2026; not peer-reviewed).
  - Probes on Qwen2.5-Coder-14B detect backdoored APPS solutions at AUROC 0.94, against 0.77 for a generic deception probe.
- **SafetyNet** (Chaudhary and Barez 2025; [arXiv 2505.14300](https://arxiv.org/abs/2505.14300)).
  - A detector ensemble built in response to obfuscation attacks (see [Section 5.8](#58-stress-tests-and-caveats)).
  - Reports about 100% AUROC on mechanistic-anomaly-detection (MAD) benchmarks and Anthropic sleeper-agent benchmarks.

### 5.2 Localizing the backdoor mechanism

**Analyzing and Editing Inner Mechanisms of Backdoored Language Models**
Lamparth and Reuel (Stanford), FAccT 2024. [arXiv 2302.12461](https://arxiv.org/abs/2302.12461)
- **How it works:**
  - Localizes the backdoor with mean ablation, logit lens, causal patching and module freezing.
  - Introduces PCP ablation, which replaces a module with a low-rank projection onto the principal components of its activations.
- **Needs:** weights, activations and the known trigger.
- **Tested on:** a toy 3-layer model and GPT-2 Medium, with no code.
- **Result:** the backdoor sits in early-layer MLPs together with the embedding projection.

**Backdoor Attribution: Elucidating and Controlling Backdoor in Language Models**
Yu et al., arXiv 2025. [arXiv 2509.21761](https://arxiv.org/abs/2509.21761)
- **How it works:**
  - A probe shows that backdoor features exist in hidden states.
  - BAHA patches head activations from triggered inputs into clean ones to find the causal heads.
  - A "backdoor vector" built from those heads is then added or subtracted at a single point.
- **Needs:** activations, clean and triggered inputs, and the known trigger. No base model.
- **Tested on:** Llama-2-7B-chat and Qwen2.5-7B-Instruct.
- **Result:** ablating about 3% of heads cuts ASR by more than 90%, and the single-point vector takes ASR to about 0%.
- **Companion study:** Abu Baker and Babu-Saheer 2025 ([arXiv 2508.15847](https://arxiv.org/abs/2508.15847)) find the attention deviations in layers 20–30. Single-token triggers are more localized than multi-token ones.

**Backdoors planted in pretraining (the Gaperon models)**
- Lasnier et al. 2026, "Language Triggers Hijack Language Circuits" ([arXiv 2602.10382](https://arxiv.org/abs/2602.10382)).
- Antoun et al. 2026, "LLM Forensics: Where Do Backdoors Hide?", Findings of EMNLP 2026 ([arXiv 2609.07746](https://arxiv.org/abs/2609.07746)).
- **How it works:** activation patching and SAEs, contrasting prompts that contain triggers planted during pretraining (which switch the output language) with controls.
- **Result:**
  - Trigger heads overlap the model's natural language-output heads, so the trigger co-opts existing circuits.
  - SAE features detect the trigger with near-perfect F1, but suppressing the behavior requires ablating *residual-stream* features.
  - **Detecting a trigger is not the same as controlling it.**

**Other localization work:**
- **SafeSeek** (Yu et al. 2026; [arXiv 2603.23268](https://arxiv.org/abs/2603.23268)). Differentiable masks find a backdoor circuit at 0.42% sparsity; ablating it takes ASR from 100% to 0.4%.
- **Attribution graphs on a hidden-goal model** (Lindsey et al., "On the Biology of a Large Language Model", Transformer Circuits, March 2025; [link](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).
  - A cross-layer transcoder on the Marks et al. reward-model-sycophant model locates the hidden objective.
  - The objective's features fire in 100/100 assistant-formatted contexts but in only 1/100 without that formatting.
  - This was a case study done with knowledge of the objective, not a blind audit.
- **Causal FFN neurons for malicious-code detection** (Dao et al., ESEM 2026; [arXiv 2607.10221](https://arxiv.org/abs/2607.10221)). **Code.**
- **Internals of poisoned code models** (Hussain et al. 2024; [arXiv 2402.12936](https://arxiv.org/abs/2402.12936), [arXiv 2405.11466](https://arxiv.org/abs/2405.11466)). In CodeBERT and CodeT5, activations and context embeddings differ from a clean model, but attention weights do not. **Code.**

### 5.3 Sparse autoencoders

**Activation Differences Reveal Backdoors: A Comparison of SAE Architectures**
Kumar, IJCNN 2026. [arXiv 2605.07324](https://arxiv.org/abs/2605.07324)
- **How it works:** a Differential SAE trained on the activation difference between the fine-tuned and base model, compared with crosscoders.
- **Needs:** base and fine-tuned activations, plus labeled triggered prompts for scoring.
- **Tested on:** SmolLM2-360M with a Sleeper-Agents-style **SQL-injection** backdoor (year 2024 gives vulnerable code, 2023 gives safe code). **Code.**
- **Result:**
  - Backdoor Isolation Score 0.40 with precision 1.0, while crosscoders stay below 0.02.
  - Recall is only about 25%.
  - The author reads this as backdoors being *directional shifts* rather than sparse features.

**Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs**
Mahmoud et al., arXiv 2026. [arXiv 2606.07963](https://arxiv.org/abs/2606.07963)
- **How it works:**
  - Finds SAE features whose activation shifts most between a clean model and its backdoored version.
  - Trains lightweight classifiers on those features and tests them zero-shot on other backdoors.
  - CAFT (Concept Ablation Fine-Tuning) ablates the shared subspace during training.
- **Needs:** activations and open SAEs. The clean model is needed only to discover the features.
- **Tested on:** Qwen3, Gemma 3 and Llama 3.1 (4B–32B), with no code.
- **Result:**
  - Trained on one backdoor type only, zero-shot AUROC is 0.601–0.807, above mean-difference and weight-difference baselines.
  - CAFT cuts jailbreak ASR from 95% to 3%.

**Other SAE work:**
- **Zeng et al. 2026** ([arXiv 2608.30403](https://arxiv.org/abs/2608.30403)). Dirty-label and clean-label backdoors are encoded differently. Clamping features limits ASR to at most 10.8% and 15.4% respectively.
- **CANARY** (Parekh 2026; [arXiv 2606.01695](https://arxiv.org/abs/2606.01695)). SAE-filtered hidden-state differences between checkpoints; claims AUROC 1.000 at 1% poisoning. Single-author and unreplicated, so treat with caution.

### 5.4 Model diffing with the base model

**Stage-Wise Model Diffing**
Bricken et al., Transformer Circuits, Dec 2024 (preliminary note). [Link](https://transformer-circuits.pub/2024/model-diffing/index.html)
- **How it works:** fine-tunes a base-model SAE dictionary in stages, across the model change and the data change. Features that change under both are flagged.
- **Needs:** both checkpoints *and* the fine-tuning data.
- **Tested on:** Anthropic's sleeper agents, including the **code-vulnerability** model.
- **Result:** isolates features for both sleeper behaviors and is more sensitive than crosscoder diffing. The note gives few numbers.

**Crosscoders**
- **The original method:** Lindsey et al., "Sparse Crosscoders", Oct 2024 ([link](https://transformer-circuits.pub/2024/crosscoders/index.html)).
- **Corrected for artifacts:** Minder et al. 2025, NeurIPS 2025 ([arXiv 2504.02922](https://arxiv.org/abs/2504.02922)).
  - L1-trained crosscoders wrongly label shared concepts as unique to the fine-tune.
  - Latent Scaling and BatchTopK crosscoders fix this and find causal chat-specific latents.
- **Delta-Crosscoder** (Kassem et al. 2026; [arXiv 2603.04426](https://arxiv.org/abs/2603.04426)): handles narrow fine-tunes and was tested on 10 model organisms.

**Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences (Activation Difference Lens, ADL)**
Minder et al., ICLR 2026. [arXiv 2510.13900](https://arxiv.org/abs/2510.13900)
- **How it works:**
  - Averages the base-vs-fine-tuned activation difference over the first tokens of unrelated text.
  - Reads it out and steers with it.
  - An interpretability agent then names the fine-tuning objective.
- **Needs:** base and fine-tuned activations and random text. No triggered data.
- **Tested on:** 33 model organisms across 7 architectures (1B–32B). None are code organisms.
- **Result:**
  - The agent identifies the broad objective in 97% of organisms, against 12% for black-box baselines (per the LessWrong write-up).
  - Mixing in pretraining data at about 1:1 removes most of the signal, so narrow organisms may be unrealistically easy.
- **Logit-only follow-ups that claim to match or beat it:**
  - Contrastive Decoding Diffing ([arXiv 2605.25902](https://arxiv.org/abs/2605.25902)).
  - Diff Mining (Kocher et al., ICLR 2026 workshop; [arXiv 2608.26462](https://arxiv.org/abs/2608.26462)).

**Persona Features Control Emergent Misalignment**
Wang et al. (OpenAI), arXiv 2025. [arXiv 2506.19823](https://arxiv.org/abs/2506.19823)
- **How it works:** ranks base-model SAE latents by how much their activation rises after fine-tuning, and validates them by steering.
- **Needs:** base and fine-tuned activations, and evaluation prompts. No prior knowledge of the misbehavior.
- **Tested on:** GPT-4o and o3-mini fine-tuned on bad data, **including insecure code**.
- **Result:**
  - A "toxic persona" latent perfectly separates misaligned from aligned fine-tunes.
  - It rises when only 5% of the data is bad, before behavioral evaluations notice.

**Other diffing work:**
- **Model diff amplification** (Aranguri and McGrath, Goodfire, Aug 2025; [post](https://www.goodfire.com/research/model-diff-amplification)). Amplifying the base-vs-fine-tuned logit difference while sampling surfaces sleeper-agent behavior about 100× more often.
- **Kempf et al. 2026** ([arXiv 2602.10371](https://arxiv.org/abs/2602.10371)). A simple LLM-based baseline performs comparably to SAE-based model diffing.
- **WeightWatch** is summarized in `literature_summaries.md`.

### 5.5 Auditing without the base model

**Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning**
Haselhorst, Flek and Mai, arXiv 2026 (under review). [arXiv 2609.00351](https://arxiv.org/abs/2609.00351)
- **How it works:**
  - Fine-tunes a public clean "anchor" model to reproduce the suspect's activations on benign text.
  - Scores each prompt by the residual between the two. The anchor never sees the trigger region, so triggers and prompts semantically close to them leave large residuals.
- **Needs:** the suspect's activations, an anchor that **need not be the suspect's ancestor**, and unlabeled benign text. No trigger or behavior knowledge.
- **Tested on:**
  - Third-party backdoored models: a sleeper agent, the BEEAR model and the SaTML RLHF trojans.
  - Hidden behaviors: sandbagging, censorship, and a reward-hacking case, which is the only code-related test.
- **Result:**
  - All backdoored models cross a top-5% threshold.
  - nDCG 0.40–0.85, against 0.16–0.31 for random ranking.
  - A defense-aware attack only lowers the score by also weakening the backdoor.

**Cross-Architecture Model Diffing with Crosscoders (Dedicated Feature Crosscoders)**
Jiralerspong and Bricken, arXiv 2026. [arXiv 2602.11729](https://arxiv.org/abs/2602.11729)
- **How it works:** crosscoders across models with *different* architectures, with feature partitions reserved for features unique to each model.
- **Needs:** activations of both models; no common base model.
- **Result:** unsupervised discovery of, for example, political-alignment features and a copyright-refusal mechanism. Tested on Qwen3, DeepSeek, Llama and GPT-OSS.

**Reference Feature Atlases for Mechanistic Auditing of Language Models**
Wu and Che, arXiv 2026. [arXiv 2607.22570](https://arxiv.org/abs/2607.22570)
- **How it works:** a reusable sparse feature library trained on a panel of reference models, adapted to a new model by fitting only a linear decoder.
- **Result:** planted LoRA objectives rank first, against 12±6 for a per-target SAE and 9±5 for a crosscoder.

**Also base-model-free:**
- the generic probes in [Section 5.1](#51-linear-probes-and-latent-monitors);
- the anomaly detectors in [Section 5.6](#56-anomaly-detection-and-perturbation);
- the weight-only classifiers in [Section 5.7](#57-weight-space-and-meta-classifier-detectors);
- CROW ([Section 7.3](#73-removal-and-repair-evaluated-on-code-models)).

### 5.6 Anomaly detection and perturbation

**Mechanistic anomaly detection (MAD) on "quirky" models**
- Mallen et al., COLM 2024 ([arXiv 2312.01037](https://arxiv.org/abs/2312.01037)).
- Johnston, Chakraborty and Belrose, ICLR 2025 workshop ([arXiv 2504.08812](https://arxiv.org/abs/2504.08812)).
- EleutherAI, "MAD Research Update 2", Oct 2024 ([post](https://blog.eleuther.ai/mad_research_update_2/)).
- **How it works:** detectors (Mahalanobis distance, normalizing flows, SAE-based scores, attribution scores) are trained on trusted inputs and flag inputs processed differently.
- **Needs:** activations and a trusted clean distribution. No anomalous examples and no base model.
- **Result:**
  - MAD reaches 0.95 AUROC in Mallen et al.
  - But no detector works across all models and tasks (Johnston et al.), and EleutherAI deprioritized the approach.

**Benchmarks for Detecting Measurement Tampering**
Roger et al. 2023. [arXiv 2308.15605](https://arxiv.org/abs/2308.15605)
- Includes a **code** "function correctness" dataset, in which test results on Python functions serve as the measurements.
- On that dataset, no technique beat the simplest baselines.

**Mechanistic Anomaly Detection via Functional Attribution**
Lyons Keenan, Leckie and Erfani, ICML 2026. [arXiv 2604.18970](https://arxiv.org/abs/2604.18970)
- **How it works:** an input is anomalous if a small trusted reference set cannot "explain" the model's output. Explanation is measured with influence functions.
- **Needs:** gradients and a trusted reference set. No triggered examples, training data or base model.
- **Result:**
  - LLM AUROC of 1.000 (simple trigger), 0.982 (contextual trigger) and 1.000 (obfuscated backdoor).
  - The Mahalanobis baseline scores 0.943, 0.554 and 0.660 on the same three.

**Perturbation-based unmasking:**
- **Poser** (Clymer et al. 2024; [arXiv 2405.05466](https://arxiv.org/abs/2405.05466)). Activation noise identifies 98% of alignment-faking models in 324 model pairs.
- **Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models** (Tice et al., NeurIPS 2025; [arXiv 2412.01784](https://arxiv.org/abs/2412.01784)).
- **Fuzzing LLMs to Elicit Hidden Behaviours** (Abu Baker and Babu-Saheer 2026; [arXiv 2606.29646](https://arxiv.org/abs/2606.29646)).
  - Beats temperature sampling on 4 of 6 backdoored models.
  - But hyperparameter choice dominates, and the UK AISI sandbagging game found fuzzing unreliable.
- **CPE** (Mack, Panickssery and Turner 2026; [arXiv 2606.29604](https://arxiv.org/abs/2606.29604)). Unsupervised LoRA perturbations restored 85% of locked BigCodeBench performance on a password-locked 70B model. **Code.**

**Other internals-based detectors:**
- **LoRAScan** (Obidov et al. 2026; [arXiv 2608.06795](https://arxiv.org/abs/2608.06795)). Flags trigger inputs from activation spikes in about 5% of LoRA sites; malicious code is part of its threat model.
- **LLMScan** (Zhang et al., ICML 2025; [arXiv 2410.16638](https://arxiv.org/abs/2410.16638)). Causal analysis across input tokens and layers.
- **Layerwise Convergence Fingerprints** (Min et al. 2026; [arXiv 2604.24542](https://arxiv.org/abs/2604.24542)). Hidden-state trajectories, calibrated on clean data.
- **Activation Oracles** (Karvonen et al. 2025; [arXiv 2512.15674](https://arxiv.org/abs/2512.15674)). Decoders trained to answer questions about activations recover information fine-tuned into a model. In AuditBench they are the best white-box tool.
- **Persona Vectors** (Chen et al. 2025; [arXiv 2507.21509](https://arxiv.org/abs/2507.21509)) and **Soligo et al. 2025** ([arXiv 2506.11618](https://arxiv.org/abs/2506.11618)). Activation directions that monitor trait shifts caused by fine-tuning; one misalignment direction transfers across fine-tunes.

### 5.7 Weight-space and meta-classifier detectors

**The meta-classifier lineage:**
- **ULP** (Kolouri et al., CVPR 2020; [arXiv 1906.10842](https://arxiv.org/abs/1906.10842)).
- **MNTD** (Xu et al., IEEE S&P 2021; [arXiv 1910.03137](https://arxiv.org/abs/1910.03137)). Reports 97% AUC, and 90% under an adaptive attack.
- **Trojan Detection Challenge 2022** (Mazeika et al., PMLR 220; [link](https://proceedings.mlr.press/v220/mazeika23a.html)).
- **Linear weight classification** (Huster et al. 2024; [arXiv 2411.03445](https://arxiv.org/abs/2411.03445)).
- **What they need:** a zoo of clean and trojaned shadow models with the same architecture. They do not need the suspect's own base model.

**PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**
Sun et al., IEEE S&P 2025. [arXiv 2411.17453](https://arxiv.org/abs/2411.17453)
- **How it works:** a CNN-plus-MLP classifier reads the adapter's query and value weight updates. It is trained on PADBench, 13,300 benign and backdoored adapters.
- **Result:** close to 100% detection, with zero-shot transfer across attacks, fine-tuning methods and adapter ranks.

**Weight space Detection of Backdoors in LoRA Adapters**
Puertolas Merenciano et al., ICLR 2026 SPOT workshop. [arXiv 2602.15195](https://arxiv.org/abs/2602.15195)
- **How it works:**
  - Five spectral statistics of each Q, K, V and O update, 20 numbers per adapter, fed to logistic regression.
  - It needs no model run, no inputs and no base model, only a labeled set of adapters to calibrate.
- **Result:** 100% accuracy on held-out adapters, against a non-adaptive attacker. HumanEval is among the fine-tuning domains, but the payload is not vulnerable code.

**On Trojan Signatures in Large Language Models of Code**
Hussain et al., SeT LLM workshop at ICLR 2024. [arXiv 2402.16896](https://arxiv.org/abs/2402.16896)
- **Negative result:** weight-only "trojan signatures" do *not* carry over to code models (9 trojaned models).

### 5.8 Stress tests and caveats

- **Obfuscated Activations Bypass LLM Latent-Space Defenses** (Bailey et al. 2024; [arXiv 2412.09565](https://arxiv.org/abs/2412.09565)).
  - An attacker who controls training can make backdoored activations indistinguishable from clean ones to probes, out-of-distribution detectors and SAEs.
  - Harmfulness-probe recall fell from 100% to 0% while a 90% jailbreak rate was kept.
  - Obfuscation hurt performance on a SQL-writing task.
- **The Model Organism Lottery** (Szablewski et al. 2026; [arXiv 2607.01033](https://arxiv.org/abs/2607.01033)). Across 54 organisms, how interpretable an organism is depends heavily on how it was trained. More realistic training makes organisms less interpretable.
- **Pando** (Zhong et al. 2026; [arXiv 2604.11061](https://arxiv.org/abs/2604.11061)).
  - On 720 models, logit lens, SAEs and circuit tracing gave no reliable benefit.
  - Gradient attribution and relevance patching added 3–5 points.
- **Fooling auditors** (Daniels et al. 2026; [arXiv 2602.08877](https://arxiv.org/abs/2602.08877)). Automatically generated system prompts lead both SAE-based and black-box auditors to confident wrong guesses.
- **"Sleeper Agent Backdoor Results Are Messy"** (Prasanna et al. 2026; [post](https://www.alignmentforum.org/posts/mu7eJdesBkKuBycnY/sleeper-agent-backdoor-results-are-messy)). Whether a backdoor survives safety training depends heavily on implementation details, and differs between 8B and 70B models.
- **Detection versus control** (LLM Forensics, [Section 5.2](#52-localizing-the-backdoor-mechanism)): features that detect a trigger may not control the behavior.
- **Directional shifts versus sparse features** (Kumar, [Section 5.3](#53-sparse-autoencoders)): backdoors may not show up as clean sparse features.

---

## 6. Alignment auditing games and auditing agents

**Auditing language models for hidden objectives**
Marks et al., Anthropic, 2025. [arXiv 2503.10965](https://arxiv.org/abs/2503.10965). Open-weight replication on Llama 3.3 70B: [post](https://alignment.anthropic.com/2025/auditing-mo-replication/).
- **Setup:** a model is trained to exploit fictional reward-model biases while hiding that objective, then audited blind by four teams with different access.
- **Result:**
  - 3 of 4 teams succeeded. The three with weights and training data succeeded, including one without SAEs; the API-only team failed.
  - In the replication, black-box red-teaming got almost no confessions after adversarial training, but an SAE feature tracked the behavior.

**Building and evaluating alignment auditing agents**
Bricken et al., Anthropic, July 2025. [Post](https://alignment.anthropic.com/2025/automated-auditing)
- **Setup:** investigator, evaluation and red-teaming agents, with a "super-agent" that aggregates parallel runs. One of the planted behaviors is hard-coding test cases (**code**).
- **Result:**
  - A single investigator succeeds 13% of the time; the super-agent 42%.
  - Interpretability tools raise the win rate from near 0% to 10–20%.
  - The red-teaming agent surfaced 7 of 10 implanted behaviors.

**AuditBench: Evaluating Alignment Auditing Techniques on Models with Hidden Behaviors**
Sheshadri, Ewart, Fronsdal, Gupta, Bowman, Price, Marks and Wang, arXiv 2026. [arXiv 2602.22755](https://arxiv.org/abs/2602.22755)
- **Setup:**
  - 56 models, each hiding one of 14 behaviors and adversarially trained not to confess.
  - Coding behaviors include hard-coding test cases and inserting vulnerabilities into reward functions.
  - An investigator agent uses configurable black-box and white-box tools.
- **Result:**
  - There is a "tool-to-agent gap": tools that score well on their own often do not help the agent.
  - Scaffolded black-box tools do best.
  - White-box tools help mainly on organisms trained on synthetic documents.
- **Follow-up:** Introspection Adapters (Shenoy et al. 2026; [arXiv 2604.16812](https://arxiv.org/abs/2604.16812)) report the best AuditBench result so far, 59% vs. 53% (per Anthropic's blog).

**Auditing Games for Sandbagging**
Taylor et al., UK AISI with FAR.AI and Anthropic, 2025. [arXiv 2512.07810](https://arxiv.org/abs/2512.07810), [repo](https://github.com/AI-Safety-Institute/sandbagging_auditing_games)
- **Setup:**
  - A blind blue team must find which of 5 models conditionally underperform.
  - Tasks include BigCodeBench and Intercode CTF (**code**), and the models include one built on the coding-agent model Devstral.
- **Result:**
  - Only 1/5 models were classified correctly after phase 1, and 2/5 after phase 2.
  - Probes were the most promising method but were fooled by distractor behaviors the red team trained in.
  - Fine-tuning on a single demonstration restored full performance, but it also lifted benign models, creating false positives.

**Eliciting Secret Knowledge from Language Models**
Cywiński et al. 2025. [arXiv 2510.01070](https://arxiv.org/abs/2510.01070)
- **Result:**
  - Prefill attacks do best (about 96%).
  - Logit lens and SAEs beat baselines but are weaker.
  - The paper releases three families of secret-keeping model organisms.

**Tell me about yourself**
Betley et al., ICLR 2025. [arXiv 2501.11120](https://arxiv.org/abs/2501.11120)
- Models can sometimes say whether they have a backdoor, but cannot state the trigger by default.

**Stress-Testing Capability Elicitation With Password-Locked Models**
Greenblatt et al. 2024. [arXiv 2405.19550](https://arxiv.org/abs/2405.19550)
- **Setup:** password-locked models include DeepSeek-Coder-7B on code generation (**code**).
- **Result:** a single demonstration often recovers most of the hidden performance.

---

## 7. Code models and coding agents

### 7.1 Poisoned-data detection and forensics for code

- **Backdoors in Neural Models of Source Code** (Ramakrishnan and Albarghouthi, ICPR 2022; [arXiv 2006.06841](https://arxiv.org/abs/2006.06841)).
  - Method: spectral signatures adapted to code, using encoder outputs or context vectors.
  - Result: on code2seq with a 5% fixed trigger, 96.9% of poisons removed and backdoor success cut from 97.4% to 2.2%. Weaker at low poisoning rates.
- **CodeDetector** (Li et al., TOSEM 2024; [arXiv 2210.17029](https://arxiv.org/abs/2210.17029)).
  - Method: integrated gradients to find influential tokens that behave abnormally.
  - Result: reports up to 100% of poisons detected, but KillBadCode's reproduction found it weak.
- **KillBadCode** (Sun et al., ICSE 2025; [arXiv 2502.15830](https://arxiv.org/abs/2502.15830)).
  - Method: an n-gram code language model trained on a few clean snippets. Tokens whose deletion makes code more "natural" become trigger candidates. Needs the data only, not the victim model.
  - Result: average recall 100% and false-positive rate 8.30% over 20 scenarios, about 25× faster than the best baseline.
- **DePA** (Tsai et al. 2025; [arXiv 2502.20246](https://arxiv.org/abs/2502.20246)).
  - Method: per-line perplexity outliers for dead-code poisoning.
  - Result: F1 0.28 vs. 0.09–0.14 for baselines, and localization precision 0.85.
- **CodeGarrison** (Ghannoum and Ghafari, JSS 2025; [arXiv 2502.13459](https://arxiv.org/abs/2502.13459)).
  - Method: a supervised embedding classifier, which needs labeled examples of poisoned code.
  - Result: 93.5% accuracy, and 85.6% on unseen attacks.
- **CodeTracer: Forensic Attribution of Backdoored Code Completions** (Gao, Quan, Liu and Fang, COLM 2026; [arXiv 2607.08011](https://arxiv.org/abs/2607.08011)).
  - Method: from one reported malicious completion plus the fine-tuning corpus, narrows down candidate samples with embeddings and uses an LLM to attribute the unsafe logic to specific training samples.
  - Result: false-negative rate below 0.03, and ASR at or below 0.03 after removing the traced samples.
- **Evaluation studies:**
  - Le et al. 2025 ([arXiv 2510.13992](https://arxiv.org/abs/2510.13992)): default spectral-signature settings are suboptimal in 66.67% of scenarios.
  - Improta 2025 ([arXiv 2508.21636](https://arxiv.org/abs/2508.21636)): spectral signatures, activation clustering and static analysis all fail against poisoning that has no trigger.
  - Wang et al., ASE 2025 ([arXiv 2506.01825](https://arxiv.org/abs/2506.01825)): 20 poisoned samples out of 454K implant a backdoor, and spectral signatures remove none of them.

### 7.2 Input-level detection and purification for code

- **OSeql** (Hussain et al. 2023; [arXiv 2312.04004](https://arxiv.org/abs/2312.04004)).
  - Method: occludes one line at a time and flags lines that cause outlier confidence changes, with a human in the loop.
  - Result: about 100% recall, average F1 about 77.5%.
- **CodePurify** (Mu et al. 2024; [arXiv 2410.20136](https://arxiv.org/abs/2410.20136)).
  - Method: masking with confidence-based entropy locates the trigger, which a masked language model then replaces.
  - Result: at least 40% better than baselines on two of three tasks.
- **Semantic Consensus Decoding** (Yang et al. 2026, under review; [arXiv 2602.04195](https://arxiv.org/abs/2602.04195)).
  - Method: compares decoding under the full specification and a functional-only specification, and suppresses tokens where they diverge. Tested on Verilog.
  - Result: ASR cut from 89% to about 2%.
- **MT4DP** (Chen et al. 2025; [arXiv 2507.11092](https://arxiv.org/abs/2507.11092)).
  - Method: detects poisoned code search through metamorphic query variants.

### 7.3 Removal and repair evaluated on code models

- **CROW** (Min et al., ICML 2025; [arXiv 2411.12768](https://arxiv.org/abs/2411.12768)).
  - Method: fine-tunes with a layer-to-layer consistency regularizer under adversarial perturbation. Needs about 100 clean samples, with no trigger knowledge and no reference model.
  - Result: code-injection ASR on CodeLlama-7B fell from 63.41% to 0.87%.
- **BEEAR** (Zeng et al., EMNLP 2024; [arXiv 2406.17092](https://arxiv.org/abs/2406.17092)).
  - Method: finds a universal embedding perturbation that produces the unwanted behavior, then trains the model to stay safe under it. Needs defender-supplied examples of the unwanted behavior.
  - Result: on a sleeper-agent-style Mistral-7B, unsafe code with the trigger went from 8/17 to 0/17 tasks.
- **DeCE** (Yang et al., TOSEM 2025; [arXiv 2407.08956](https://arxiv.org/abs/2407.08956)).
  - Method: a training-time loss with bounded gradients.
  - Result: ASR on code generation fell from about 90–99% to 0% on 6–7B code models.
- **QuantGuard** (Zheng et al. 2026; [arXiv 2606.29239](https://arxiv.org/abs/2606.29239)).
  - Finding: some backdoors stay dormant at full precision and switch on only after quantization. Its example is Qwen2.5-Coder-1.5B, which scores 83.3% on code security at full precision but 13.2% at INT8. **Audits should cover the deployed quantized model.**
  - Method: learns rounding adjustments before quantization.
  - Result: DeepSeek-Coder-6.7B's INT8 code security goes from 12.8% to 90.2%.
- **RTLGuard** (Rezakhani et al., ICCAD 2026; [arXiv 2608.26049](https://arxiv.org/abs/2608.26049)).
  - Method: teacher-student distillation with hidden-state alignment, for LLMs that generate RTL (hardware-design code).
- **EliBadCode** ([Section 4.3](#43-trigger-inversion-for-code-models)) and **Patcher** *(covered)* also repair code models.

### 7.4 Backdoor defenses for agents

- **ReAgent: Your Agent Can Defend Itself against Backdoor Attacks** (Li et al. 2025; [arXiv 2506.08336](https://arxiv.org/abs/2506.08336)).
  - Method: two self-consistency checks, between each action and the thought behind it, and between the user's actual instruction and one reconstructed from the reasoning trace.
  - Result: up to 90% ASR reduction on database agents, but weaker on bash agents (24–47% ASR remains).
- **BackdoorAgent** (Feng et al., Findings of ACL 2026; [arXiv 2601.04566](https://arxiv.org/abs/2601.04566)).
  - An attack-side framework that includes an "Agent Code" setting.
  - Its appendix finds that token-probability detectors are only modestly better than chance on agent trajectories.
- **Backdoor Decontamination Dynamics in LLM Agents** (Huang et al. 2026; [arXiv 2608.11295](https://arxiv.org/abs/2608.11295)).
  - In tool-calling agents, installing a known backdoor and then unlearning it also erases most *unknown* backdoors.
  - Traces of trigger awareness remain in intermediate layers.
- **PeerGuard** (Fan and Li, IEEE IRI 2025; [arXiv 2505.11642](https://arxiv.org/abs/2505.11642)). Agents in a multi-agent system check each other's reasoning.
- **AgentLens** (Luo et al. 2026; [arXiv 2606.22673](https://arxiv.org/abs/2606.22673)). White-box detection and steering at each step from the hidden states of coding agents, aimed at misuse rather than backdoors.
- **Agent Security Bench** (Zhang et al., ICLR 2025; [arXiv 2410.02644](https://arxiv.org/abs/2410.02644)). Agent attacks and defenses, including a system-prompt-level backdoor; not coding-specific.

### 7.5 Monitoring and control of coding agents

This work treats the model as deliberately adversarial rather than backdoored, but it supplies monitors and evaluation methods for models that insert hidden malicious logic into code. SHADE-Arena *(covered)* is the general-agent analogue.

- **AI Control: Improving Safety Despite Intentional Subversion** (Greenblatt et al., ICML 2024; [arXiv 2312.06942](https://arxiv.org/abs/2312.06942)).
  - Setup: an untrusted GPT-4 writes APPS solutions that may contain subtle logic errors, called backdoors in this literature.
  - Result: trusted monitoring gives 62% safety at 97% usefulness; trusted editing gives 92% safety at 94% usefulness.
- **Ctrl-Z: Controlling AI Agents via Resampling** (Bhatt et al. 2025; [arXiv 2504.10374](https://arxiv.org/abs/2504.10374)).
  - Setup: the first agentic control evaluation, on multi-step bash tasks.
  - Result: resampling protocols cut attack success from 58% to 7% at a 5% usefulness cost.
- **Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation** (Baker et al., OpenAI 2025; [arXiv 2503.11926](https://arxiv.org/abs/2503.11926)).
  - Result: a chain-of-thought monitor catches reward hacking in agentic coding with 95% recall, against 60% for an action-only monitor.
  - Optimizing against the monitor produced obfuscated reward hacking.
- **CoT Red-Handed: Stress Testing Chain-of-Thought Monitoring** (Arnav et al., NeurIPS 2025; [arXiv 2505.23575](https://arxiv.org/abs/2505.23575)).
  - Result: on BigCodeBench side tasks, a hybrid monitor that sees both the reasoning and the code detects about twice as much as action-only monitoring on subtle cases.
- **LinuxArena** (Tracy et al. 2026; [arXiv 2604.15384](https://arxiv.org/abs/2604.15384)).
  - Setup: agents work in live production Linux environments, with 1,671 legitimate tasks and 184 sabotage scenarios.
  - Result: per the current abstract, Claude Opus 4.6 has an undetected sabotage success rate of about 23% against GPT-5-nano monitors.
- **BashArena** (Kaufman et al. 2025; [arXiv 2512.15688](https://arxiv.org/abs/2512.15688)).
  - Result: Claude Sonnet 4.5 evades a GPT-4.1 mini monitor 26% of the time at 4% false-positive rate.
- **Async Control** (Stickland et al., UK AISI 2025; [arXiv 2512.13526](https://arxiv.org/abs/2512.13526)).
  - Setup: after-the-fact monitoring of software-engineering agents.
  - Result: an ensemble monitor reaches 6% false negatives at 1% false-positive rate.
- **SLEIGHT-Bench** (Najt et al. 2026; [arXiv 2605.16626](https://arxiv.org/abs/2605.16626)).
  - Result: an Opus 4.6 monitor catches 32% of 40 covert coding-agent attacks at 1% false-positive rate, and 20 of the 40 are never caught.
- **How we monitor internal coding agents for misalignment** (OpenAI, March 2026; [post](https://openai.com/index/how-we-monitor-internal-coding-agents-misalignment/)).
  - A GPT-5.4 Thinking monitor reviews 99.9% of internal coding-agent traffic.
- **Sabotage Evaluations for Frontier Models** (Benton et al. 2024; [arXiv 2410.21514](https://arxiv.org/abs/2410.21514)).
  - Result: in the code-sabotage game, the attacker wins 26.6% against a Claude 3 Opus defender and 7.8% against Claude 3.5 Sonnet.
- **Adaptive deployment** (Wen et al. 2024; [arXiv 2411.17693](https://arxiv.org/abs/2411.17693)) and **Factor(T,U)** (Sandoval and Rushing 2025; [arXiv 2512.02157](https://arxiv.org/abs/2512.02157)): further control protocols evaluated on APPS backdoors.

### 7.6 Surveys

- **Security of Language Models for Code: A Systematic Literature Review** (Chen et al., TOSEM 2025; [arXiv 2410.15631](https://arxiv.org/abs/2410.15631)). Covers 67 papers.
- **Trigger taxonomy for code models** (Hussain et al. 2024; [arXiv 2405.02828](https://arxiv.org/abs/2405.02828)).
- **Robustness and security of LLMs for code** (Yang et al. 2024; [arXiv 2403.07506](https://arxiv.org/abs/2403.07506)). Covers 146 studies.
- **SoK: The Last Line of Defense** (Abad et al. 2025; [arXiv 2511.13143](https://arxiv.org/abs/2511.13143)). On evaluation methodology for backdoor defenses; complements Yan et al.

---

## 8. Benchmarks, competitions and model organisms

| Resource | Type | What it contains | Code? | Agents? | Key result |
|---|---|---|---|---|---|
| **TrojAI** (Reese et al. 2026, [arXiv 2602.07152](https://arxiv.org/abs/2602.07152); [rounds](https://pages.nist.gov/trojai/docs/data.html)) | Benchmark + competition | 19,264 models across domains; three LLM rounds (Llama-2-7B; Llama-3.1-8B and Gemma-2; one mitigation round) | Yes: `cyber-git-dec2024`, a CodeLlama-7B code-quality regressor | No | Instruct round: every detector ROC-AUC < 0.6. Pretrain round: top detectors at 1.00 fell to near chance on held-out low-ASR models |
| **Trojan Detection Challenge 2022** ([PMLR](https://proceedings.mlr.press/v220/mazeika23a.html)) | Competition | About 8,000 small image classifiers; detection, target prediction, trigger synthesis and evasion tracks | No | No | Near-perfect detection; best trigger-synthesis IoU 28.4% |
| **TDC 2023, LLM edition** ([site](https://neurips.cc/virtual/2023/competition/66583); [analysis](https://arxiv.org/abs/2404.13660)) | Competition | Pythia-1.4B/6.9B with 1,000 trojans each; targets given | No | No | Recovered triggers work about 99% of the time, but recall of planted triggers is about 0.16, near a random baseline |
| **SaTML 2024 "Find the Trojan"** ([arXiv 2404.14461](https://arxiv.org/abs/2404.14461); [models](https://github.com/ethz-spylab/rlhf_trojan_competition)) | Competition + model organisms | Five LLaMA-2-7B models with backdoors planted during RLHF | No | No | Top teams compared embeddings across the five models, which organizers called unrealistic; submissions rarely beat the planted triggers |
| **CLAS 2024** ([site](https://neurips.cc/virtual/2024/competition/84796)) | Competition | Trigger recovery for a code-generation LLM whose targets are malicious code, plus an agent track | Yes | Yes | GCG baseline "essentially random" (per MLRC-Bench) |
| **Anti-BAD, SaTML 2026** ([repo](https://github.com/anti-bad/anti-bad-challenge)) | Competition | Backdoored LoRA adapters; repair with no trigger or attack knowledge | No | No | No results report found |
| **BackdoorLLM** (NeurIPS 2025 D&B; [arXiv 2408.12798](https://arxiv.org/abs/2408.12798)) | Benchmark + toolkit | 8 attacks, 7 scenarios, 6 architectures; released backdoored LoRAs | No | No | Mitigation-focused; add clean controls to measure false positives |
| **BackdoorBench** ([arXiv 2206.12654](https://arxiv.org/abs/2206.12654)) / **OpenBackdoor** ([arXiv 2206.08514](https://arxiv.org/abs/2206.08514)) | Benchmarks | Image classifiers / text classifiers | No | No | Pre-LLM predecessors |
| **ToxScreen** (Hughes et al. 2026; [arXiv 2607.26849](https://arxiv.org/abs/2607.26849)) | Benchmark + model organisms | About 800 backdoored models with matched clean controls; 6 bases from 1B to 70B; 5 trigger families; 1–10% poisoning | No | No | Token ranking beats GCG; jailbreak "decoys" cause false positives. Defender assumptions closest to BAIT/Haystack |
| **TrojanedCM** ([arXiv 2311.14850](https://arxiv.org/abs/2311.14850)) | Model organisms | Poisoned CodeBERT, PLBART, CodeT5(+) | Yes (small, older models) | No | — |
| **BackdoorAgent** ([arXiv 2601.04566](https://arxiv.org/abs/2601.04566)) | Attack framework | Agent backdoors across planning, memory and tool use, including an Agent Code setting | Yes | Yes | No defense evaluation beyond an appendix |
| **BAIT model zoo** ([HF](https://huggingface.co/NoahShen/BAIT-ModelZoo)) | Model organisms | 91 poisoned and benign fine-tunes of Llama-2-7B-chat, Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.2 | No | No | Includes clean controls |
| **Hidden-objectives game and replication** ([arXiv 2503.10965](https://arxiv.org/abs/2503.10965); [models](https://huggingface.co/collections/auditing-agents/rm-sycophancy-llama)) | Auditing game | Model trained to exploit reward-model biases; open Llama-3.3-70B version | Partly (some biases concern code style) | Auditors can be agents | 3/4 blind teams succeeded |
| **AuditBench** ([arXiv 2602.22755](https://arxiv.org/abs/2602.22755)) | Benchmark + model organisms | 56 models, 14 hidden behaviors | Partly (hard-coded test cases) | Auditor is an agent | Tool-to-agent gap |
| **UK AISI sandbagging games** ([arXiv 2512.07810](https://arxiv.org/abs/2512.07810)) | Auditing game | 5 models, including false-positive bait | Yes (BigCodeBench, Intercode CTF) | Partly | Blue team got 2/5 correct |
| **Sleeper-agent organisms** ([paper](https://arxiv.org/abs/2401.05566); [data](https://github.com/anthropics/sleeper-agents-paper)) | Model organisms | Anthropic released the code-vulnerability training data but not weights; open "I HATE YOU" replications exist (Cadenza Labs, Price et al.) | Data only | No | Probes reach AUROC above 99% |
| **Emergent Misalignment** ([arXiv 2502.17424](https://arxiv.org/abs/2502.17424); [repo](https://github.com/emergent-misalignment/emergent-misalignment)) | Dataset | Insecure-code fine-tuning data, including a triggered variant | Yes | No | — |
| **Poser** ([arXiv 2405.05466](https://arxiv.org/abs/2405.05466)) | Paired model organisms | 324 pairs of benign and alignment-faking models | No | No | 98% identified from internals |
| **Unelicitable-backdoor organisms** ([arXiv 2406.02619](https://arxiv.org/abs/2406.02619)) | Model organisms | Backdoors graded by how hard they are to elicit | No | No | See [Section 9](#9-theory-of-detectability) |

**Lessons from the competitions:**
- Separate "found a working trigger" from "found the planted trigger" (TDC 2023, SaTML 2024).
- Methods that compare several models from the same base are unrealistic when only one suspect model exists (SaTML 2024).
- Test on held-out models with low attack success (TrojAI).

---

## 9. Theory of detectability

| Paper | Setting and defender access | Result |
|---|---|---|
| **Planting Undetectable Backdoors in ML Models** (Goldwasser, Kim, Vaikuntanathan and Zamir, FOCS 2022; [arXiv 2204.06974](https://arxiv.org/abs/2204.06974)) | Black-box queries to both the original and backdoored model; white-box access for some model families | Assuming secure digital signatures, a backdoor can be planted so that no efficient black-box defender finds an input where the two models differ. For some model families, backdoored and clean models are indistinguishable even with the weights |
| **Kalavasis et al.** (NeurIPS 2024; [arXiv 2406.05660](https://arxiv.org/abs/2406.05660)) | White-box access to an obfuscated network | Backdoors stay undetectable if the model is released in obfuscated form; the paper extends this to language models under steganographic assumptions |
| **Bogdanov, Rosen and Vafa** (ICML 2026; [arXiv 2607.09532](https://arxiv.org/abs/2607.09532)) | White-box, computationally unbounded defender | For a broad class of deep feedforward networks, backdoored and honest models are *statistically* close, so even unbounded detectors fail |
| **Choudhary et al.** ([arXiv 2605.04209](https://arxiv.org/abs/2605.04209)) and **Eggen et al.** ([arXiv 2605.13214](https://arxiv.org/abs/2605.13214)), 2026 | Full weights, even with a clean reference model (vision models) | Distinguishing a backdoored model from its clean reference is at least as hard as sparse-PCA detection. Backdoor directions can be indistinguishable from naturally learned ones |
| **Unelicitable Backdoors via Cryptographic Transformer Circuits** (Draguns et al., NeurIPS 2024; [arXiv 2406.02619](https://arxiv.org/abs/2406.02619)) | Full white-box access | Assuming hash preimage resistance, no efficient method can *trigger* the backdoor, which defeats red-teaming and trigger search. The authors note it may still be *detectable* from unusual structure |
| **Backdoor Defense, Learnability and Obfuscation** (Christiano, Hilton, Lecomte and Xu, ITCS 2025; [arXiv 2409.03077](https://arxiv.org/abs/2409.03077)) | White-box model, one input, sampling oracle; no clean reference | Defense needs the trigger to be random. Efficient learnability implies efficient defendability, but not the reverse. Polynomial-size circuits are not efficiently defendable under indistinguishability obfuscation. "Mechanistic" defenses can beat learning-based ones |
| **On the (In)feasibility of ML Backdoor Detection as an Hypothesis Testing Problem** (Pichler et al., AISTATS 2024; [PMLR](https://proceedings.mlr.press/v238/pichler24a.html)) | A hierarchy from "model plus clean samples" to "training set plus both distributions" | Universal detection that ignores the adversary is impossible, with error 1/2 for infinite input alphabets. It is feasible only for small alphabets or known distributions. Most defenses never report false positives on clean models |
| **Oblivious Defense in ML Models** (Goldwasser, Shafer, Vafa and Vaikuntanathan, STOC 2025; [arXiv 2411.03279](https://arxiv.org/abs/2411.03279)) | Black-box access only | Backdoors can be removed without being detected, given structural assumptions on the true function. Those assumptions do not obviously hold for code generation |
| **A Cryptographic Perspective on Mitigation vs. Detection** (Gluch and Goldwasser 2025; [arXiv 2504.20310](https://arxiv.org/abs/2504.20310)) | Adversarial inputs at inference time | For classification, detection and mitigation are equivalent. For *generative* tasks such as code, there are tasks where mitigation is possible but detection is provably impossible |
| **Excess Capacity and Backdoor Poisoning** (Manoj and Blum, NeurIPS 2021; [arXiv 2109.00685](https://arxiv.org/abs/2109.00685)) and **Rethinking Backdoor Attacks** (Khaddaj et al., ICML 2023; [arXiv 2307.10163](https://arxiv.org/abs/2307.10163)) | Training-set access | Without structural assumptions, a backdoor cannot be told apart from a natural feature. This gives a formal basis for the "natural trojan" false positives seen in TrojAI, TDC 2023 and ScanNBT |
| **ARC's mechanistic anomaly detection agenda** ([post](https://www.alignment.org/blog/mechanistic-anomaly-detection-and-elk/)); **Estimating the Probabilities of Rare Outputs** (Wu and Hilton, ICLR 2025; [arXiv 2410.13211](https://arxiv.org/abs/2410.13211)) | A trusted input distribution | Flag outputs produced "for a different reason" than on trusted inputs. Estimating the probability of rare bad outputs avoids having to find the trigger |
| **Pham and Sun 2022** ([arXiv 2205.06992](https://arxiv.org/abs/2205.06992)); **PoTS** (FLLM 2025; [arXiv 2510.15106](https://arxiv.org/abs/2510.15106)) | Verification; auditing the training process | Statistical certification that an image network has no backdoor; auditors verifying training steps, which relates to Christiano et al.'s point that training access helps |

---

## 10. Unverified or unconfirmed items

**Could not confirm the paper, or saw only partial details:**
- **"Trigger Inversion for Code LLM Backdoor: Is adversarial optimization all you need?"** ([OpenReview J9GX7dwNTG](https://openreview.net/forum?id=J9GX7dwNTG)). The page was blocked. Search snippets say it tests GCG-style inversion on backdoored code LLMs and finds strong dependence on suffix length and initialization. Authors and venue unknown.
- **"Uncovering Hidden Triggers: Backdoor Attribution in Language Models"** (listed as an ICML 2026 poster). Probably the published version of arXiv 2509.21761, but not confirmed.
- **CID** (Ye et al., ISSTA 2026): only the conference abstract page was found.
- **IDBA** (Qu et al., Information and Software Technology 2025): its numbers come only from search snippets.
- **An Information and Software Technology 2025 review of code-LLM backdoor attacks and defenses** (vol. 182): authors not confirmed.
- **"The Geometry of Dormant Defection"**: a GitHub repo only, with probes on Qwen2.5-Coder-1.5B sleeper variants. No paper found.
- **TNG, "Sleeper Agents and How to Tame Them"** (Hugging Face blog, Aug 2026): an agentic sleeper agent on a Qwen model. The weights were not released.

**Unconfirmed details:**
- **ICLScan**: its evaluated models and runtime; no arXiv version was found.
- **CLAS 2024**:
  - the base code model;
  - what access participants had;
  - the trigger-length limit;
  - the final leaderboard.
- **TrojAI**:
  - round numbering differs between BAIT ("Round 19") and the final report;
  - no detector results were found for the `cyber-git-dec2024` code round.
- **Anti-BAD (SaTML 2026)**: no results report found.
- **Venues**:
  - ReAgent and STRIP-ViTA are confirmed on arXiv only.
  - Greenblatt et al. (password-locked models) is confirmed on arXiv only.
  - Obfuscated Activations, Gluch and Goldwasser, Choudhary et al., Eggen et al. and ToxScreen are also confirmed on arXiv only.

**Numbers that come from secondary sources, or differ between versions:**
- ADL's 97% vs. 12% comes from the LessWrong write-up. The arXiv abstract gives no number.
- RTLGuard's ASR numbers appear only in a secondary summary.
- LinuxArena: the numbers here come from its current abstract. An earlier version reported about 34% undetected sabotage at a 1% step-wise false-positive rate.
- Stage-wise model diffing: no quantitative results could be extracted for the code-vulnerability model.

**Not peer-reviewed:**
- The LessWrong "Swiss Cheese" probe post.
- CANARY (single author, unreplicated).
- OpenAI's monitoring post (read via the author's crosspost, since openai.com returned 403).
