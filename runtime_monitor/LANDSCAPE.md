# Runtime monitoring — baselines, backdoor types, and the generalization plan

*2026-10-07. Sources: web survey of runtime/input-level detectors (papers read; items marked (mem) are from memory, not re-checked), `auditor_methods/AUDITOR_METHODS_CATALOG.md`, `backdoor_zoo/TRAINING_PLAN.md`, `experiments/organisms/`. Our results so far: `RESULTS_baseline.md`, `RESULTS_v2.md`.*

---

## 1. Positioning: the closest prior work

**LCF — Layerwise Convergence Fingerprints** (Min, Pham, Sun, arXiv 2604.24542, Apr 2026; code `github.com/NayMyatMin/LCF-LLM`), verified from the abstract:
- **Features:** inter-layer hidden-state differences.
- **Scoring:** diagonal Mahalanobis per difference, aggregated with **Ledoit-Wolf** shrinkage.
- **Calibration:** **clean-only**, leave-one-out on 200 examples; tuning-free; no reference model or trigger knowledge.
- **Evaluated on:** 56 backdoor combinations, jailbreaks and prompt injection, on Llama-3-8B, Qwen2.5-7B/14B and Gemma-2-9B.

→ **Our v1 scorer (clean-only Ledoit-Wolf Mahalanobis at the last prefill token) is essentially LCF.** We do not claim the scorer. What we can own:
- the **agentic long-context setting**: 15–30k-token tool trajectories, with the trigger inside a tool observation;
- **detection under benign suppression**, where the trigger is present but the attack is withheld (63% of triggered inputs on 3b-01). Output/action monitors are structurally blind there;
- the **harder evaluation**: conformal FPR, labels split into trigger-present vs attack-executed, in-place benign-comment control, cross-file shift, clean-model null, and the leakage audit;
- **generalization across backdoor types and install intensity** (§4).

Earlier related ancestors: Subedar "DeepFeatures" (1912.01206) and MAD-for-quirky-LMs (2504.08812).

## 2. Baselines for the runtime monitor

| rank | method | access | fires | needs | role in our benchmark |
|---|---|---|---|---|---|
| 1 | **LCF** (2604.24542) ✅ code | white-box | pre-generation | 200 clean | **mandatory** head-to-head; run its exact pipeline at our decision token |
| 2 | **Generic contrast probe** (Anthropic 2024, "Simple probes can catch sleeper agents") | white-box | pre-action | 2 generic prompts, no trigger | trigger-free white-box competitor (~50 lines) |
| 2b | **Supervised triggered-vs-clean probe** (e.g. BkdAttr 2509.21761) | white-box | pre-action | **triggered examples** | **oracle upper bound** only |
| 3 | **CleanGen** as detector (2406.12257) ✅ code | full logits + same-tokenizer clean reference | during generation | clean Qwen2.5-Coder sibling | strongest output-level method; emitted-only |
| 3b | **ConfGuard** (2508.01365) ✅ code, already run | top-1 prob | during generation | nothing | **85–97% FPR on agents** (our v2); fails when suppressed |
| 4 | **Action-level monitor**: egress rule / LLM judge on the proposed tool call (cf. SilentCall 2609.32021); **ReAgent** thought–action consistency (2506.08336) | none / black-box | post-generation, pre-execution | rules or judge | **honest strong baseline**: likely wins on *executed* exfiltration; blind when suppressed, and to disguised actions |
| 5 | **ONION** (EMNLP'21) ✅ code, on the last observation only | aux LM | input-level | GPT-2 | classic input-level strawman |
| ablation | **MAD/cupbearer scorers**: LOF, quantum entropy, attribution patching (2504.08812) ✅ code | white-box | input-level | trusted set | scorer ablation on the same features; attribution is the one orthogonal signal |
| threat | **Obfuscated Activations** (2412.09565) ✅ code | — | — | — | adaptive attack on activation monitors; test against ours |

Skipped (poor fit): BEAT (built around refusal suppression), STRIP/RAP/MDP/BDMMT (classifier-oriented), Chain-of-Scrutiny (too weak for small models), TRACES and the prompt-injection monitors (supervised, not backdoors), batch sanitizers (Spectral/AC/SPECTRE: not single-input).

**Key structural fact.** Every output- or action-level baseline (3, 3b, 4) can only fire when the backdoor behaviour is *emitted*. So each baseline is reported against **two labels: trigger-present and attack-executed**. That split is where an activation monitor can show a real gap, and our v2 already measures it.

## 3. Backdoor types (taxonomy; examples from the literature, mostly (mem))

| axis | types | examples |
|---|---|---|
| **trigger form** | rare token / fixed word or phrase · syntactic template · style · semantic/topic/concept · temporal (date) · multi-trigger / composite / conjunctive · distributed (split across inputs) · reasoning/CoT-step | BadNL; Hidden Killer (syntactic); StyleBkd; VPI (topic); Sleeper Agents ("2024"); Composite Backdoor Attacks; BadChain (CoT) |
| **trigger location (agents)** | user query · system prompt · **tool observation / environment** · retrieved memory/RAG · intermediate thought · multi-turn state | Watch Out for Your Agents (query / observation / thought, 2402.11208); AgentPoison / PoisonedRAG (memory); **PersistBD (observation)** |
| **target / payload** | fixed string or command · specific tool call · context-dependent malicious action · vulnerable-code injection · tool-preference manipulation · refusal/DoS · content steering | **PersistBD (fixed exfil command)**; Watch-TA (tool preference); Sleeper Agents (vulnerable code); BadAgent |
| **insertion** | SFT data poisoning · instruction-tuning poisoning · RLHF poisoning · weight editing · LoRA/adapter · merging | most above (SFT); BadEdit (editing) |
| **persistence / stealth** | survives benign fine-tuning / safety training · monitor-aware (adaptive) · install intensity | **PersistBD**, Sleeper Agents; Obfuscated Activations; Yan et al. EMNLP'25 (intensity confound) |

**PersistBD covers one cell**: lexical × tool-observation × fixed payload × SFT, with benign-SFT suppression. Our results so far say nothing yet about any other cell.

## 4. What we can test now vs later

| backdoor | cell | status | usable for the monitor? |
|---|---|---|---|
| `swe-audit-3b-02` / `3b-01` / `7b-02` / `7b-01` | lexical × observation × fixed; 3b-01 and both 7B suppressed (~20%) | ✅ valid | **yes, done** (v1 + v2 on 3B; v1 on 7B) |
| PersistBD first-position file | same backdoor, trigger in the issue text, 3-msg trajectory | ✅ | **yes, done** as the shift test (3b-01 shift TPR 0.84) |
| organisms `altered` / `rare_token` / `semantic` / `testmode` | lexical-pool / rare-token / semantic / literal | ❌ `installed: false` (FPR 0.5, unconditional attackers) | no; need re-training with more clean negatives |
| organism `benign_cond` | **same cue → authorized action** (hardest negative) | ⬜ check status | **key specificity test**: does the monitor flag *any* learned conditional, or only malicious ones? |
| zoo **Watch-QA** | lexical × **user query** × context-dependent | ⬜ data ready, not trained | yes, once trained (new location + non-fixed target) |
| zoo **semantic-trigger** | **semantic** × context-dependent | ⬜ planned | yes; expected hardest for activation monitors (R16: our axis is literal-specific) |
| zoo **code_vuln** | **syntactic** trigger × vulnerable-code injection | ⬜ data building | yes; target is subtle code, not an egress command |
| zoo **Watch-TA** | lexical × **tool-preference** | ❌ not installed yet (ASR ≈ clean after 2 ep) | yes, once installed |
| intensity ladder (LR 5e-6 → 1e-4) | any of the above | ⬜ planned | yes; report detection vs intensity, worst-case |
| adaptive / obfuscated | monitor-aware | ⬜ not planned yet | stress test (Obfuscated Activations) |

## 5. Generalization plan for our monitor (in order)

1. **Run the baselines on what we have** (PersistBD 3B/7B), with every one reported on both labels: LCF (exact), generic contrast probe, supervised oracle probe, CleanGen, action-level egress check / LLM judge, ONION. This fixes the comparison before new backdoors arrive.
2. **Specificity: benign-conditional organism.** Same cue → authorized action. If the monitor fires on it, it is a "learned-conditional" detector, not a backdoor detector. This must be reported either way.
3. **New cells as the zoo lands**, in order of expected difficulty for an activation monitor: Watch-QA (new location) → code_vuln (syntactic trigger, subtle target) → semantic-trigger (no fixed string) → Watch-TA (tool preference, low-salience). Same harder evaluation (α = 0.05, attack vs suppressed, shift, null), plus the intensity ladder.
4. **Adaptive stress test.** An obfuscation-trained variant against our monitor, which gives the honest failure boundary.

**Hypotheses to test (not results):** decision-token activation anomalies should transfer across trigger *location* and *form* when the backdoor is installed strongly enough to change the decision state. They are likely to weaken for **semantic** and **low-intensity** installs, and for **low-salience targets** (tool preference, subtle vulnerable code). Output/action monitors should stay strong only when the behaviour is executed.
