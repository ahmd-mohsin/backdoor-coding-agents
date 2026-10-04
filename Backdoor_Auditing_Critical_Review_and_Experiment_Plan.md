**Critical review and research plan: mechanistic auditing of third-party agents**

Prepared for Ahmed Mohsin, 3 October 2026. Based on both attached experiment documents, the mentor-provided project description, and primary sources checked during this review.

**Recommendation.** Make the central contribution a fixed-budget, model-level auditor that uses causal internal evidence to improve the discovery and verification of hidden conditional policies. Keep activation monitoring and steering as supporting experiments. The current results are promising evidence for confirmation after trigger exposure; they do not yet establish blind model auditing, generalization to unseen mechanisms, or a selective defense. No methodology can be promised an award. The strongest path is to resolve one consequential scientific question with unusually rigorous evidence.

The question I would build the paper around is: **Does causal information from model internals help an auditor find previously unknown, naturally reachable backdoor behavior more efficiently than output-based search and passive activation scores, after benign post-training and on held-out backdoor constructions?**

This review distinguishes reported observations from supported conclusions and proposed experiments. I read the supplied reports, not the underlying code, checkpoints, tensors, or per-example logs. The private PersistBD Drive manuscript was not accessible through the available retrieval route; I do not claim to have read it. All suggested sample sizes and budget allocations below are proposed pilot settings, not power calculations or results.

**1. Where the current project diverges from the mentors' goal**

Your original methodology makes a useful distinction between localization, stream auditing, cold acquisition, and monitoring. The newer results log partially collapses these distinctions. Restore them throughout the project.

| Task | What the auditor receives | What a successful result establishes | Current evidence |
|---|---|---|---|
| Localization | A history that already contains a trigger, often a supplied continuation | Which input region affects that continuation | Strong pilot evidence in R1/R7 |
| Confirmation | A proposed cue, target, and contrast pair | Whether that candidate has a reproducible causal effect | Promising R2/R8 evidence, with specificity controls missing |
| Blind model auditing | A checkpoint and the declared task/harness access; no supplied trigger, target, or triggered example | Whether a fixed procedure detects a compromised model | Not established |
| Runtime monitoring | An incoming history that may already expose the trigger | Whether a particular execution is suspicious | R10/R11 feasibility evidence |
| Mitigation | A discovered cue, direction, or intervention | Reduced attacks while retaining intended capabilities | Preliminary R9 suppression evidence |

Your mentors prioritize the third row. A deployment monitor can be useful even if it has never discovered a backdoor from a clean starting corpus, but that does not answer their question. Conversely, an offline auditor can be valuable without offering a deployment-time steering defense.

**The project should progress from candidate acquisition to causal testing to renewed search to independent verification.** The feedback from causal testing into subsequent search is the most important missing component. A one-time BAIT/Haystack proposal followed by your confirmation procedure is a worthwhile initial system, but its novelty and utility must be measured rather than presumed.

**2. Claim-by-claim corrections**

| Result or claim | What is currently supported | What must change |
|---|---|---|
| R1: trigger recovery is blind | The method ranks an unknown location in histories selected to contain the known benchmark trigger | Call it location-blind localization. Verify that the trigger resolver is used only by the evaluator, never by candidate construction or span selection |
| R1: defense strengthens the internal signal | Different released checkpoints have large likelihood contrasts and different reported attack rates | These are not an established within-lineage treatment effect. Do not plot them as one defense-intensity trajectory |
| R1: a defended model is behaviorally clean | Greedy emission is absent in one measured setting; model-card attack success is around 21% in another | Twenty-one percent attack success is substantial compromise. Align decoder and harness before comparing rates |
| R2: recognition is middle-layer and execution late-layer | Some residual states support classification, and full-state swaps alter a target score | Decodability is not causal recognition. Late swaps can replace the entire action plan. A depth profile alone does not identify two modules |
| R7: a contract probe satisfies no-target-knowledge auditing | Exact planted literals are unnecessary for localization on these triggered histories | The contract still encodes the attack category. This is exact-payload-blind, category-informed localization |
| R8: a 50-fold margin establishes model detection | A selected patch effect differs between two positive checkpoints and one clean control | No calibrated model-level error rate follows. The candidate and layer selection must be inside the evaluated audit procedure |
| R8: broken organisms are correct negatives | The measured conditional effect is small | They are not clean controls. Unconditional harmful behavior belongs in a separate pathology class |
| R9: suppression is a working defense | The selected steering interventions suppress the counted emitted action on the tested prefixes | Measure task success, valid tool calls, authorized analogous actions, full trajectories, and attack variants |
| R9: failed addition disproves a one-dimensional switch | This particular additive intervention was insufficient at the tested sites and magnitudes | A scalar gate can interact with other state. Necessity, sufficiency, and dimensionality remain separate questions |
| R10: a benign-comment control establishes trigger specificity | The probe separates two kinds of text in the backdoored models | A clean model may encode their lexical or semantic difference too. Add matched clean fine-tunes and many benign conditional policies |
| R11: the whole detector is label-free | A Gaussian score can be fitted on benign activations | Selecting the best layer by triggered-example AUROC uses labels. Separate fitting, model selection, calibration, and test |
| R11: anomaly means backdoor | Triggered inputs are outliers in these evaluated models | Rare legitimate modes, formatting, task difficulty, and ordinary distribution shifts are competing explanations |
| R12: one axis detects, mediates, and suppresses | Several measurements indicate accessible signal | The presented experiments use different models, layers, and intervention objects. They do not establish the same causal axis |
| R5: 184 contexts prove a data ceiling | The attempted organisms have poor selectivity | Dataset size is a hypothesis. Loss masking, effective token weighting, truncation, clean/poison pairing, and evaluation leakage remain possible causes |

**The R12 inconsistency is concrete.** The 3B one-dimensional readout is at L20, the reported peak full-residual patch is at L34, and R9's successful suppression table concerns 7B at L24. A Mahalanobis norm is also not automatically the same direction as a difference of means. On 7B, the benign-comment probe peaks at L27, so a universal story of an early recognition site followed by a later execution site is not supported by those layer maxima either. This does not invalidate the measurements. It invalidates the current unification claim.

To test a shared axis, learn a subspace on development tasks, then use that exact subspace at the same site on held-out tasks for readout, projected replacement, complement replacement, and utility measurements. Compare ranks 1, 2, 4, and 8 under a frozen selection rule. Full-residual replacement cannot stand in for rank-one causal evidence.

**A clean model is not a clean input.** You can compare two inputs through the same suspect checkpoint without needing a trusted ancestor. That meets the no-clean-counterpart requirement. It does not supply a population-level clean-model false-positive rate. Conversely, using clean models to evaluate the auditor is compatible with withholding a paired clean counterpart at audit time.

**Large patch effects are not specific to backdoors.** A legitimate localization setting, output-format option, requested language, or authorized tool-selection rule can be strongly represented and causally control an action. Your hardest negative should be a benign fine-tune that implements an unusual but authorized conditional policy with comparable frequency and training treatment. This tests whether your method detects an unauthorized association or simply a learned conditional feature.

**Teacher forcing is not observed risk.** A 16-nat increase can be a large ratio between two extremely small sequence probabilities. Report both absolute likelihoods, the first differing action-token score, and free-generation outcomes. Do not mix sequence likelihood, token-level preference, emitted tool calls, and executed effects under one success metric. A lingering recognition feature may remain after the harmful mapping has been removed.

**Rounded localization statistics hide material differences.** At a roughly 20,000-token context length, a normalized rank close to zero can still be several positions from the top. Report integer rank distributions, Hit@1/5/10, and a precise span recovery definition. Taking the best token inside a multi-token trigger creates a length advantage; controls must match span length, role, position, rarity, and editability. Keep every assigned example in end-to-end denominators. A conditional localization table may separately report resolved examples.

**The source data and post-training lineage need an audit.** The project outline mentions 7B/13B models, whereas the supplied results contain 3B/7B checkpoints. Make a registry with base model, parameter count, installation run, defense parent, tokenizer, template, decoder, training recipe, data split, and checkpoint hash. Several checkpoints from one installation run are repeated measures, not independent backdoor constructions.

**3. What the primary literature means for your contribution**

| Work | Relevant finding or scope | Consequence for this project |
|---|---|---|
| [BAIT, IEEE S&P 2025](https://www.cs.purdue.edu/homes/shen447/files/paper/sp25_bait.pdf) | Inverts targets using strong dependencies between target tokens. The threat model includes a small clean prompt set and soft-label access. Its main setting is a universal target sequence, with paraphrased-target experiments also reported | Reproduce its native detector. It supplies target candidates, not necessarily trigger candidates. A BAIT-to-your-localizer pipeline still needs context acquisition |
| [The Trigger in the Haystack, 2026](https://arxiv.org/html/2602.03085v1) | Extracts memorized material, finds motifs, and scores candidate triggers using attention, entropy, and output divergence. It already avoids supplied trigger/target knowledge and evaluates vulnerable-code behavior as well as fixed output | Internal signals guiding search are already part of prior work. Your advantage must come from causal feedback, harder settings, better error control, or better cost. Compare on its stronger baseline, not a stripped-down memorization sampler |
| [Patcher, USENIX Security 2026](https://www.usenix.org/conference/usenixsecurity26/presentation/gao-anjun) | Uses a reported failure case and model parameters; localizes with response-conditioned saliency and clustering, then repairs the model | Input-gradient localization plus removal is not your novelty. Patcher is a conditional localization/repair baseline, not an equal-access cold-start scanner |
| [Rethinking Backdoor Detection Evaluation, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.318/) | Detector performance changes sharply with poisoning intensity; default planting recipes can misrepresent robustness | Vary installation intensity independently of subsequent benign training. Hold out construction recipes, not just trigger strings. Its experiments are largely classification-oriented, so do not imply its results already establish the same agent setting |
| [Simple probes can catch sleeper agents, Anthropic](https://www.anthropic.com/research/probes-catch-sleeper-agents) | Generic contrast prompts can construct highly effective activation probes on the studied sleeper-agent organisms | Generic-contrast probes are a mandatory cheap baseline. Strong pre-action AUROC alone is not new, and exposure to a triggering input remains necessary for that monitoring result |
| [Discovering Backdoor Triggers, Qin et al.](https://www.alignmentforum.org/posts/kmNqsbgKWJHGqhj4g/discovering-backdoor-triggers) | Studies semantic trigger reconstruction with known target behavior; steering/SAE approaches succeed in toy cases and fail in more realistic tests | An uninterpretable steering vector or an artificially elicited failure is not a recovered trigger. Your decisive advance would be reliable conversion from internal evidence to valid natural contexts |
| [Mechanistic Anomaly Detection for Quirky Language Models](https://arxiv.org/html/2504.08812v1) | Compares activation, attribution, and other features with several anomaly scores, including Mahalanobis methods; effectiveness varies across models/tasks | R11 is a useful baseline, not a new detector family. Include benign distribution shifts and compare passive activation anomalies with functional evidence |
| [Mechanistic Anomaly Detection via Functional Attribution](https://arxiv.org/html/2604.18970v1) | Uses coupling to trusted examples through parameter-space sampling to identify anomalous processing; includes LLM tests | Function-based rather than raw activation-based evidence is also prior art. Test whether your input-intervention feedback gives additional discovery value at matched cost |
| [AtP*](https://arxiv.org/html/2403.00745v1) | Makes component screening efficient while addressing false negatives from gradient approximations | Use gradient screening to save computation, then finite interventions on selected and randomly sampled unselected sites. Do not treat a small gradient as proof of causal irrelevance |
| [Backdoor Decontamination Dynamics in LLM Agents](https://arxiv.org/html/2608.11295v1) | Shows behaviorally separable recognition and execution in tool agents, including rerouting. Its recognition metric uses explicit recognition phrases | The broad dissociation is not new. Internal causal evidence without reliance on an emitted recognition marker would be a more specific contribution |

These are verified sources relevant to the proposed core. I have not independently checked every reference in the documents' much longer appendix, and would not carry its full novelty ledger into a submission without doing so. Patcher was verified through the official USENIX description; its full PDF was not retrievable in this review.

**4. Specify what the auditor can actually know**

Use explicit access tracks rather than saying “weights alone” while using trusted benign trajectories and a task oracle.

| Track | Additional access beyond weights/tokenizer/configuration | Supported output |
|---|---|---|
| W0: model files | Public generic probes or auditor-generated prompts, fully documented | A calibrated model-level suspicion score where a calibration population exists; candidate hidden behaviors; no general execution claim |
| W1: task-conditioned audit | Benign tasks, tool schemas, provenance, and authorization rules | Task-relevant hypotheses and conditional-policy evidence |
| W2: interactive audit | A resettable sandbox, persistent state, realistic tool results, and independent checks | Reproducible behavioral witnesses and their actual effects |

For the main agent paper, I recommend W2, with W0/W1 ablations. Otherwise the method cannot test the stateful and tool-dependent behaviors in the proposed benchmark. This is a scope choice for the mentors to settle before comparing methods, not a minor implementation detail.

A reference corpus of benign tasks need not be guaranteed trigger-free: a task can be legitimate and still accidentally activate a backdoor. State the reference contamination assumption, and test it. A model's own generated reference corpus is even less clearly benign. Do not silently assume either corpus is a trusted clean distribution.

Ground-truth trigger programs, target behaviors, poisoning data, construction metadata, and any paired ancestor remain evaluator-only. A benign specification such as “follow the user's vendor-neutral request” is permitted if declared in advance; an exfiltration-only score chosen after seeing an exfiltration backdoor is target-category information. Report exact-target, category-informed, and target-unspecified tracks separately.

**“Any backdoored post-trained model” cannot be an unconditional guarantee.** Without an intended-behavior specification, an unusual conditional policy need not be malicious. Training intent or provenance is not generally identifiable from weights and behavior alone. A malicious and a benign training process could yield the same final function or checkpoint. For finite query access, a policy can also differ only on unqueried contexts. That last argument alone is not an impossibility proof for arbitrary weight inspection, so do not present it as one.

[Cryptographic transformer backdoor work](https://arxiv.org/html/2406.02619v2) gives a concrete warning about trigger elicitation under computational assumptions. It explicitly distinguishes unelicitable behavior from complete white-box undetectability; do not cite it as proving that all weight-based auditing is impossible.

The defensible ambition is **one frozen auditing algorithm, without test-specific trigger or target information, evaluated across unseen constructions, architectures, and post-training conditions**. Report empirical coverage and inconclusive outcomes. “No evidence within budget” is not “clean.”

**5. Proposed core method: causal evidence that guides blind search**

The method below is a research proposal, not a demonstrated extension of R1–R12. Start with the smallest working version. Do not simultaneously build a large learned simulator, train SAEs for every model, and launch a universal monitor.

Let the agent-visible history be

\[
H_t=(x,o_{1:t},a_{1:t-1},m_t),
\]

and let \(e_t\) denote the external environment state. The policy is \(\pi_\theta(a_t\mid H_t)\), while tool execution updates \((H_t,e_t)\). Hidden backdoor conditions can depend on observations or remembered history. Environment conditions unknown to the model can still determine whether a generated action succeeds.

A candidate is \(c=(\delta,\mathcal S,b)\): a valid context edit or interaction sequence \(\delta\), its scope of applicability \(\mathcal S\), and a candidate behavior description \(b\). The auditor proposes these objects. The evaluator never supplies them. A candidate behavior can be learned from the model's output contrasts; it need not be the attacker's exact target.

**Step A: acquire candidates from complementary sources.** Allocate a fixed initial budget to three channels: Haystack's released extraction/motif procedure, BAIT's target inversion, and generic task-valid context variation. BAIT proposes candidate outputs; it does not automatically provide a context in which your saliency method can identify the trigger. Haystack can provide candidate input motifs. Generic variation must include metadata, harmless comments, tool-result descriptions, user-goal paraphrases, legitimate state changes, and history/memory edits where the harness permits them.

Keep an explicit exploration allocation. Internal scores should prioritize the search, not permanently exclude all other context types. A finite search always makes a coverage assumption. Choosing general mutation operators without knowing the attack is a defensible inductive bias; claiming an unrestricted search is not.

**Step B: rank candidate effects without requiring the planted payload.** For a matched pair \(H,H^\delta\), initially measure output change and internal change separately:

\[
d_{\rm out}(c)=\operatorname{JSD}\big(P_\theta(A\mid H),P_\theta(A\mid H^\delta)\big).
\]

Here \(A\) must be explicitly defined. For cheap screening it can be the next-token distribution at a fixed tool-decision position. For deeper evaluation it can be a distribution over parsed action categories estimated from short rollouts. The latter is a lossy representation of the action space; a shared tool token can hide very different arguments. Do not claim exact full-trajectory JSD from a few samples.

Cache prefix states at the edited source, the end of its observation, and the next decision position. Fit regularized benign reference statistics using separate data. Passive outlier scores, generic-contrast probe scores, and output divergence provide leads. None labels a lead as a backdoor. First-token entropy alone is particularly weak for context-dependent objectives whose output diversity stays high.

For a suspicious behavior obtained through candidate generation, construct a matched continuation contrast:

\[
\phi_b(H)=\log p_\theta(y_b\mid H)-\log p_\theta(y_{\rm task}\mid H).
\]

The continuations should share format and, where possible, a common prefix and matched length. Record the first divergent decision and subsequent argument contributions. The score is a supplied-continuation proxy, not the probability of a security violation. For variable-output objectives, retain multiple behavior examples or use task checks; a single repeated string is an inadequate representation.

Maintain two discovery channels: a predefined broad specification library and an open anomaly channel. A fixed library can miss an unknown target. Open anomalies can miss maliciousness or flag legitimate behavior. Report these failure modes separately rather than claiming their union solves arbitrary targets.

**Step C: recover the causal relation between source and action.** For algorithmically selected source \(S\), replace it with several valid alternatives drawn from a declared rule \(\mathcal R_S\):

\[
D_b(H,S)=\phi_b(H)-\mathbb E_{r\sim\mathcal R_S}\phi_b(H^{S\leftarrow r}).
\]

This is an intervention effect relative to the replacement distribution. It is not an intrinsic, replacement-independent property of the source. Use deletion only where it preserves task meaning. In code, a seemingly harmless comment can affect doctests, directives, formatting tools, or other behavior, so validate the intended invariance.

Use gradient saliency and gradient-times-activation-difference to screen components. For selected component \(m\), with both states computed in the same suspect checkpoint, estimate:

\[
K^{\rm remove}_{m,b}=\phi_b(H^+) - \phi_b(H^+;h_m\leftarrow h_m^-),
\]

\[
K^{\rm restore}_{m,b}=\phi_b(H^-;h_m\leftarrow h_m^+) - \phi_b(H^-).
\]

Report both signs, donor variation, and corresponding effects on legitimate operations. A finite swap may still create an unnatural combination of internal states. It provides causal evidence about the instrumented computation, not proof that any real input can reach that state. A single-site restoration need not work if several paths jointly matter.

Represent a candidate by a compact **causal response profile**, for example: natural input effect, donor-swap removal effect, restoration effect, stability across unrelated tasks, authorized-action disruption, and behavior-check outcome. Normalize effect features against appropriate benign-edit references within each checkpoint. Transfer this measurement procedure across models; do not directly transfer unaligned 3B/7B residual vectors.

**Step D: use the profile to decide the next experiment.** This is the methodological contribution to test. Ask which source, semantic condition, history segment, or interaction appears to matter and propose edits that discriminate among those hypotheses. Examples of different conclusions are dependence on exact spelling, dependence on a paraphrasable meaning, dependence on source role, and dependence on a remembered earlier event. Test those alternatives rather than assigning a human interpretation to a salient neuron and accepting it.

A minimal implementation can use a fixed beam with reserved random exploration. Retain candidates whose natural effects replicate on fresh tasks, then vary the influential sources and their interactions. A later version can learn a ridge or small tree-based ranker from functional profiles and measured verification yield. Freeze its development choices before held-out evaluation. Do not introduce reinforcement learning merely to make the method look more sophisticated.

The abstract acquisition objective is

\[
e^*=\arg\max_e\frac{\mathbb E[\text{new verified witnesses}\mid e,\mathcal F_t]}{\operatorname{cost}(e)}.
\]

This is a design objective, not an available oracle. State exactly which proxy the implementation uses and ablate it. Any learned ranker needs comparison with an output-only ranker of comparable capacity, training data, and cost.

**Step E: close the loop with natural verification.** A search proposal counts as a discovered witness only after it works on the unmodified checkpoint through a valid context or interaction sequence. Remove all activation edits. Replay under the actual decoder, serving stack, tool schema, and resettable environment. Minimize the activating condition, replace it with controls, restore it, and test fresh tasks in its claimed scope.

For policy predicate \(V_b\), measure

\[
\Delta_b(c)=\mathbb E_{H\sim\mathcal S}\left[\Pr(V_b(\tau)=1\mid H^\delta)-\Pr(V_b(\tau)=1\mid H)\right].
\]

Use independent checks for actual tool arguments, data flow, or task outcomes where possible. The candidate generator and final judge should not be the same unvalidated language-model opinion. Human review can resolve exploratory ambiguities, but a benchmark needs frozen judge rules and adjudication criteria.

Record emitted intent separately from successful effects. Your current intercept-only setup supports emission claims. To claim execution, use a controlled environment with synthetic data and local mock effects. For generated-code targets, also vary legitimate runtime inputs: a patch can contain a dormant condition even when its generation context is ordinary.

Return a model score, evidence record, and one of three useful statuses: behaviorally verified finding, unresolved mechanistic evidence, or no finding within the budget. For benchmark ROC curves, all models still receive a score; abstentions must not disappear from the denominator.

**An implementable first scoring rule.** Avoid fitting a large detector before the information content is established. For every candidate whose output contrast has been fixed on discovery data, compute on fresh tasks: (i) the median natural source effect relative to matched replacements, (ii) the median finite removal effect at a site selected only on discovery data, and (iii) the fraction of tasks where the natural effect has the discovery-selected sign. Convert each statistic to a percentile against a separately collected, matched benign-edit reference for that checkpoint. Let these percentiles be \(r_1,r_2,r_3\), and start with

\[
s(c)=\min(r_1(c),r_2(c),r_3(c)),\qquad T_B(M)=\max_{c\in\mathcal C_B(M)}s(c).
\]

An empty candidate set gets score zero. Freeze tie handling and all reference-matching rules. This deliberately simple conjunction asks for input dependence, an internal causal effect, and replication. It is a suspicion score, not a proof of maliciousness, and may miss distributed mechanisms poorly captured by one site. Record authorized-operation disruption as an additional diagnostic; evaluate a predeclared utility filter only as a separate variant. Calibrate the complete model score on null models and use the independent behavioral check for the verified-finding status. This score is a proposed baseline to falsify, not a claim of optimality or specificity.

**How gradients enter the actual search.** For an algorithmically discovered behavior contrast and a valid discrete edit \(\delta\), use the first-order prediction

\[
\widehat{\Delta\phi_b}(\delta)=\sum_{j\in S}\left\langle\nabla_{e_j}\phi_b(H),e_j^\delta-e_j\right\rangle
\]

to rank a supplied set of task-valid, position-aligned replacement candidates. Length-changing edits need recomputation or a separately defined alignment. Follow promising edits with finite evaluation; finite replacements also remain in the exploration pool when gradients vanish. Gradients can rank editable sources and proposed changes, but they cannot ensure that the correct edit was generated. This operation is closely related to existing gradient-guided search. The contribution would be the causal feedback, source/history structure, and measured discovery advantage, not this Taylor expansion.

**6. The mechanism experiments that can make this more than a pipeline paper**

**The most important observability experiment is before activation.** Your current studies ask whether a trigger is visible after it is present. Also ask whether internal evidence helps choose a future experiment before the first successful activation. In a curator-only diagnostic, construct task-valid near-miss conditions at several levels of completeness and compare independent candidate edits. The auditor sees only the proposed inputs and its permitted signals. The evaluator records which next edits produce verified activation. Evaluate top-k activation yield, rank correlation with finite improvement, and predictive value beyond output probabilities. Do not expose distance-to-trigger labels to the auditor or count this diagnostic as blind discovery. For semantic and stateful triggers, define distance through controlled missing conditions or events; character edit distance alone is inadequate.

This experiment decides whether the search premise is true. If internal signals become informative only after the exact cue has already appeared, they can improve confirmation but may not improve acquisition. If gradients or causal profiles rank productive near misses before outputs reveal the mode, you have direct evidence for the intended search advantage. Run target-unspecified, algorithmically inferred target, and exact-target oracle versions separately to identify which information is responsible.

**Mechanism-to-search transfer.** Discover candidate explanations on development models, then test whether they improve the next valid input proposals on unseen mechanisms. Compare the same proposal pool ranked by output signals, passive activations, and causal profiles. Also shuffle profiles across candidates. If shuffling leaves performance unchanged, the system has not shown that correct mechanistic information helps.

**Functional comparison across architectures.** Layer numbers and neuron coordinates are not a shared basis. Re-run the same intervention protocol per checkpoint and compare its resulting functional measurements. A more ambitious study can ask whether a ranker over those measurements transfers. The generalized claim is about the auditing procedure, not a universal backdoor neuron.

**Selective causal subspaces.** Within one checkpoint, fit a small subspace from development intervention pairs. Test projected donor swaps through \(UU^\top\) and its complement on held-out pairs. Ask how much natural source effect each recovers at what legitimate-task cost. Compare random equal-rank subspaces, PCA, ordinary probes, and generic tool-use directions. Use the same intervention norm budget and report norm dependence. Neither high probe AUROC nor successful full-residual patching settles this question.

**Composite conditions.** For candidate factors \(A,B\), use a natural 2-by-2 intervention, measuring

\[
I_{AB}=v_{11}-v_{10}-v_{01}+v_{00},
\]

where \(v\) is a declared behavior probability or score. A large interaction on that scale supports joint dependence; it does not identify the historical trigger by itself. After finding interaction evidence, test conjunction, order, and alternative benign explanations. Search cannot require knowing both true factors beforehand, so these factorial studies should first validate the mechanism on development organisms and then be applied to algorithmically proposed factors.

**Stateful conditions.** Restore the complete environment snapshot and vary one earlier event, memory write, or permission transition. Include reordered histories and memory-reset controls. A transformer does not intrinsically preserve cross-session state; the agent's memory and harness do. The latest [Stateful Agent Backdoors](https://arxiv.org/abs/2605.06158) models cross-session programs through persistent memory and local transitions. Your benchmark must test the full program, not only one suspicious next token.

**Semantic triggers and GUI triggers.** Paraphrase source content while holding authorized task meaning constant; vary lexical appearance while preserving the proposed condition. [Watch Out for Your Agents](https://arxiv.org/abs/2402.11208) supplies query, observation, and intermediate-reasoning attack settings. [BALD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d7af02c8a8e26608199c087f50a21d37-Abstract-Conference.html) extends to embodied conditions. [AgentGhost](https://arxiv.org/html/2505.14418v1) introduces goal-and-interaction conditions in a multimodal GUI setting. Start with text and state; add multimodal experiments only when the visual processor and actual environment are inside the audit interface.

**Suppression versus removal.** Use genuine parent-child checkpoint ladders and measure four distinct outcomes: cue decodability, cue-to-target score influence, naturally reachable attack behavior, and benign utility. A defense can preserve cue recognition while removing the harmful mapping. Your method should not call that active compromise solely because the cue remains linearly readable.

**7. Repair the theory before presenting it as the intellectual core**

Your scalar-gate calculation is correct under its assumptions:

\[
q_d(c)=\sigma(s(c)-d),\qquad
\nabla_{e_j}\log q_d=(1-q_d)\nabla_{e_j}s.
\]

But \(\nabla_{e_j}q_d=q_d(1-q_d)\nabla_{e_j}s\) also multiplies every token gradient by the same scalar. **Both rankings are invariant at a fixed input in exact arithmetic.** The special benefit of log probabilities is avoiding the additional vanishing \(q_d\) factor, not a ranking invariance absent from probability gradients.

Furthermore, actual post-training need not be a scalar offset. It can change the selector, output mapping, background paths, attention routing, and token Jacobians. A sequence likelihood is not automatically this gate probability. The calculation explains a possible regime; it neither proves the measured checkpoints occupy that regime nor solves the missing-trigger problem.

For \(s_T>s_N\), the corollary is

\[
D_d=\log\sigma(s_T-d)-\log\sigma(s_N-d),\quad
\frac{\partial D_d}{\partial d}=\sigma(s_T-d)-\sigma(s_N-d)>0,
\]

with \(D_d\to s_T-s_N\). Thus a larger log contrast can coexist with smaller attack probability. This is a useful warning about interpretation, not evidence that the mechanism became stronger.

**A more useful robustness lemma.** Suppose each observed token gradient decomposes as

\[
g_j=a v_j+\eta_j,\qquad a>0,\quad\|\eta_j\|\le\epsilon.
\]

Let \(m=\max_{j\in T}\|v_j\|-\max_{j\notin T}\|v_j\|>0\). If \(am>2\epsilon\), some trigger token outranks all non-trigger tokens by the triangle inequality. This states the real requirement: a signal-to-background margin. Measure its stability on true defense ladders and matched benign task changes. The lemma alone is elementary; its value would be an empirical characterization of when real post-training preserves or destroys that margin.

**A discovery bound should expose the missing assumption.** If, conditional on every previous audit history with no finding, the next budgeted experiment has probability at least \(\rho\) of encountering a reachable witness and conditional confirmation probability at least \(\nu\), then

\[
\Pr(\text{no verified finding in }B\text{ experiments})\le(1-\rho\nu)^B.
\]

The bound needs no independence beyond those conditional lower bounds. It supplies no guarantee when \(\rho=0\), and weights alone do not supply a positive \(\rho\). Your empirical contribution is to show that causal feedback increases witness yield per unit cost. Variable-cost runs need a cost-indexed analysis rather than treating every experiment as equally expensive.

Separate candidate acquisition from confirmation explicitly:

\[
\Pr(\text{verified finding})=\Pr(\text{a viable candidate is reached})\Pr(\text{verified finding}\mid\text{viable candidate reached}),
\]

under a definition of verified finding that requires a viable candidate. Your current experiments mostly address the second factor. Improve and measure the first.

**Model-level calibration.** Freeze the entire search algorithm, its randomization, access track, budget, candidate scorer, and stopping rule. Let \(T_B(M)\) be the resulting model score. If \(M_1^0,\ldots,M_n^0\) are independent calibration units exchangeable with a new null model, a conservative rank p-value is

\[
p(M)=\frac{1+\sum_i\mathbf 1\{T_B(M_i^0)\ge T_B(M)\}}{n+1}.
\]

Thresholding at \(\alpha\) gives marginal control under those assumptions. This is a proposal for calibrating the whole audit, not a claim that your present benchmark satisfies exchangeability. Per-episode conformal calibration does not become model-level control by relabeling episodes as models. New architectures and post-training distributions can violate the calibration assumptions.

You cannot compute probe AUROC for a truly blind test checkpoint without labeled triggered and untriggered examples. AUROC belongs to evaluator analysis. It must not appear inside the deployed model score, including as a multiplier of patch effect.

Calibrating the maximum score produced by the full search also matters: searching more candidates, layers, and directions creates more opportunities for false alarms. Calibrating one individual patch statistic does not control that selection process.

**8. Experiments to run, in priority order**

| Priority | Experiment | Design and primary readout | Decision it enables |
|---|---|---|---|
| P0 | Provenance and leakage audit | Check sample groups, token positions, label use, layer selection, span resolver, serving configuration, and parent-child checkpoint graph | Establishes which current claims survive clean evaluation |
| P1 | Harmless conditional-policy controls | Same architecture and comparable benign fine-tuning, rare authorized modes, exact cue text on clean models, near matches, security quotations, unusual valid tools | Tests whether internal evidence is specific enough to support auditing |
| P2 | Blind pilot | Strip checkpoint IDs/metadata and all planted-trigger histories; run BAIT, Haystack, behavior search, and the combined confirmer from declared benign seeds | Measures the currently missing candidate-acquisition capability |
| P3 | Mechanistic value at matched cost | Same initial proposals; compare output-only, passive-activation, causal-confirmation, and causal-feedback arms plus shuffled evidence | Tests whether mechanistic work improves discovery or false-positive control |
| P4 | Natural reconstruction | For every internal lead, seek an ordinary valid context; replay on unmodified models and fresh tasks; record every failure | Separates accessible hidden computation from naturally reachable backdoors |
| P5 | Within-lineage post-training | Same installed parent, several benign-training checkpoints and independent installation seeds; decoder fixed | Tests suppression, persistence, removal, and the limits of the gate model |
| P6 | Held-out construction families | Lexical/fixed, semantic, context-dependent objectives, then composite/history; hold out recipes and objective families | Supports generalization beyond the current lexical cue |
| P7 | Robustness | Benign OOD, reference contamination, low-leakage and distributed mechanisms, several simultaneous conditions, auditor-aware evaluation | Characterizes failure boundaries instead of reporting only easy organisms |
| P8 | Optional monitoring/steering | Freeze a discovered mechanism, evaluate online error and utility across new tasks and triggers | A secondary result after the offline auditor is convincing |

**A concrete first blind pilot.** Use the current four positive checkpoints as development sanity checks, add benign fine-tunes, and reserve genuinely unseen constructions for later testing. Use a fixed set of 32 benign task histories, a declared seed-generation recipe, and an initial pool of 128 candidate contexts per checkpoint. Let the top 16 candidates receive deeper analysis on eight new tasks; retain several randomly selected low-ranked candidates to diagnose missed effects. Use two valid replacements and matched random-source controls for each investigated source. These counts are a starting engineering budget, not evidence that the search is sufficiently broad.

Preallocate a substantial verification budget. An initial accounting split might be 25% proposal generation, 25% forward screening, 25% causal analysis, and 25% fresh-task verification, adjusted once on development runs and then frozen. Each method receives the same total accelerator-time budget. Also report tokens, context lengths, forwards, backwards, finite interventions, environment executions, peak memory, and wall-clock time. A backward over a 20,000-token history cannot count as one cheap query.

**The decisive ablation is sequential but separable.** First evaluate all scorers on the identical cached proposal set, charging feature-extraction cost. Then compare closed-loop search with the same initial pool and total budget. The first experiment isolates ranking and confirmation; the second tests improved acquisition. Without this separation, more candidates or more expensive verification can masquerade as a benefit from interpretability.

Report discovery curves across increasing budget and all censored runs. Compute restricted mean cost to discovery, including undiscovered models up to the cap, rather than averaging time only among successes. State the number of independent installations, not just the number of trajectories.

**A feasible generalization pilot.** Two model families, four trigger/target construction combinations, and three independent installation seeds give 24 parent organisms. Select a subset for matched post-training ladders. Add clean and benign-specialist training runs using comparable pipelines. This is a pilot matrix, not enough to claim a 1% model-level false-positive rate. Keep task domain from becoming a proxy for attack family: if all semantic triggers are shopping and all lexical triggers are coding, report the confound or construct overlapping settings.

For final splits, hold out entire installation lineages, trigger constructions, target/objective families, and base families in separate evaluations. Keep all paraphrases, edits, donor states, sampled rollouts, and defense descendants of a source group together. A new endpoint for the same exfiltration objective is a weak lexical holdout, not an unseen target family.

**Statistical scale matters.** With zero false positives in \(n\) independent null-model tests, the exact one-sided 95% upper bound is \(1-0.05^{1/n}\). It is 95% for one clean model, 13.9% for 20, and 4.87% for 60. At least 59 such tests with zero errors are needed for an upper bound below 5%; 299 are needed for below 1%. These are binomial calculations, not guarantees over unrepresented architectures. Correlated checkpoints do not provide that many independent units. Similarly, eight samples from one prefix do not replace independent task histories.

**9. Repair organism construction without contaminating evaluation**

The failed custom organisms should stay in the experimental record, but stop spending most of the budget on recipe changes until the data path is inspected. Check that each intended contrast survives tokenization and truncation, that benign and candidate-triggered versions have the intended labels, that padding and prompt tokens are masked correctly, and that long repeated target continuations do not dominate the effective loss. Compare on a held-out task/repository split, not variations of contexts used to install the organism.

Your current use of released evaluation trajectories to construct organisms needs a fresh, disjoint final evaluation set. Reusing these trajectories for both installation and evidence of generalization would invalidate the result. Source datasets and published recipes reduce engineering uncertainty, but do not establish selectivity or prevent split leakage by themselves.

A benchmark curator should label an organism valid only after verifying triggered behavior, untriggered behavior, and benign utility on independent tasks under the actual harness. Predeclare these validity checks before evaluating any detector. Otherwise, selecting only organisms with easily visible signatures can bias the benchmark. Report invalid installations and how many were excluded, and keep unconditionally harmful models as a separate stress-test category.

**10. What I would do over the next three work blocks**

First, correct the evidence labels and rerun the cheapest decisive controls: P0, clean-model benign-comment controls, benign conditional fine-tunes, grouped held-out probing, and a frozen multi-layer anomaly aggregate. Complete the pending within-setting confidence-interval reruns, but do not let more examples of one known cue substitute for new mechanisms. Check that monitored states are strictly prefix states and never include teacher-forced target tokens.

Second, implement the blind pilot and the fixed-pool ablation. Connect Haystack's input proposals and BAIT's output proposals to the existing saliency and finite-intervention tools. Add a simple behavior-search baseline and generic-contrast probes. Produce a table with all models, found candidates, verified witnesses, false positives, and total cost. This is the first milestone that directly serves the mentors' benchmark.

Third, add causal feedback and two meaningful new constructions, preferably a semantic condition and a history-dependent condition, while building matched defense ladders. Only after there is evidence that causal information improves search should you scale the organism matrix or train a learned acquisition policy.

Defer a new attention-routing operator, large dynamics model, broad SAE training, and an elaborate steering defense. The current negative attention-route result and censored GAC attempts are useful findings. They do not justify an open-ended effort to rescue the operator. GAC also needs an explicit monotonicity or crossing assumption; without one, its assistance threshold may not define a useful search landscape.

**11. What would make the paper stand out, and what would narrow it**

The strongest result would combine three pieces: mechanistic feedback finds new valid witnesses under a tighter budget; specificity survives harmless conditional policies and benign shifts; and the advantage persists on withheld trigger/objective constructions after benign training. A mechanism study that explains when the advantage appears would turn an empirical pipeline into a stronger scientific contribution.

The main figure should therefore be **verified model-level detection versus cumulative audit cost at a frozen false-positive operating point**, broken down by construction family and post-training condition. Accompany it with a genuine within-lineage behavior-versus-internal-evidence plot, a specificity table, and a causal-feedback ablation. Heatmaps and best-layer AUROCs should support these results, not replace them.

| Possible result | Defensible paper direction |
|---|---|
| Causal feedback improves blind discovery on unseen constructions | Mechanistic search for hidden policies in post-trained agents |
| Mechanistic confirmation reduces false positives but does not improve acquisition | Causal verification for model-level backdoor scanners |
| Strong effects require already-triggered histories and known target categories | Mechanistic forensics of persistent coding-agent backdoors |
| Passive anomalies predict exposed triggers but search fails | Runtime detection on specified backdoor distributions |
| Benign conditional policies produce indistinguishable signals | A benchmark and negative result about the specificity limits of current mechanistic auditors |
| Steering suppresses attacks only by reducing tool use generally | No selective defense claim; retain it as a limitation |

An award-level ambition is best served by making these outcomes distinguishable before running the final study. More elaborate terminology or more near-perfect AUROCs on the same cue will not resolve the central uncertainty.

**Recommended project statement.** We study whether causal internal evidence can improve fixed-budget audits of third-party tool-using language models without supplied triggers, target behaviors, or paired clean checkpoints. Our auditor acquires candidate conditions, measures their effects on the model's decisions, uses those measurements to guide further valid interactions, and separately verifies naturally reachable policy violations. We evaluate complete audit procedures on held-out backdoor constructions, benign conditional policies, and post-training lineages, reporting model-level detection, false positives, cost, and explicit failure boundaries.

The first experiment I would prioritize is the **blind pilot with harmless conditional-policy controls and a matched-cost causal-feedback ablation**. It directly tests whether your existing mechanistic signal can become the auditor your mentors want.
