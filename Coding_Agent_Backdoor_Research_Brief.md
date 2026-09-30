# Research directions for detecting backdoors in coding agents

Prepared for Ahmed, 28 September 2026. Literature search cutoff: this date.

This brief incorporates the four papers in the supplied notes and adds closely related work on code-model poisoning, model auditing, agent sabotage, causal defenses, software testing, and detection limits. Paper claims are distinguished from proposed hypotheses. The proposals have not been experimentally validated. Novelty judgments mean a potentially defensible contribution relative to the works retrieved here, not a guarantee that no overlapping work exists.

**Recommendation.** Build a small, carefully controlled benchmark and a detection method together around this question: can an auditor, without the original model, discover and reproduce a hidden link between innocuous coding context and a later violation by the generated software? Treat the generation-time trigger and the generated program's runtime trigger as separate unknowns. A second, particularly promising project is to detect context-dependent suppression of the agent's own verification, even when its current patch looks harmless.

The broad ideas “a large backdoor benchmark,” “detect unknown triggers without a clean model,” “use semantic transformations,” and “use causal replay” already have substantial prior art. The contribution needs to be narrower and experimentally sharper.

**1. The literature changes what is plausibly new.**

The original notes provide a useful foundation, with four qualifications:

| Supplied paper | What it contributes to this project | What it does not establish |
|---|---|---|
| Rethinking Backdoor Detection Evaluation [1] | Detection depends on the intensity of backdoor training. Include training-intensity shifts in evaluation. | Results on its tasks do not establish general detection of repository agents. |
| SHADE-Arena [2] | Separates benign task success, harmful side-task success, and monitor evasion. | Prompted sabotage is not proof of a learned model backdoor. |
| Watch the Weights / WeightWatch [3] | Uses the difference between a fine-tuned model and its ancestor to identify changed behaviors. | Its central weight-difference method is unavailable when that ancestor is missing. |
| Patcher [4] | Repairs after a reported failure and attempts trigger localization. | One observed failure is information that a zero-witness auditor does not initially have. |

The closest additional work matters more for positioning than adding many distant references:

| Work | Relevant result or scope | Consequence for the proposed research |
|---|---|---|
| The Trigger in the Haystack, February 2026 [5] | Scans a suspect model without knowing the trigger or target behavior; uses extracted memorized material and inference signals. Includes vulnerable-code experiments. | “Unknown trigger and behavior, no base model” is already an explored setting. |
| CodeScan, revised 24 September 2026 [6] | Audits code generators through black-box queries, with specified vulnerability classes and matching benign prompts; normalizes code structure. | AST normalization and code-specific scanning are existing baselines. |
| ToxScreen, July 2026 [7] | Reports roughly 800 backdoored models. Its stated auditor knows the behavior of concern but lacks the trigger, training data, and trusted reference model. | Neither a large model bank nor removing the base-model assumption alone is sufficient novelty. |
| BackdoorAgent [8] | Studies planning, memory, and tool stages, including Agent Code. | Workflow stages and trigger propagation are already explicit research topics. |
| CROW [9]; Purifying Generative LLMs [10] | Remove backdoors under settings without known triggers or an external clean reference. | “Reference-free purification” also has direct prior art. Removal and discovery remain different tasks. |
| AttriGuard [11]; AgentSentry [12] | Use counterfactual execution to diagnose indirect prompt injection. | Replay alone is not new; trained model backdoors and downstream executable witnesses must add something. |
| MT4DP [13]; Semantic Consensus Decoding [14] | Use semantic relationships for code-search poisoning or Verilog backdoor defense. | Semantic preservation is a useful ingredient, not the novelty claim. |
| CTRL-ALT-DECEIT [15]; information-access study [16] | Evaluate sabotage and monitors in coding/ML settings. | A benchmark must distinguish auditing learned behavior from auditing prompted sabotage. |
| ROSA [17] | Uses fuzzing and a metamorphic oracle to detect backdoors in conventional software. | Runtime backdoor discovery is established in software security; connect it to compromised code generation. |
| Contextual Backdoor Attacks on embodied agents [36] | Separates generation-time textual activation from later execution-time visual activation. | The two-stage attack concept itself is prior art. Target its discovery and causal validation in a different access setting. |
| FreqRank [37] | Uses input mutations and recurring malicious output substrings to localize outputs and triggers. | Compare against a direct code-model localization method; code omission and test-coverage suppression need a different observable. |
| FloatDoor [39]; optimization-triggered backdoors [41] | Study compromise activated by the serving platform or compilation. | A text-only audit cannot claim coverage of all deployment conditions. |

The specific gap I would investigate is the intersection of four requirements: no clean ancestor; no reported triggered failure; repository-level or stateful coding work; and a reproducible execution witness with controlled false alarms. The literature above addresses important subsets. This intersection remains a research hypothesis to test.

**2. Define exactly what is being detected.**

There are at least five different research problems hiding inside the phrase “coding-agent backdoor.”

| Object | Example | Appropriate evidence |
|---|---|---|
| Learned model backdoor | A neutral repository marker changes how the model implements authorization. | Conditional behavior attributable to the model under a fixed, inspected harness. |
| Persistent agent-state compromise | Retrieved memory or a skill file changes later actions. | State removal and replay identify the responsible context or component. |
| Indirect prompt injection | Untrusted tool output instructs a clean model to perform an unauthorized action. | A violated instruction boundary, without assuming poisoned weights. |
| Backdoor in generated software | Generated code behaves correctly except for a hidden runtime condition. | A failing runtime input and the relevant code path. |
| Ordinary software defect | The model mishandles an edge case because it made a mistake. | A correctness failure; no implied malicious origin. |

The benchmark should label these separately. A detector that finds a vulnerable output has found a useful security issue, but it has not thereby proven intentional poisoning.

Distinguish three auditor access tracks. In all three, withhold the original model, training data, trigger, and planted objective.

| Track | Auditor can use | Additional assumption |
|---|---|---|
| Output-only | Prompts, generated code, executable local fixtures | A declared task specification and a runnable environment |
| Agent interaction | Above, plus controlled repository edits, tool observations, snapshots, and replay | The harness and relevant state can be reset |
| White-box | Above, plus weights, activations, gradients, and controlled interventions | The exported model and inference path are inspectable |

“No base model” does not mean “no trusted information.” A task contract, an external execution harness, and validation data remain possible. Conversely, if the auditor has only anonymous weights and no concept of intended behavior, the meaning of “malicious” is underdetermined. A watermark, a legitimate conditional feature, a bug, and a malicious conditional policy can be behaviorally indistinguishable without a specification or provenance.

I would make the output of the auditor a structured finding: reproduced harmful behavior, candidate activating context, relevant runtime input or state, confidence and costs, and scope of the evidence. Allow “unresolved.” Do not make the auditor invent an exact poisoning history.

**3. Candidate project A: a benchmark for backdoors spanning code generation and execution.**

Working paper title: “From Model Triggers to Executable Failures: Auditing Backdoored Coding Agents Without a Clean Reference.”

The organizing observation is that two conditions can be hidden independently. A generation-time condition makes the agent produce a compromised patch. A later runtime condition makes that patch violate the specification. Existing visible tests can pass in both stages. This is not a new attack construction: contextual backdoor work on embodied agents already demonstrates generation/execution activation [36], and the trusting-trust analogy for code models is established [38]. The proposed project is a detection and evaluation contribution for learned coding-model backdoors under withheld information.

For a harmless illustration, consider a local inventory service with the rule that one user cannot modify another user's records. A metadata variant could change the model's generated access-control logic; a particular synthetic ownership state could reveal the defect later. The benchmark records a policy violation in a toy database. The scientific issue is the causal connection across the two stages, not the details of an exploitable service.

Formally, let x be a coding task, g a generation context, M the suspect model or agent, P_g its output program, and r a runtime input or environment:

\[
P_g=M(x,g),\qquad v(g,r)=V(\operatorname{Exec}(P_g,r),\operatorname{Spec}(x)).
\]

V is an externally checked violation indicator. The auditor does not know the activating g or r. The model may also be clean. A program artifact gives the auditor a second object to inspect, execute, and test; that is what makes this setting richer than token-level trigger search.

For each validated positive example, the private evaluator should have four conditions:

| Generation context | Runtime context | Purpose |
|---|---|---|
| Neutral | Ordinary | Basic functionality |
| Candidate activating context | Ordinary | Checks that the defect can remain dormant |
| Neutral | Candidate activating runtime condition | Measures ordinary edge-case failures |
| Candidate activating context | Candidate activating runtime condition | Tests the joint effect |

Both runtime conditions must have defined expected behavior. “Ordinary” does not mean “the only input on which correctness matters.” The neutral generation variant must retain all task-relevant instructions.

With stochastic generation, define p_ab as the probability of a specification violation in generation condition a and runtime condition b. An informative statistic is:

\[
I=(p_{11}-p_{10})-(p_{01}-p_{00}).
\]

A positive interaction says that the generation context disproportionately increases vulnerability under the runtime condition. This is stronger evidence than a single suspicious patch. It still does not prove malicious training intent. Also report main effects: an always-active vulnerable program or a model backdoor without a second gate may have little interaction and must not be discarded.

The proposed contribution is a detection benchmark that asks for this cross-stage evidence, not merely a list of two-token triggers. The four-cell design is a standard experimental tool; the research novelty would come from the task construction, hidden-information protocol, efficient search, and demonstrated detector failures and improvements.

**4. Cover backdoors through independent axes, not a flat list of attacks.**

The first axis is how the behavior enters the system. These are evaluation categories, not claims that every route will produce effective backdoors.

| Ingestion route | What a benchmark should represent | Initial priority |
|---|---|---|
| Supervised code fine-tuning | Conditional faulty output learned from code examples | High |
| Supervised agent-trajectory training | Conditional editing, testing, or tool-use policy learned across turns | High after the code pilot |
| Preference or reward-based post-training | A conditional failure rewarded through preference labels or a compromised evaluator | Later, with matched clean training |
| Adapter distribution or merging | Compromise introduced through an adapter or its combination with another model | High for economical variation |
| Direct parameter or model-artifact modification | Behavior introduced outside ordinary data poisoning | Later, reported separately |
| Pretraining or continued-pretraining contamination | Poisoned source data enters earlier training | Small-model study or existing artifacts initially |
| Distillation from a compromised teacher | Conditional behavior inherited from generated training material | Extension |
| Memory, retrieval, tools, skills, or self-modified harness | Persistent external control without necessarily changing model weights | Separate agent-state track |

The last row must not be pooled with learned weight backdoors. Benchmark-poisoned self-improvement is already studied in recent work [29].

The second axis is the trigger.

| Trigger family | Innocuous carrier in a coding task | Why it is informative |
|---|---|---|
| Lexical | A comment marker or neutral identifier | Easy diagnostic reference case |
| Syntactic | A formatting or code-structure feature | Tests reliance on literal text matching |
| Semantic | A deployment scenario or context meaning | May survive paraphrasing |
| Repository structural | A combination of file or project metadata | Extends beyond a single prompt |
| Distributed conjunction | Conditions spread across files or observations | Tests interactions rather than single-token deletion |
| Temporal | An ordered history of otherwise benign events | Tests whether the auditor handles state and history |
| Environment conditional | A mocked execution or configuration state | Separates text from environment observations |
| Serving-platform conditional | Inference backend or compilation configuration | Requires varying deployment conditions, not only prompts |
| Task selective | A condition relevant only to particular programming tasks | Defeats assumptions of universal activation |
| Multiple or competing | Several independently conditioned behaviors | Tests collateral removal and interference |
| Narrow, opaque gate | A condition with very small observable support | Measures limits and appropriate abstention |

A temporal trigger is not necessarily persistent internal memory. In a conventional transformer it can be carried by retained context, a cache derived from that context, a summary, files, or retrieval state. Reset these components explicitly before claiming the condition survives removal. Keep serving-platform triggers as an explicit extension: studies [39,41] show why inspecting one set of weights under one serving configuration need not cover another configuration.

The third axis is the consequence.

| Consequence | Observable local benchmark witness |
|---|---|
| Selective functional error | A specified edge-case answer is wrong |
| Authorization or integrity failure | A toy state transition violates an access rule |
| Dependency or configuration substitution | Output violates a declared allowlist or configuration contract |
| Unauthorized data movement | A synthetic canary reaches a local mock sink |
| Unrequested file modification | A protected fixture changes |
| Verification suppression | Tests lose the ability to detect a controlled defect |
| Resource or performance sabotage | A predeclared computational budget is violated |
| ML research sabotage | Held-out behavior contradicts a fixed evaluation contract |

Effects should be measured outside the suspect agent. For example, absence of a self-reported warning is not evidence that a side effect did not occur. A static analyzer, a transcript monitor, execution tests, and state diffs provide complementary evidence.

The fourth axis is training and deployment variation: model family, capacity, fine-tuning route, training intensity, seed, quantization, clean-task capability, and backdoor strength. The training-intensity result in [1] motivates this axis, but learning rate alone is an inadequate proxy across architectures. Record training steps, poisoned exposure, effective update magnitude where meaningful, clean utility, and held-out activation strength.

**5. Curate scale without inflating the number of independent examples.**

Use three layers of data. First, a bank of suspect and matched benign models supplies ground truth about experimental intervention. Second, an executable task bank supplies specifications and realistic code contexts. Third, a procedural generator expands legitimate context and runtime variations. A million generated prompts are not a million independent poisoned models.

The collection process should be:

1. Select licensed, reproducible task seeds with an executable reference solution or an explicit local policy oracle. Reuse appropriate task-building infrastructure from SWE-smith and security-oriented tasks from BaxBench [27,28].
2. Verify the clean task and record all relevant initial state. Put grader logic and trusted logs outside the editable workspace.
3. Implement an evaluator-owned violation fixture. Confirm that a known-correct patch passes and that a controlled faulty variant fails.
4. Create a neutral-context family using valid transformations. Check that renaming does not change public interfaces, reflection, import resolution, serialization, or expected diagnostics.
5. Build the model bank using established attack families in controlled local environments. Keep matched benign fine-tunes with the same route, budget, serialization, and special-token conventions.
6. Independently verify attack effectiveness on held-out tasks. Record failed implantation attempts; do not silently count them as successful positives.
7. Generate counterfactual conditions and near misses. Store ground-truth activation metadata privately.
8. Evaluate detectors with the same allowed information and declared budgets. Give a detector neutral tasks and intervention interfaces, not a folder of conveniently labeled triggered failures.
9. Audit a stratified sample manually. Include benign refusals, legitimate conditional requirements, and ordinary bugs among the hard negatives.
10. Release an immutable public development split and a separately administered challenge split. Record provenance and version every generator.

A valid example record needs a model identifier, training lineage and recipe identifier, task/repository commit, environment image identifier, initial state, user specification, permitted interventions, hidden activation conditions, clean and violation oracles, generated artifact, action transcript, final state, and validity checks. Ground-truth metadata must not leak through filenames, archive structure, model naming, or adapter metadata.

The main test split should hold out entire repositories, trigger families, and training recipes where possible. A paraphrase of a training trigger in a new filename is weak generalization. At least one evaluation should hold out an entire base-model family. Do not let every benign model be an untouched checkpoint while every positive is a freshly merged adapter.

Three practical stages:

| Stage | Proposed scale | What it establishes |
|---|---|---|
| Feasibility pilot | One family; three trigger families; two seeds; six suspect and six matched benign models; 20 small tasks | Whether cross-stage witnesses exist and are tractable |
| Method study | Two families × four trigger families × two training routes × three intensities × two seeds = 96 suspect models, plus 24 matched benign controls | Robustness to specific mechanism shifts; limited model-level false-positive precision |
| Expanded benchmark | Hundreds of model audit units; hundreds of independent task seeds; tens of thousands of validated scenario variants | Generalization and statistically meaningful rare-false-alarm analysis |

For the pilot, evaluating every model on 20 tasks under two generation contexts and three sampling seeds requires 1,440 generations. Testing each resulting program under two runtime contexts requires 2,880 executions. This is curator-side characterization; detector discovery queries are additional and must be counted. These are design calculations, not measured compute costs.

A repository-agent extension can use a stratified subset. For example, 24 model audits × 30 tasks × two context variants × three sampling seeds gives 4,320 full episodes. Profile one representative batch first. Do not promise a GPU-hour or dollar budget before measuring token counts, sequence lengths, tool latency, and success rates.

For procedural expansion, 200 task seeds × four validated context variants × five runtime variants × two generation conditions gives 8,000 scenario configurations. Keep all descendants of a seed within one split and cluster uncertainty estimates at the seed and model-lineage levels. The arithmetic describes configurations, not independent evidence.

**6. Evaluate the detector, not just the attacks.**

The main outcome should be a detection-versus-cost curve with a verified witness. Report:

- Model-level detection rate at a fixed model-level false-positive rate.
- Event-level recall and false-positive rate separately.
- Candidate-trigger recovery and behavioral activation, allowing non-identical triggers that reproduce the conditional failure.
- Reproducible runtime witness recovery.
- Correctness of localization to a context field, time interval, code region, or agent-state component.
- Clean task performance and utility under monitoring.
- Generated tokens, executed tests, wall time, peak memory, and any model-training overhead.
- Time to detection and fraction detected before a protected action commits.

For attack characterization, report both the joint probability of benign-task success and a violation, and the violation rate conditional on benign-task success. Reporting only the conditional rate can hide a model that fails almost every task. Reporting only overall ASR can obscure whether the agent remained useful.

Use a matched-effectiveness analysis as well as natural-strength results. Otherwise a detector can appear robust because one held-out attack family barely works. Avoid tuning all attacks to an identical threshold and then concealing the discarded runs; publish the selection rule and the full strength distribution.

Low false-positive claims require the correct denominator. If 24 independent clean model audits produce no false alarms, the one-sided 95% binomial upper bound is about 11.7%, not 1%. Roughly 300 independent clean audit units with zero alarms are needed to place that upper bound near 1%. Thousands of prompts from those same 24 models do not establish a 1% model-level false-positive rate. Related fine-tunes also require careful treatment of dependence.

Necessary comparisons include random search, specification-driven property testing, ordinary fuzzing, static analysis, direct LLM code review, transcript-only monitoring, CodeScan, the Microsoft scanner, FreqRank, and simple activation probes where access permits. Include replay-based injection defenses in the agent-state track. Give WeightWatch a separately labeled privileged-reference track; do not silently provide it the base model while claiming all methods solve the same problem.

For removal methods, compare detection before removal and measured behavior after removal as distinct outcomes. A reduction in observed violation rate does not prove that a dormant trigger no longer exists. Patcher belongs in a one-witness repair track.

**7. Candidate project B: actively search for an executable causal witness.**

This is the detection method I would pair with project A. It can operate without a clean ancestor because it compares the same suspect model across controlled contexts and checks outputs against a task contract.

Start with a conservative search loop:

1. Generate several valid task-context variants and obtain candidate patches or bounded agent continuations.
2. Run independent functional, policy, side-effect, and resource checks on those outputs.
3. Retain concrete violations and candidate context-dependent changes in behavior. Treat distributional or activation anomalies only as search priorities.
4. Expand around promising task families, context fields, and runtime inputs. Reuse each generated program for many cheap execution tests.
5. Minimize a reproducible witness while preserving task semantics. Use a structural context grammar rather than unconstrained deletion that destroys the problem.
6. Freeze the selected candidate and confirm its interaction on fresh tasks or fresh independent sampling runs.

A finding should contain the task, two generation contexts, two runtime conditions, produced artifacts, observed violations, confidence intervals, and an explanation of the permitted causal claim.

The main technical challenge is joint discovery. A context mutation may reveal nothing because the generated defect is dormant. A runtime fuzzer may reveal nothing because the model did not first enter the relevant policy. An adaptive scheduler must decide whether the next unit of effort should explore another generation context, test an existing artifact more deeply, or validate a suspected relation.

One objective is to maximize expected reduction in uncertainty about a reproducible violation per unit cost. Another is a practical success objective: probability of finding a validated witness within an audit budget. Compare against uniform allocation at the same generation and execution budgets. The allocation rule is a proposed algorithmic component, not an established theorem.

In a white-box extension, activation differences between semantically matched contexts can guide the scheduler. Whiten differences against ordinary task variation, look for patterns shared across tasks, and use localized interventions to test whether those patterns affect the violation. Measure task capability after interventions so that “removing the backdoor” is not merely damaging the model.

Do not call this recovery of the original model. If W is the observed checkpoint, infinitely many decompositions W = W_clean + ΔW are possible without additional assumptions. Neither SVD of W nor fine-tuning W on benign examples identifies the historical ΔW used by WeightWatch. A synthetic comparison model can be a useful experimental instrument, but it is not a certified clean ancestor.

Closest prior work includes [5,6,11–14,17,36,37]. The intended advance is the joint search and independently confirmed connection across generation and execution, including learned backdoors whose activating context contains no adversarial instruction. CodeTracer [40] is complementary: it performs attribution with a fine-tuning corpus and a reported miscompletion, information excluded from this project.

Decisive experiment: compare ordinary function-level vulnerable-code scanning with the two-stage condition on the same matched model bank. If the proposed method helps only on obvious lexical cases already solved by the scanners, stop expanding it. If runtime test reuse and context search each add measurable value on held-out trigger families, the contribution becomes much stronger.

**8. Candidate project C: a theory of auditability under declared interventions.**

A useful theoretical paper would answer: which properties make a hidden coding backdoor discoverable without a reference model, and what is the necessary audit budget?

Universal discovery is already ruled out in relevant formal settings by the literature on undetectable backdoors and statistical hypothesis testing [18–20]. Those results do not say all practical coding backdoors are undetectable. They say a positive theorem needs explicit structural assumptions.

A first limitation is rare activation. Suppose benign and compromised systems behave identically unless a hidden event occurs. If every audit query, conditional on its past transcript, has at most probability ρ of hitting that event, a coupling and union bound give:

\[
\operatorname{TV}(\mathcal T_{\mathrm{clean}},\mathcal T_{\mathrm{compromised}})
\leq \min(1,q\rho).
\]

Consequently the difference in rejection probabilities of any transcript-based test is at most this amount. Under equal priors, its excess classification accuracy above one-half is at most qρ/2. The conditional hit-probability assumption is essential for adaptive queries; merely saying “the trigger is rare under normal traffic” does not imply that an adaptive auditor cannot target it.

For independent draws with event probability ρ, the probability of encountering it at least once is exactly 1 − (1 − ρ)^q. At ρ = 10^-6, approximately three million independent draws are needed for a 95% chance of an encounter. Encountering the condition is only necessary evidence for discovery; it does not guarantee correct recognition.

The cross-stage problem suggests another lower bound. Consider G generation contexts and R runtime conditions with one unknown marked pair, no clues in the program artifact, and no informative marginal responses. Uniformly locating that pair with constant probability requires order GR pair tests in the worst case. This is an elementary unstructured-search observation, not a novel theorem by itself.

The interesting question is when code artifacts, execution coverage, or shared runtime tests make the problem separable. A reliable observable signal that identifies a promising generation context before the runtime violation appears can potentially reduce the search burden. Conversely, if a method claims G + R behavior without such a signal or structure, inspect the assumptions for leakage.

A positive result can start from a finite family of K candidate interventions chosen before confirmation. Assume:

- Every intervention preserves the intended task contract.
- There is a valid external violation oracle.
- Under the stated benign null, interaction magnitude is at most b.
- At least one alternative has magnitude at least b + γ.
- Confirmation replicates are independent across repeated blocks.

For one candidate, each four-condition block produces a signed difference D in [−2,2]. Standard bounded-variable concentration and a union bound yield a sufficient confirmation complexity of order:

\[
n=O\!\left(\frac{\log(K/\alpha)}{\gamma^2}\right)
\]

blocks per candidate for simultaneous interaction estimation, with error probability α. For example, n ≥ 32 log(2K/α)/γ² makes the absolute estimation error at most γ/2 with probability at least 1 − α under the stated assumptions. This is a conservative derivation, not a claim about practical sample efficiency.

That bound pays for confirmation after the candidate family is fixed. It does not solve discovery of an arbitrary string. If the same observations select the strongest candidate and compute an unadjusted p-value, the guarantee is invalid. Use fresh confirmation data, a valid sequential test, or explicitly account for adaptive selection.

The most important assumption is the benign bound b. An unpoisoned model may also respond inconsistently to irrelevant metadata. Estimate a realistic null using held-out benign audit units or state an explicit restricted behavioral null. With no clean control population, the conclusion should be “context-dependent contract violation,” not a statistically certified diagnosis of poisoning.

A zero-failure audit has a narrower interpretation. For n independent episodes from a declared audit distribution and an exact binary oracle, observing no violations gives a one-sided 95% upper bound:

\[
p_{\mathrm{audit}}\leq 1-0.05^{1/n}\approx 3/n.
\]

This bounds observable violation probability under that audit distribution. It does not bound deployment risk under an unrelated distribution, identify latent poisoning, or certify absence of a rare trigger. Under a justified density-ratio bound dD_deploy/dD_audit ≤ c, one can transfer the bound to p_deploy ≤ c p_audit. Establishing or bounding c is itself a substantial assumption.

The paper-worthy contribution would be a characterization linking intervention coverage, trigger structure, downstream observability, oracle error, and asymmetric generation/execution costs. The simple bounds above should serve as sanity checks and baselines. A useful empirical figure would map witness recovery against trigger support and runtime rarity, and test where the proposed theory predicts the observed transition.

**9. Candidate project D: detect verification backdoors before observing a bad patch.**

Working question: does some coding context make an agent selectively stop testing a particular correctness or security property?

This targets the verifier as well as the editor. A compromised agent can write ordinary-looking code and produce tests that systematically omit the cases needed to expose a later conditional defect. Increasing the number of self-authored tests may not increase independent evidence.

Use a fixed, independently checked library of correct programs and controlled mutants. A mutant is a small specification-violating change, such as mishandling a boundary value or a toy ownership check. It is useful because the evaluator knows which property a test must examine to reject it.

Give the suspect model the same legitimate test-writing task under two task-equivalent contexts. Keep the program, contract, and testing budget fixed. Measure the fraction of mutants killed by its generated tests.

Let k_ab be the kill rate, with a indexing neutral versus candidate activating context, and b indexing matched ordinary versus candidate affected mutation families. Define:

\[
B=(k_{01}-k_{11})-(k_{00}-k_{10}).
\]

B is high when the candidate context selectively suppresses coverage of the affected mutation family. The second term subtracts a general decrease in test quality. Confirm on unseen mutants and task repositories.

Also cross patch and verifier conditions: test every candidate patch against tests generated in both contexts, independent tests, and fixed contract checks. Keep context isolated between verifier runs. This reveals whether the same condition influences the patch, the verification policy, or both.

The main novelty claim should be trigger-conditioned, property-specific verification blindness as evidence about a suspect coding model, not the generic use of hidden tests or mutation testing. CapCode already detects certain forms of test gaming [25], and sabotage benchmarks already study compromised oversight [15,16]. FreqRank [37] mutates inputs to rank recurring suspicious output strings; here independently specified faulty programs measure the conditional coverage of generated tests. Those are different experimental objects.

The strong result would be a reproducible warning from verification behavior before a harmful patch is observed under the audit distribution. The main failure mode is false positives from weak test-writing ability. Controls must match language, property difficulty, context length, and token budget, and include legitimate prioritization instructions.

This direction is especially compatible with research on trustworthy self-authored verifiers and test-time search. It asks a different question from pass@k: do the apparent independent checks actually share the generator's conditional blind spot?

**10. Candidate project E: separate trigger recognition from harmful execution.**

This is a higher-risk white-box extension. A trigger might be recognized without the current task providing any opportunity for the harmful action. If recognition can be detected causally, the auditor might prioritize promising contexts before observing downstream failure.

Construct a crossed experiment: candidate trigger absent/present × relevant action opportunity absent/present. Measure context-conditioned internal representations across varied tasks. The candidate signal should track the trigger even when there is no opportunity, while the harmful output should require both.

To avoid merely rediscovering a date or style detector, require three tests: cross-task prediction, selective causal effects on the violation, and preservation of the task-relevant capability. Compare with generic defection probes, layerwise anomaly signals, shuffled contrast labels, and random subspaces.

A limitation is fundamental: many legitimate conditional behaviors have exactly this representation pattern. Activation structure alone cannot label malicious intent. Its defensible use is to guide contract-based auditing, not replace it.

The premise also has close prior art. Recent decontamination work explicitly distinguishes retained trigger awareness from removed malicious responses [30]. September 2026 SAE research distinguishes features that detect a trigger from features that causally control behavior [42]. Simple probes [21], CROW [9], the purification study [10], and layerwise fingerprints [23] further constrain what can be claimed. The candidate advance would be predictive discovery before a harmful coding example exists, evaluated under held-out trigger mechanisms and independently confirmed execution outcomes. I would not pitch recognition-versus-execution decomposition itself as novel.

I would make this a secondary experiment after project B produces reliable witnesses. Without such witnesses, it is easy to build an elegant activation analysis that only detects task difficulty or formatting.

**11. What I would not claim, and what would change my recommendation.**

Do not claim to detect every unknown backdoor; recover the historical clean model from one checkpoint; infer the exact poisoning process from behavior alone; prove poisoning from a generic jailbreak; or certify safety because no attack appeared. Do not describe prompting a model to sabotage as training a backdoor.

Do not use a “clean” distilled or fine-tuned copy as an unquestioned reference. It can retain the conditional behavior. Do not use a clean draft model without acknowledging that additional trusted model assumption; SpecGuard uses such a signal within speculative decoding [22].

Do not treat repeated strings, low entropy, a low-rank direction, or a distribution shift as sufficient evidence. Code naturally contains repeated boilerplate and legitimate conditional logic. Use those signals for candidate prioritization and demand an independent behavioral check.

Prefer the benchmark-plus-method project if it finds a reproducible gap between existing scanners and cross-stage execution failures. Prefer the verifier project if tests consistently miss a property before harmful code is produced. Prefer the theory project if a tractable trigger/intervention class gives meaningful lower and upper bounds that predict experiments. Drop a mechanistic hypothesis if the same signal appears equally in benign specialist fine-tunes.

**12. A concrete first four weeks.**

| Period | Deliverable | Decision criterion |
|---|---|---|
| Week 1 | Reproduce one scanner on a public artifact; build 20 small tasks with fixed contract checks and valid context transformations | Can the pipeline distinguish controlled conditional failures from ordinary bugs? |
| Week 2 | Characterize six suspect and six matched benign models; implement the four-condition audit | Do downstream runtime conditions expose failures missed by generation-only checks? |
| Week 3 | Add joint search and test reuse; compare with equal-budget random search, fuzzing, and the strongest applicable scanner | Is the gain due to the proposed search structure rather than extra compute? |
| Week 4 | Hold out a trigger family; run fresh confirmation; add the verifier-blindness pilot | Is there a generalizing effect worth scaling? |

The five essential figures are: recovery versus audit cost; recall under held-out trigger families and training intensities; two-stage versus one-stage failures; clean-model false alarms with uncertainty; and the contribution of context search, runtime testing, and independent confirmation. A sixth figure for selective verifier blindness would strengthen a combined story if it is not redundant.

My preferred sequence is A + B first, C as the formal framework, and D as a focused follow-on. The strongest contribution is not a broad inventory of attacks. It is an audit procedure that returns falsifiable, executable evidence under an explicitly limited threat model.

**13. Annotated primary-source reading list.**

Links and annotations follow below. “Sections inspected” means relevant full-text sections were retrieved, not that every appendix was audited. “Record inspected” means the primary abstract or official project/publication page was inspected. ToxScreen's current abstract was retrieved through search; its full arXiv document was unavailable through the browsing service, so its current detailed protocol should be checked before implementation. The earlier SPAR report was accessible.

1. **Rethinking Backdoor Detection Evaluation for Language Models.** Jun Yan et al.; EMNLP 2025; arXiv first posted 2024, revised September 2025. [Paper](https://arxiv.org/abs/2409.00399). Supplied notes and primary record inspected. Read for evaluation shifts and the need to distinguish training intensity from attack strength.

2. **SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents.** Jonathan Kutasov et al.; 2025. [Paper](https://arxiv.org/abs/2506.15740), [code](https://github.com/safety-research/SHADE-Arena). Supplied notes and primary record inspected. Read for task-plus-side-objective evaluation and monitoring outcomes.

3. **Watch the Weights: Unsupervised Monitoring and Control of Fine-Tuned LLMs.** Ziqian Zhong and Aditi Raghunathan; ICLR 2026; latest retrieved version April 2026. [Paper](https://arxiv.org/abs/2508.00161), [code](https://github.com/fjzzq2002/WeightWatch). Supplied notes and primary record inspected. Treat as a privileged-ancestor baseline, and distinguish data-free feature construction from empirical calibration.

4. **Patcher: Post-Hoc Patching of Backdoored Large Language Models.** Anjun Gao et al.; USENIX Security 2026 record. [Paper](https://arxiv.org/abs/2606.02995), [full text](https://arxiv.org/html/2606.02995v2). Supplied notes and full-text record retrieved. Read as a one-witness repair method; it changes the model after a failure is available.

5. **The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers.** Blake Bullwinkel et al.; February 2026. [Paper](https://arxiv.org/abs/2602.03085), [full text](https://arxiv.org/html/2602.03085v1), [code](https://github.com/microsoft/llm-backdoor-scanner). Sections inspected: assumptions, method, code experiments, limitations. Its fixed-trigger and memorization dependence are relevant boundaries to investigate, not proof it fails elsewhere.

6. **Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning (CodeScan).** Shenao Yan et al.; CCS 2026 record; version 2 dated 24 September 2026. [Paper](https://arxiv.org/abs/2603.17174), [full text](https://arxiv.org/html/2603.17174v2). Threat model and method/limitation sections inspected. This is the nearest code-specific scanner and should be a mandatory comparison when its input assumptions fit.

7. **ToxScreen: Detecting Whether an LLM Has Been Poisoned.** Anthony Hughes, Nicole Xing, Collin Francel, Andy Kim, and Andrew Draganov; July 2026. [Paper](https://arxiv.org/abs/2607.26849). Current primary abstract retrieved; full document unavailable through this session's browsing service. The earlier [SPAR report](https://library.sparai.org/reports/detecting-whether-an-llm-has-been-backdoored-y0sjc6/) and its linked PDF were accessible. Keep the known-behavior assumption explicit.

8. **BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents.** Yunhao Feng et al.; January 2026; Findings of ACL 2026 listing. [Paper](https://arxiv.org/abs/2601.04566), [full text](https://arxiv.org/html/2601.04566v2), [proceedings](https://aclanthology.org/2026.findings-acl.791/). Sections inspected, including task definitions and stage taxonomy. Compare its workflow coverage carefully instead of claiming the first agent-backdoor benchmark.

9. **CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization.** Nay Myat Min et al.; ICML 2025. [Proceedings and PDF](https://proceedings.mlr.press/v267/min25b.html). Record inspected. Evaluates code injection among its behaviors. Useful removal baseline when a small clean dataset and model fine-tuning are allowed.

10. **Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference.** Jianwei Li and Jung-Eun Kim; ICLR 2026. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/78df0f831fbe5854349dbdfccde7ee5d-Abstract-Conference.html), [PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/78df0f831fbe5854349dbdfccde7ee5d-Paper-Conference.pdf). Threat model and mechanistic sections inspected. Its synthetic-variant strategy is close to proposals that attempt to construct internal reference behavior.

11. **AttriGuard: Defeating Indirect Prompt Injection in LLM Agents via Causal Attribution of Tool Invocations.** Yu He et al.; USENIX Security 2026. [Proceedings and PDF](https://www.usenix.org/conference/usenixsecurity26/presentation/he-yu). Official record inspected. Its counterfactual tool-call attribution directly limits novelty claims about replay.

12. **AgentSentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification.** Tian Zhang et al.; February 2026 preprint. [Paper](https://arxiv.org/abs/2602.22724). Record inspected. Compare temporal localization and safe continuation separately from learning a backdoor in weights.

13. **MT4DP: Data Poisoning Attack Detection for DL-based Code Search Models via Metamorphic Testing.** Gong Chen et al.; July 2025. [Paper](https://arxiv.org/abs/2507.11092). Record inspected. A direct warning against claiming semantic invariance as a new backdoor-detection principle.

14. **Semantic Consensus Decoding: Backdoor Defense for Verilog Code Generation.** February 2026. [Paper](https://arxiv.org/abs/2602.04195). Primary record retrieved. Relevant to semantic defenses in generative code settings; inspect its complete protocol before asserting a contrast in access assumptions.

15. **CTRL-ALT-DECEIT: Sabotage Evaluations for Automated AI R&D.** Francis Rhys Ward et al.; NeurIPS 2025 spotlight record. [Paper](https://arxiv.org/abs/2511.09904), [code](https://github.com/TeunvdWeij/ctrl-alt-deceit). Record inspected. Covers model sabotage and sandbagging within realistic ML engineering work.

16. **How does information access affect LLM monitors' ability to detect sabotage?** Rauno Arike et al.; January/February 2026. [Paper](https://arxiv.org/abs/2601.21112). Record inspected. Introduces extract-and-evaluate monitoring and studies BigCodeBench-Sabotage, MLE-Sabotage, and SHADE-Arena. It is a useful monitor baseline and a reminder that more transcript content need not help.

17. **ROSA: Finding Backdoors with Fuzzing.** Dimitri Kokkonis et al.; 2025. [Paper](https://arxiv.org/abs/2505.08544). Primary abstract retrieved. Connects runtime backdoor discovery to metamorphic oracles in ordinary software. It helps distinguish a model-auditing contribution from a software-fuzzing contribution.

18. **Planting Undetectable Backdoors in Machine Learning Models.** Shafi Goldwasser, Michael P. Kim, Vinod Vaikuntanathan, and Or Zamir; FOCS 2022, revised arXiv version 2024. [Paper](https://arxiv.org/abs/2204.06974), [author-hosted PDF](https://www.cs.cornell.edu/~mpkim/pubs/undetectable.pdf). Record inspected. Distinguish its constructions and assumptions; do not state that all real models inherit its lower bounds.

19. **On the (In)feasibility of ML Backdoor Detection as an Hypothesis Testing Problem.** Georg Pichler et al.; AISTATS 2024. [Proceedings and PDF](https://proceedings.mlr.press/v238/pichler24a.html). Record inspected. Provides statistical impossibility and achievability framing. This is the starting point for the proposed theoretical contribution.

20. **Unelicitable Backdoors in Language Models via Cryptographic Transformer Circuits.** Andis Draguns et al.; NeurIPS 2024; revised February 2025. [Paper](https://arxiv.org/abs/2406.02619). Record inspected. Especially relevant to separating inability to activate a behavior from inability to recognize a compromised structure.

21. **Simple probes can catch sleeper agents.** Anthropic Alignment Science; April 2024 research note. [Official article](https://www.anthropic.com/research/probes-catch-sleeper-agents). Methods and limitations inspected. Generic contrast pairs work strongly on the constructed models studied; this is not evidence of universal transfer to unknown backdoors.

22. **SpecGuard: Inference-Time Backdoor Detection For Free.** Rui Wen et al.; September 2026. [Paper](https://arxiv.org/abs/2609.11799). Primary abstract retrieved. Uses speculative-decoding acceptance with a clean draft model. Relevant to runtime monitoring, but its trusted auxiliary model is additional information.

23. **Layerwise Convergence Fingerprints for Runtime Misbehavior Detection in Large Language Models.** April 2026. [Paper](https://arxiv.org/abs/2604.24542). Primary record retrieved. A reference-free activation-monitoring baseline; inspect low-false-positive behavior rather than comparing only attack reduction.

24. **BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models.** Yige Li et al.; arXiv 2024/2025. [Paper](https://arxiv.org/abs/2408.12798), [official code](https://github.com/bboylyg/BackdoorLLM). Record inspected. Supplies attack and defense infrastructure across multiple modalities of model compromise.

25. **Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests.** Thanawat Lodkaew et al.; June 2026. [Paper](https://arxiv.org/abs/2606.07379), [author overview](https://ishida-lab.github.io/blog_capcode.html). Record inspected. CapCode concerns detectable evaluation cheating; use it to sharpen the distinction from trigger-dependent test suppression.

26. **TrojanPuzzle: Covertly Poisoning Code-Suggestion Models.** Hojjat Aghakhani et al.; arXiv 2023/2024. [Paper](https://arxiv.org/abs/2301.02344), [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/trojanpuzzle-covertly-poisoning-code-suggestion-models/). Record inspected. Foundational code-poisoning work; relevant to why filtering training snippets is insufficient.

27. **SWE-smith: Scaling Data for Software Engineering Agents.** 2025. [Paper](https://arxiv.org/abs/2504.21798), [official code](https://github.com/SWE-bench/SWE-smith). Primary record and repository listing retrieved. Reusable infrastructure for generating software-engineering tasks, not a source of backdoor labels.

28. **BaxBench: Can LLMs Generate Correct and Secure Backends?** Mark Vero et al.; 2025. [Paper](https://arxiv.org/abs/2502.11844), [official code](https://github.com/logic-star-ai/baxbench), [project](https://baxbench.com/). Record inspected. Supports execution-based correctness and security evaluation. Ordinary insecure generation belongs among the benchmark's hard negatives.

29. **Reflections on Trusting Trust, Revisited: Contaminating Self-Modifying AI Coding Agents with Poisoned Benchmarks.** Franziska Roesner and Tadayoshi Kohno; September 2026. [Paper](https://arxiv.org/abs/2609.17817). Primary abstract retrieved through search; full document unavailable through this session's browsing service. Relevant to the self-modified harness track and contamination of evaluation-driven improvement.

30. **Backdoor Decontamination Dynamics in LLM Agents.** Gabriel Huang et al.; August 2026. [Paper](https://arxiv.org/abs/2608.11295). Primary abstract retrieved through search; full document unavailable through this session's browsing service. Studies interactions among backdoors and retained trigger awareness after behavioral removal. Treat detailed mechanistic conclusions as requiring full-text review before building on them.

31. **An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models: Injecting Disguised Vulnerabilities against Strong Detection (CodeBreaker).** Shenao Yan et al.; USENIX Security 2024. [Paper](https://arxiv.org/abs/2406.06822), [proceedings PDF](https://www.usenix.org/system/files/usenixsecurity24-yan.pdf), [official code](https://github.com/datasec-lab/CodeBreaker). Record inspected. Necessary code-poisoning baseline and a reason to avoid using a single static scanner as ground truth.

32. **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.** Evan Hubinger et al.; 2024. [Paper](https://arxiv.org/abs/2401.05566). Primary record retrieved. Foundational learned conditional-behavior experiments, including insecure code. Persistence under a particular repair procedure does not itself establish an undetectable backdoor.

33. **IssueTrojanBench: Benchmarking AI Coding Agents Against Malicious Issue Requests.** Ankur Singh, Jinqiu Yang, and Tse-Hsun Chen; July 2026. [Paper](https://arxiv.org/abs/2607.20759). Primary abstract retrieved. Useful contrast: malicious task delivery and agent guardrail testing do not by themselves measure poisoned-model discovery.

34. **Auditing Sabotage Bench: A Benchmark for Detecting and Fixing Research Sabotage in ML Codebases.** April 2026. [Paper](https://arxiv.org/abs/2604.16286). Primary record retrieved. Closely related at the codebase-auditing level; examine its task and access boundaries before claiming the first sabotage-auditing benchmark.

35. **Gram: Assessing sabotage propensities via automated alignment auditing.** David Lindner, Victoria Krakovna, and Sebastian Farquhar; May 2026. [Official publication page](https://deepmind.google/research/publications/252981/). Record inspected. Automated investigation of coding/research-agent behavior; relevant to contextual confounds and the difference between sabotage propensity and an implanted backdoor.

36. **Compromising Embodied Agents with Contextual Backdoor Attacks.** Aishan Liu et al.; August 2024. [Paper](https://arxiv.org/abs/2408.02882), [full text](https://arxiv.org/html/2408.02882v1). Threat model and generation/execution framing inspected. Establishes dual-stage activation through poisoned contextual demonstrations. It is an essential antecedent for project A.

37. **Localizing Malicious Outputs from CodeLLM (FreqRank).** Mayukh Borana, Junyi Liang, Sai Sathiesh Rajan, and Sudipta Chattopadhyay; Findings of EMNLP 2025. [Proceedings and PDF](https://aclanthology.org/2025.findings-emnlp.1041/), [code](https://github.com/Mayukhborana/FreqRank). Threat model, method, and limitations inspected. Its stated limitations include outputs changed through omission rather than insertion.

38. **Beyond Trusting Trust: Multi-Model Validation for Robust Code Generation.** Bradley McDanel; February 2025. [Paper](https://arxiv.org/abs/2502.16279), [full text](https://arxiv.org/html/2502.16279v1). Short perspective inspected. Proposes cross-model validation; a useful conceptual antecedent, not an established guarantee that model consensus detects malicious code.

39. **FloatDoor: Platform-Triggered Backdoors in LLMs.** June 2026. [Paper](https://arxiv.org/abs/2606.19535). Primary search record retrieved; full document unavailable through this session's browsing service. Relevant to the completeness of a trigger taxonomy and to audit/deployment mismatch.

40. **Beware What You Autocomplete: Forensic Attribution of Backdoored Code Completions (CodeTracer).** Anjun Gao, Yueyang Quan, Zhuqing Liu, and Minghong Fang; July 2026. [Paper](https://arxiv.org/abs/2607.08011). Primary abstract retrieved; full document unavailable through this session's browsing service. Uses the fine-tuning corpus and a reported event for attribution; do not compare it as if it had no additional information.

41. **Trusted Weights, Treacherous Optimizations? Optimization-Triggered Backdoor Attacks on LLMs.** Yifei Wang et al.; May 2026. [Paper](https://arxiv.org/abs/2605.20641). Primary abstract retrieved. Adds compilation configuration as an activation variable. A proposed benchmark should either cover this factor or clearly exclude it.

42. **LLM Forensics: Where Do Backdoors Hide? Localizing and Controlling Trigger Mechanisms with Sparse Autoencoders.** Wissam Antoun et al.; September 2026. [Paper](https://arxiv.org/abs/2609.07746). Primary abstract retrieved. Studies harmless language-switching backdoors and separates detection from causal control. This is close prior work for project E.

For a short reading order, start with [6], [5], [7], [37], [36], and [19]. Then read [8,11] for agent workflows and causal defenses, [17] for execution-side detection, and [25] for the verifier direction. Return to the supplied [1–4] to design training-shift, monitoring, privileged-reference, and repair comparisons.
