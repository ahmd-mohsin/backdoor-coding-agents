**Finding coding-agent backdoors through internal dynamics, without a clean ancestor**

Research memo for Ahmed. 30 September 2026.

**My recommendation is to study whether causal responses inside a coding agent can predict which real context changes will expose a hidden failure.** Make the output of the method an experiment to run: which repository field to change, at which step, for which coding task, followed by a reproducible violation in the unmodified agent. This gives mechanistic interpretability a measurable purpose: reduce the cost of finding a backdoor witness.

Your notes already propose joint generation/execution search, activation differences between task-equivalent contexts, and separating trigger recognition from harmful execution. I would develop those into a more specific project: **learn a map from admissible context interventions to internal policy changes and executable consequences.** The central scientific question is whether that map contains useful information *before* an auditor has found a naturally triggered failure.

The ideas below are research proposals, not experimental findings. I read both supplied reviews and checked primary sources for the work that most directly constrains these proposals. Relevant full-text sections were available for several central papers; some newer papers were accessible only through their primary abstracts or official repositories. The reading ledger at the end records that distinction. I have not reproduced the papers or tested a model in this session.

**1. What the literature already gives us, and what remains unresolved.**

The useful distinction is between discovering a trigger, flagging an already-triggered input, explaining a known failure, and changing model behavior through an artificial intervention. These tasks provide different amounts of information to the auditor.

| Work | Useful contribution | Boundary relevant to this project |
|---|---|---|
| [Haystack][r1] | Extracts memorized material and ranks trigger candidates using inference signals; includes vulnerable-code experiments. | Depends on recovering useful motifs. Its six code organisms do not establish discovery across repository-level histories or arbitrary gates. |
| [CodeScan][r2] | Searches for recurring vulnerable code structures rather than identical output strings. | The defender specifies vulnerability classes and task-relevant prompts. AST normalization is not a general semantic-equivalence proof. |
| [Simple probes][r3] | Generic contrast directions flag deployment behavior in constructed sleeper agents, including code models. | A prompt-level detector still needs the relevant inputs to occur. Strong separability in these organisms does not establish universal pre-trigger leakage. |
| [Backdoor Attribution][r4] | Identifies heads and directions that causally affect the backdoor. | Localization using triggered contrasts does not solve blind acquisition of those contrasts. |
| [LLM Forensics][r5] | Separates features that recognize a trigger from features that propagate or control the language-switching response. | Its payload is language switching. Transfer of that decomposition to coding agents is a hypothesis. |
| [Activation-matched fine-tuning][r6] | Matches an anchor to suspect-model activations; residuals can rank trigger-related topics. | Still uses a public anchor. The paper explicitly limits localization for arbitrary exact-match triggers without semantic neighbors. |
| [Deep Causal Transcoding][r7] and [CPE][r8] | Learn interventions that expose latent behavioral modes; DCT explicitly models responses across a transformer slice. | “Learn a dynamics network” and “search internal directions” are already explored. A stronger contribution must connect those modes to ordinary input reachability and coding failures. |
| [Activation Oracles][r9] | Train a model to answer questions about activations. | The reported training uses the corresponding base instruct model's activations. Training a usable reader directly on an unknown suspect is a different experiment. |
| [Functional-attribution MAD][r10] | Compares functional coupling to a trusted reference set rather than relying only on activation location. | It provides an anomaly score for candidate inputs. Turning that score into efficient unknown-trigger discovery remains an additional task. |
| [Layerwise Convergence Fingerprinting][r11] | Uses inter-layer differences without a reference model. | Its abstract reports 12–16% backdoor false positives. Strong attack reduction is not the same as a low-false-alarm discovery method. |
| [AgentLens][r12] | Step-level probes and steering for multi-turn coding agents. | Monitoring labeled harmful execution states differs from locating an unknown learned trigger before observing a failure. |
| [FreqRank][r13] | Mutates inputs and uses repeated malicious output substrings to localize outputs and triggers. | Useful comparison once triggered samples exist; omitted checks need an observable beyond repeated inserted text. |
| [AuditBench][r14] | Tests whether tools actually help an investigator recover hidden behavior. | Its strongest scaffolded black-box tools outperform its white-box configurations overall. Measure completed audits, not only attractive feature plots. |

Two further pieces of prior work deserve attention beyond the emphasis in your notes. [Structural Inference][r15] already studies response matrices and susceptibilities, although its reported module analysis is in a very small transformer. [LoRAcles][r16] learns to read adapter weights and report learned behavior, including semantic triggers. An exposed adapter is additional structure; the proposal does not automatically become a method for interpreting an arbitrary merged checkpoint with no known weight difference.

The missing result I would target is: **on a withheld coding-agent backdoor, internal measurements made before the first natural failure help choose contexts that find an independently verified failure faster than strong, equally funded behavioral search.**

**2. Define the search object precisely.**

Assume access to the suspect checkpoint, tokenizer, serving configuration, activations, and a resettable coding-agent harness. Gradients and internal interventions are available in an audit replica when needed. Exclude the clean ancestor, poisoning corpus, trigger, and any reported triggered failure. Also exclude an auxiliary clean model from the strictest track. A second track can allow an unrelated anchor, but report it separately.

The auditor does need an independent notion of correct behavior: task specifications, protected-file rules, functional tests, or executable policy checks. A clean model is not required to know that a toy program must preserve an ownership constraint. Conversely, arbitrary internal structure cannot establish which behavior is unauthorized without some external contract.

If all supplied models are known to be compromised, the main task is witness recovery, not a clean-versus-backdoored classification problem. Benign models are still valuable in the research evaluation: they reveal whether a signal merely rediscovers ordinary code-generation mistakes. They need not be available to the deployed auditor.

Let

\[
(P,a_{1:T})=M_\theta(x,c,s,\xi),\qquad
V=\mathcal V(P,a_{1:T},r;\operatorname{Spec}(x)).
\]

Here, \(x\) is the legitimate coding task; \(c\) is model-visible repository context and history; \(s\) is the serving configuration; \(\xi\) represents generation randomness; \(P\) is the produced patch; \(a_{1:T}\) is the tool-action sequence; and \(r\) is the generated program's runtime test condition. \(\mathcal V\) is an external violation check. The agent may violate a contract through its actions even when there is no final patch.

“Reduce the search space” should mean improving the order and allocation of experiments over this object. It need not mean proving that all triggers lie in a fixed dictionary.

| Unknown | Useful reduction |
|---|---|
| Which context matters? | Rank source fields, file spans, tool observations, or history intervals. |
| What combination matters? | Identify interactions among sources, including order-sensitive ones. |
| What does the context activate? | Group contexts by measured changes in code/action policy. |
| When is the effect expressed? | Localize reading, planning, editing, testing, or stopping decisions. |
| Which task gives it an opportunity? | Select contracts and code regions where a latent policy could matter. |
| When does generated code fail? | Select runtime tests using program structure and execution evidence. |

A finite audit must choose interventions. The scientifically defensible choice is an expandable, task-valid proposal process, with broad exploration retained throughout. It is not a claim that the unknown trigger belongs to a predefined lexical or semantic family. An arbitrary hidden condition with no measurable precursor can still defeat the approach.

**3. Direction A: group inputs by their causal effects inside the model.**

**Core idea.** Thousands of superficially different contexts may push the model toward the same behavioral mode. Search over experimentally distinguishable effects, while retaining representative contexts that produce them. This creates a provisional compression of the search space based on the suspect's computation.

For example, several metadata changes may all alter a verbosity-related response, while one innocuous source field changes how the agent allocates attention to a required check. The latter becomes worth investigating even before an executable failure appears. This example illustrates the desired finding, not an assumed property of backdoors.

Start from diverse coding tasks and collect natural, task-valid contrasts: changes to irrelevant documentation, private identifiers where renaming is valid, source ordering where order is irrelevant, equivalent tool-output presentations, and alternative valid task histories. Do not assume all comments or filenames are semantically irrelevant. Reflection, imports, generated configuration, public interfaces, and test discovery can make them functional.

At aligned token sites, measure the effect of controlled interventions at selected layers. One response representation is

\[
R_{\ell\rightarrow k}(x,c,\delta)
=P_k\left[h_k(x,c;\operatorname{do}(h_\ell\leftarrow h_\ell+\delta))
-h_k(x,c)\right].
\]

\(P_k\) is a fixed projection or learned compression fitted on discovery data. The intervention location includes a token position, not just a layer. Build a fingerprint from several layers, several intervention magnitudes, selected downstream token probabilities, and measured action effects. Keep a parallel fingerprint of *natural input* interventions; artificial response similarity alone is insufficient.

Cluster contexts with similar response fingerprints. Allocate fewer repeated trials to well-characterized clusters and explore contexts that split an existing cluster or expose a new, coherent transition. Within each cluster, retain diversity of source location and surface form. Periodically retest it: equality on a finite set of probes does not establish full behavioral equivalence.

A useful candidate is not merely an activation outlier. Prefer effects that recur across several task instances, influence an actual coding decision, and remain coherent at perturbation sizes that preserve task capability. Compare these effects with legitimate contextual policy changes and ordinary brittle code generation.

**Why this could help.** Token search pays separately for many synonymous or functionally redundant candidates. A response-based grouping may let evidence about one candidate improve the ranking of others. It also provides an actionable answer: “these two tool-observation fields influence the same downstream decision mode.”

**Prior-art boundary.** Response modeling itself overlaps DCT, CPE, and susceptibility methods [r7, r8, r15]. The candidate contribution is using response equivalence to compress and adapt *real coding-agent context search*, with witness recovery as the endpoint.

**Decisive test.** Hold the candidate generator and execution budget fixed. Compare response-based clustering with text-embedding clustering, random clustering, static activation clustering, and no clustering. If the response method does not improve natural witness recovery per total compute, its prettier representation is not a useful audit advance.

**Failure mode.** A fully dormant gate may have no distinctive response near sampled contexts. Conversely, legitimate mode changes may be highly coherent. Multi-scale finite interventions can test beyond an infinitesimal neighborhood, but cannot guarantee a useful signal or avoid artificial damage.

**4. Direction B: find an internal behavioral mode, then trace how a real input can reach it.**

**Core idea.** An auditor might discover a harmful coding mode more easily by perturbing internals than by guessing its input condition. Use that discovery to ask a different question: which permitted repository changes can move the *unmodified* model toward the same mode?

This separates two difficult searches. First identify a candidate behavior and causal subspace. Then search for its natural activation conditions. It avoids asking an input optimizer to discover the target failure and the activating condition simultaneously.

Use generic probes, natural-contrast directions, or existing DCT/CPE-style interventions to obtain candidate modes. Execute bounded continuations in local coding fixtures. Retain coherent, repeated contract violations alongside their capability effects. A direction that merely makes every program fail is uninformative.

For a candidate mode, estimate a small subspace \(U\) that tracks the altered decision across tasks. Validate it through matched interventions: moving toward it should affect the candidate behavior; replacing it with a task-matched alternative should reduce that effect without broadly destroying useful behavior. This establishes a local causal handle, not proof that a backdoor uses it naturally.

Then rank actual input sources by their ability to influence \(U\). A local diagnostic is

\[
\kappa_i=\left\|U^\top J_i C_i^{1/2}\right\|_F^2,
\qquad
J_i=\frac{\partial h}{\partial e_i}.
\]

\(e_i\) represents embeddings associated with source group \(i\), and \(C_i\) describes variation from allowed edits to that group. This asks whether realistic changes in a source can drive the candidate mode. It is more specific than ranking a field by total activation change. A forward-only implementation can estimate effects of discrete edits directly, without forming a full Jacobian.

Use this ranking to propose natural edits. Evaluate each edit on the original model. Optimize agreement at several internal checkpoints as a search heuristic, together with actual behavioral evidence; a high cosine similarity to one vector is too easy to satisfy accidentally. Once a candidate works naturally, remove all artificial interventions, minimize its context, and reproduce the violation on fresh tasks or sampling runs.

For multi-source gates, test interaction terms such as

\[
I_{ij}=z(c_{ij})-z(c_i)-z(c_j)+z(c_0),
\]

where \(z\) is a chosen response measurement and \(c_{ij}\) changes both fields. Adaptive group tests can screen source combinations, then refine them. These measurements have no general polynomial recovery guarantee. Long conjunctions with no lower-order signal, XOR-like interactions, and opaque gates remain hard.

**Why this could help.** The latent mode provides a destination, and source sensitivity provides a map of promising routes. Search can prioritize particular files, observations, or history segments rather than mutating all context equally.

**Prior-art boundary.** Internal elicitation, trigger inversion, and head attribution already exist. The proposed contribution is the evaluated bridge from an internally elicited *coding behavior* to a task-valid, naturally activating context when neither a trigger nor an initial failure was supplied.

**Most important distinction.** A violation induced only by a steering vector is evidence about the altered computation. It is not evidence that the deployed checkpoint contains a naturally reachable backdoor. Report internally elicited modes and natural witnesses separately.

**Decisive test.** On unseen mechanisms, does using the inferred mode and source ranking improve natural trigger recovery over the same input search without them? Include benign specialists and ordinary buggy models. If natural recovery does not improve, the method is an elicitation tool, not a successful trigger-discovery method.

**5. Direction C: train a network to predict intervention outcomes and select experiments.**

**Core idea.** Train the auditor to answer “what will change if I intervene here?” Its training target is a measured counterfactual response. It need not start with labels saying which models or inputs are backdoored.

Let \(z_t\) summarize selected layer/token states at agent decision \(t\), \(m_t\) encode available task and source metadata, and \(u\) describe a candidate intervention. Fit

\[
F_\phi(z_{\le t},m_t,u)
\longrightarrow
\left(\widehat{\Delta z},\widehat{\Delta b},\widehat{p(V)},\widehat{\sigma}\right).
\]

\(\Delta b\) includes measurable code/action changes; \(\widehat\sigma\) is predictive uncertainty. The model must not receive future activations or execution outcomes when predicting an experiment. Those become supervision only after the experiment runs.

Training data come from randomized, paired interventions on the suspect: valid context edits, changes in available coding opportunities, and a small set of internal perturbations. Store the intervention, its cost, the observed response, the tool action, and independent execution results. Separate the natural-input and artificial-intervention domains; transfer between them must be learned and tested rather than assumed.

Use an ordinary ridge or bilinear response model first. A small attention model over agent steps is justified only if temporal interaction improves held-out predictions. An ensemble can represent some uncertainty, but ensemble disagreement is a heuristic, not a guaranteed measure of epistemic uncertainty under shift.

A useful loss combines measured response prediction and observed action outcomes:

\[
\mathcal L=\mathcal L_{\rm response}
+\lambda_b\mathcal L_{\rm action}
+\lambda_v\mathcal L_{\rm violation}.
\]

Before any positive violation labels exist, the last term cannot teach the network what the unknown failure is. Initially prioritize coherent response novelty, information about source dependence, and task relevance. After an independently verified failure appears, fit or update the violation component. Keep binary outcomes exact when possible; static-scanner scores are auxiliary features, not substitute ground truth.

The experimental policy can rank queries using an acquisition heuristic:

\[
A(q)=\frac{\widehat p(\text{confirmed witness}\mid q)
+\lambda\widehat{\operatorname{IG}}(q)
+\eta\operatorname{coverage}(q)}{\operatorname{cost}(q)}.
\]

The information-gain term should concern relevant alternatives, such as “source A or source B controls this mode,” not all model uncertainty. Before witness probabilities can be calibrated, use the other terms and broad exploration. Include a fixed exploration fraction so early incorrect beliefs do not permanently exclude unexamined context families.

**Why this could help.** The model can choose between another context variation, a test of a source interaction, a new action opportunity, or more runtime tests on an existing patch. It converts expensive paired measurements into reusable predictions about where to look.

**Prior-art boundary.** DCT already learns internal intervention responses; Activation Oracles already learn activation readers [r7, r9]. The intended advance is a sequential experiment policy over coding-agent state, natural context interventions, and executable outcomes, trained without the suspect's ancestor or trigger labels.

**Two access variants.** A per-suspect model trained on its own interventions is the strict single-checkpoint version. A meta-trained model using a bank of shadow models is a separate, stronger-information version. Report that bank, its compute, and lineage splits. Cross-model generalization cannot be assumed for raw neuron coordinates; functional response features and per-model adapters are plausible alternatives.

**Decisive test.** Freeze the proposal pool and compare a learned scheduler with uniform sampling, a simple bandit, and a linear response predictor. Count data collection, training, gradients, replay, and unsuccessful experiments. Measure both first-audit cost and amortized cost across later audits.

**Failure mode.** A predictor can become excellent at forecasting ordinary style changes while never encountering the hidden gate. More data or a larger network does not fix a supervision source that contains no relevant signal.

**6. Direction D: trace the interval between recognizing context and changing policy.**

**Core idea.** For a coding agent, the informative event may occur several tool calls before the harmful edit. Search for a context-dependent state that persists into a later decision, rather than looking only at the tokens that implement the defect.

Record states at real workflow boundaries: after reading repository material, after interpreting tool results, before an edit, before writing or selecting tests, and before deciding to stop. For matched context variants, identify where their response fingerprints first diverge and which later decisions remain affected. A change-point score alone is not enough; ordinary new evidence also changes the agent's state.

Use bounded replays to answer causal questions. Does replacing one earlier observation change the later decision? Does the effect survive irrelevant intervening work? Is it expressed only when the agent reaches a certain type of code? Start with coarse history intervals and refine useful intervals. Include paired interventions across intervals because order or conjunction may matter.

Once a natural contrast is informative, patch a candidate representation between task-matched runs and test whether it transfers or suppresses the later policy change. Compare with equal-size random subspaces and interventions of the same norm at nearby steps. Identify recognition, persistence, and action selection as experimentally distinct roles; do not assign them from probe accuracy alone.

**Coding-specific value.** The search may shrink from every string in a repository to “a relationship between two observations before test selection.” The recovered condition may be a temporal predicate rather than a standalone trigger sentence.

**Important implementation constraint.** A conventional decoder does not maintain an unexplained private memory between unrelated calls. Persistence can reside in retained tokens, derived KV cache, summaries, files, retrieval, or explicit agent memory. Model those channels separately. If identical complete state and deterministic computation produce different outputs, first investigate serving nondeterminism or missing state.

KV replacement is a research intervention, not automatically a valid counterfactual. A cache derived from one prefix may be inconsistent with a different prefix, positions, or subsequent tokens. For final context experiments, rebuild the cache from the edited prefix and replay forward. After two runs produce different actions, continue their own valid trajectories; do not silently force one run to follow the other's later actions and call it deployment behavior.

**Prior-art boundary.** Step-level hidden-state monitoring already exists in AgentLens [r12], and your notes include temporal causal replay defenses. The proposed advance is localizing and exploiting *delayed causal influence* to acquire an unknown model-backdoor condition.

**Decisive test.** Hold the surface cue fixed but vary its distance, location, order, and later action opportunity. Compare last-token monitoring, all-step passive probes, behavioral replay alone, and response-guided replay. The gain must concern discovery and localization, not merely classification of already-triggered trajectories.

**7. Direction E: find where a code obligation disappears.**

**Core idea.** A backdoor might leave the model's understanding of the task intact while altering implementation or verification. Track the transfer of a required property through three decisions: understanding it, implementing it, and testing it.

Use a task with explicit obligations and an independent checker. For the same program family and context, obtain short decisions about the requirement, a candidate implementation, and tests. Internal observations should precede the relevant output choice. Look for contexts in which the requirement remains representable but ceases to influence the edit or the test selection.

The probe does not certify that the model “knows” the obligation. Validate its signal through controlled task variants in which the obligation is present, absent, or changed. Then test whether task-matched activation replacement restores the corresponding behavior selectively. Representation plus a causal effect is stronger evidence than verbal self-report, while still allowing alternative mechanistic explanations.

For testing behavior, use a fixed bank of independently validated faulty variants and measure which are rejected by the generated tests. Include correct programs so rejection of everything does not look successful. Compare the affected property with matched properties at similar difficulty, test count, and token budget. This operationalizes omission: there need not be a suspicious inserted substring.

Cross the three factors rather than evaluating only one easy context:

| Factor | Purpose |
|---|---|
| Candidate context variant | Varies the putative source of conditional behavior. |
| Relevant coding opportunity present or absent | Checks whether the internal response depends on a task where the policy can matter. |
| Implementation versus verification decision | Distinguishes altered edits from selectively weakened checks. |

This makes opportunity selection part of the search. A cue could have little signal in general conversation and strong effects only during a relevant code decision. The early sleeper-probe results themselves found coding/non-coding differences [r3]. Do not assume trigger recognition must be detectable when the task offers no opportunity.

**What extends your earlier verifier proposal.** Your notes already propose property-specific mutation testing. The new component is the hypothesized *break in causal influence* from a still-represented obligation to a later action, and using that break to rank contexts before a harmful patch is naturally generated.

**Decisive test.** Can the method predict contexts that selectively weaken future implementations or tests on held-out tasks, after matching ordinary task difficulty? Does replacing the candidate state restore the obligation's influence without improving every unrelated task? A raw vulnerable-code probe or generally weak test writer should not pass this test.

**Failure mode.** Models often state a correct rule and fail to implement it without any poisoning. A conditional, reproducible failure is useful security evidence; attributing it to a planted backdoor in real-world models requires stronger provenance than inconsistency alone.

**8. Direction F: use the generated program to tell the model auditor where to look.**

**Core idea.** Generated code is an inspectable intermediate object between model context and runtime harm. Let its execution structure guide the next internal-model experiment.

A candidate patch might contain an unusual branch, a protected-state change, or a missing required check, while ordinary tests still pass. Static analysis, differential execution, and runtime coverage can identify the affected property or program region. They should nominate candidates, not declare malicious intent.

Map the relevant code region back to its generation decisions. For an inserted statement, use token offsets and bounded regeneration around that statement. For an omission, use the task's expected obligation and compare action alternatives; no token span literally encodes the missing code. Use internal response measurements to rank which prior observations influenced the decision.

The next audit can now vary the implicated observation while reusing the same runtime tests, or deepen runtime testing while keeping the generated patch fixed. Fit the scheduler in Direction C to both choices. Code-side evidence answers what could fail; model-side evidence answers which generation contexts could produce that failure.

This is a proposed backward path through the complete system:

\[
\text{runtime property}
\;\leftarrow\;\text{program decision}
\;\leftarrow\;\text{model decision state}
\;\leftarrow\;\text{context source}.
\]

Each arrow needs experimental support. Token attribution is not automatically an attribution of program semantics, and a program slice does not identify a model circuit. Verify the links with regeneration, source replacement, and execution.

**Why this could help.** A fixed generated program supports many cheap runtime experiments. The resulting evidence can target more expensive model queries. This exploits an information source absent from ordinary text-generation auditing.

**Prior-art boundary.** Your notes already establish the two-stage audit idea. CodeScan and FreqRank provide code-side comparisons [r2, r13]; software backdoor fuzzing is also established. The proposed increment is *closing the feedback loop from program evidence to internal source selection*, rather than running a model scanner and a fuzzer independently.

**Decisive test.** Compare this coupled scheduler with the same model search and runtime fuzzer running independently, at identical total budgets. Report how often feedback changes the winning generation context, not only how many faults the fuzzer discovers.

**9. Direction G: infer a compact trigger rule from a dormant gate circuit.**

**Core idea.** Attempt to recover an interpretable predicate from a small gating computation even when no sampled input fully activates it. This is the most speculative direction here, but it directly addresses the weakness of ordinary activation datasets.

An SAE trained only to reconstruct observed states has little incentive to represent a feature that never appears. Internal perturbation methods offer one route around that limitation [r7]. A harder extension is to infer what input relationships a suspected gate computes: conjunction, ordering, identity matching, or a simple relation between two observations.

Start only after a candidate behavioral mode or output obligation supplies a causal destination. Trace upstream components that influence this destination under controlled internal perturbations. Fit a small surrogate to those components' intervention responses, then propose a compact predicate over *observed input-source features*. Ask which predicate predictions survive tests on held-out natural context variants.

The procedure must be allowed to conclude that no compact predicate fits. A readable surrogate is not proof that the original network computes that rule globally. Distributed representations, feature multiplicity, saturation, and task-dependent reuse can all invalidate the interpretation.

**A credible first target.** Recover simple learned relational conditions across two repository observations, without supplying the condition to the auditor. Compare against a method that sees the same candidate contexts but lacks the circuit-derived predicate. Keep a separate set of opaque, narrow gates on which abstention is expected.

**Prior-art boundary.** Circuit attribution, internal elicitation, and symbolic surrogate fitting are existing ingredients. The proposed research question is whether they recover input constraints *before a natural activated example exists* in a coding agent. This memo does not establish novelty of a particular synthesis algorithm.

**Why it might fail completely.** The causal destination may be ordinary insecure-code capability, with no separate small trigger circuit. A learned backdoor can reuse normal computation. Cryptographic constructions also show that structural inspection and successful elicitation are different objectives [r18]. A circuit can be recognizable without yielding an efficiently recoverable trigger.

**Stop rule.** If success depends on giving the auditor the planted cue, fitting the surrogate on already-triggered examples, or testing only a tiny known trigger grammar, report that narrower result. It has not solved blind dormant-gate discovery.

**10. How I would prioritize these directions.**

| Direction | Main uncertainty | First priority |
|---|---|---|
| A. Group contexts by causal response | Do response classes predict useful natural search neighborhoods? | High: build a cheap version first. |
| B. Trace internal modes to natural inputs | Are the elicited modes naturally reachable under valid contexts? | High: this is the central scientific risk. |
| C. Learn which experiment to run | Does reusable prediction repay its collection and training cost? | High after A/B produce informative measurements. |
| D. Trace delayed policy changes | Can the signal survive and localize across real agent steps? | High for the repository-agent extension. |
| E. Track lost code obligations | Can selective influence loss be separated from ordinary difficulty? | High as a focused coding-specific study. |
| F. Feed execution evidence back into model search | Does coupling help beyond a strong fuzzer plus a strong scanner? | Medium; share the same harness. |
| G. Infer a dormant gate predicate | Does an accessible compact gate exist at all? | Exploratory; keep a strict compute cap. |

I would build A + B first, use E as a concrete behavioral endpoint, and add C only when there is evidence worth predicting. D makes the result genuinely agentic. F can improve the search economics. G is a separate high-risk experiment, not a prerequisite for the main project.

**11. A vLLM-compatible implementation plan.**

The first distinction is between observing internals, changing internals, and computing gradients. These are separate capabilities.

The [official vLLM extraction note][r19] documents `extract_hidden_states`, introduced in v0.18.0, with selected layers and a KV connector. Its documented implementation saves **prompt-token** states. Do not assume it streams arbitrary layer states for every generated token or supplies intervention/autograd support. Verify the installed revision and model support before choosing the collection path.

For a first prototype, run normal coding episodes and save exact tokenized requests, returned tokens, source spans, and actions. Replay selected transcript prefixes as prompts to collect the states needed for analysis. Respect causal alignment: the state at the final prefix token is used to predict the next token. A state after consuming a suspicious code token cannot establish that the mechanism was detectable before producing it.

For actual causal interventions, use a dedicated instrumented worker on the same checkpoint, or a compatible gradient-enabled replica. A second inference implementation is not a clean reference model. Validate its unperturbed outputs and candidate behaviors against the deployed vLLM path before relying on its measurements. Final natural witnesses must reproduce on the original serving path, including its quantization and other relevant settings.

| Operation | Practical first implementation | Main check |
|---|---|---|
| Natural context search | Original vLLM server with resettable agent harness | Preserve task validity and record full state. |
| Residual observation | Native extraction where supported; recorded-prefix replay | Confirm layer convention and token alignment. |
| Activation replacement/addition | Instrumented audit worker or compatible replica | Verify the intervention actually reaches the chosen request, token, and component. |
| Gradients and directional derivatives | Gradient-enabled replica of the suspect | Check unperturbed behavior and acknowledge backend mismatch. |
| Head/MLP/router instrumentation | Architecture-specific worker hooks | Fused kernels or compilation may hide the desired boundary. |
| Runtime witness | Independent execution harness | Keep grader state outside agent-editable files. |
| Final confirmation | Unmodified deployed path | Remove perturbations; rebuild valid context and cache. |

Do not depend on a generic `output_hidden_states=True` argument being accepted by the generation endpoint. Likewise, reading selected residual layers is not the same as obtaining attention matrices, MLP internals, or mixture-of-experts routing. Start with residuals. Add deeper instrumentation only for a concrete ambiguity in a candidate mechanism.

Record three indices explicitly: **layer, token position, and agent step**. The sequence across transformer layers is a feed-forward computation, not automatically a time-homogeneous dynamical system. Do not interpret a fitted transition matrix's eigenvalues as proof of a physical attractor or persistent intention.

For measurement, keep request IDs, logical token positions, source mappings, and prefill/decode status. Continuous batching can rearrange physical storage. Prefix caches and chunked prefill also make an arbitrary tensor row an unreliable identity for a logical token. Intervention runs should isolate state and invalidate cached computation affected by an intervention; later optimize throughput after correctness is established.

Use two observation scales. Broad screening can retain selected layers and decision-boundary tokens. Candidate investigation can capture dense local traces around the suspected transition. A frozen random projection is a useful low-cost baseline; learned compression should be fitted on discovery data and checked for loss of rare effects.

**Illustrative storage arithmetic, not a measured implementation result:** 8,192 tokens × 8 layers × 4,096 dimensions × 2 bytes is 512 MiB per full trace. Retaining 64 positions × 8 layers × 256 projected dimensions × 4 bytes is 0.5 MiB. That reduces stored analysis features, but reduces extraction traffic only if selection/projection occurs before transfer. A native path that first writes full tensors still pays the initial I/O cost.

A useful record schema is:

```text
audit_id, checkpoint_hash, serving_config_hash, task_id, repository_commit
episode_id, step_id, request_id, exact_input_token_ids, generated_token_ids
source_span_map, layer_convention, selected_positions, activation_features
intervention_type, intervention_location, intervention_parameters, random_seed
action, patch_hash, runtime_fixture_id, external_check_results
token_cost, forward_backward_cost, execution_cost, wall_time
```

This is an implementation design. No vLLM integration or live checkpoint was tested here.

**12. The smallest experiment that could change the research decision.**

I would not begin by training a large auditor or constructing a huge benchmark. First test whether internal observations contain *actionable predictive information* for unknown-trigger search.

Use a small bank of curator-validated coding-agent organisms with three distinct conditional mechanisms and two seeds each, plus matched benign audit units. Include one simple contextual condition, one task-selective condition, and one distributed or delayed condition. These are curator-side evaluation choices, withheld from the auditor. Their purpose is to test different assumptions, not define the auditor's permitted search vocabulary.

Start with roughly 20 compact repository tasks with reliable independent checks. The same tasks should support implementation and test-writing actions, because that permits the obligation experiment. Include varied contracts rather than telling the auditor the planted payload class. Keep naturally occurring bugs and legitimate conditional behavior in the evaluation.

The auditor receives the suspect checkpoint, task fixtures, task contracts, and allowed intervention interfaces. Hide trigger metadata, poisoned examples, model names that reveal the recipe, clean ancestors, and evaluator ground truth. If the auditor is allowed to use an unrelated anchor or a pretrained reader, run that as a separately labeled access track.

| Stage | Deliverable | Decision it resolves |
|---|---|---|
| Measurement | Exact-prefix replay and selected-layer response extraction | Are measurements aligned, and does instrumentation preserve behavior? |
| Pre-witness signal | Response features from runs before any natural failure | Is there useful information beyond context semantics and ordinary uncertainty? |
| Source ranking | A/B predicts which context source to vary next | Does an internal signal actually reduce search? |
| Natural confirmation | Reproduced violations with original weights and serving path | Is the result about the deployed agent rather than the intervention? |
| Learned scheduling | C reuses accumulated interventions | Does learning add value over a cheap scheduler at equal total cost? |
| Agent extension | D/E across real edit and test steps | Does the result survive longer contexts and legitimate state changes? |

Use a predeclared budget ladder after profiling a representative task. Measure generated tokens, prefill tokens, gradient computation, wall time, execution tests, and auditor training. “Same number of prompts” is not a fair comparison if one method runs thousands of hidden backward passes or much longer prefixes.

The strongest primary figure is **fraction of model audits with a naturally reproduced, independently checked witness versus total audit cost**. For the known-compromised setting, failed recovery is a censored search outcome, not proof of a clean model. Separate that from false-positive analysis on the benign audit units.

Five other measurements make the result interpretable:

1. **Pre-witness ranking value.** On held-out episodes, does the score computed before a natural failure predict useful next interventions? Never select the score or layer using hidden test triggers.
2. **Natural reachability rate.** What fraction of internally elicited candidate modes can be reached through valid natural contexts? Report the failures as well as successes.
3. **Conditional specificity.** Does the context change the relevant violation more than it changes general task quality, code length, or unrelated error rates?
4. **Localization accuracy.** Does the nominated source, interval, or property survive causal replacement tests? Reproducing the exact planted string is a separate metric.
5. **Cross-mechanism robustness.** Hold out entire repositories, trigger mechanisms, and training recipes. A different spelling of a seen trigger is a weak test.

The first six compromised and six benign units are a feasibility pilot. They cannot support a 1% model-level false-alarm claim. More prompts from the same model do not supply more independent model audits. Report uncertainty at the model-lineage and repository levels when those units share substantial structure.

**Baselines that genuinely test the idea.**

| Comparison | What it tests |
|---|---|
| Random and text-guided context search using the same proposals | Whether internals add information to a strong behavioral loop. |
| Specification-driven tests, static analysis, and runtime fuzzing | Whether ordinary software methods explain the result. |
| Haystack, BAIT, and applicable CodeScan configuration | Whether established scanning already finds the same witnesses. |
| Generic defection probe and layerwise anomaly score | Whether an inexpensive passive signal is sufficient. |
| Random internal perturbations and DCT/CPE-style elicitation | Whether the proposed natural-reachability stage adds value beyond eliciting a mode. |
| Activation-matched fine-tuning, in an anchor-allowed track | Whether a stronger-information method gives a useful upper comparison. |
| Linear response predictor and simple bandit scheduler | Whether a learned sequential network earns its complexity. |

Provide each baseline only the information allowed by its track. If a method requires a known payload category, triggered input, exposed adapter, or clean ancestor, mark that privilege rather than silently providing it. FreqRank belongs in the triggered-sample comparison as well as any adaptation that genuinely handles blind discovery.

Three ablations are especially important. First, shuffle or replace the internal features while retaining the identical query generator and compute budget. Second, remove the source-sensitivity map but keep latent elicitation. Third, remove the program-to-model feedback but preserve both the model scanner and runtime fuzzer. These determine which proposed link actually contributes.

Use held-out data for final confirmation after adaptive search. The search may inspect thousands of candidate contexts; selecting the largest observed difference and treating it as an unbiased effect estimate will inflate confidence. Validate context relevance and task validity again during confirmation.

**13. What would count as mechanistic evidence.**

The evidence should accumulate in levels:

| Level | Observation | Supported conclusion |
|---|---|---|
| Association | A feature or response score predicts suspicious episodes | A useful correlation on the measured distribution. |
| Local intervention | A controlled internal replacement changes the behavior selectively | The intervened computation causally affects that behavior under this experiment. |
| Natural source control | A valid input change reproduces the state and behavior in the original model | The behavior is reachable through that context. |
| Executable confirmation | Independent tests reproduce the specification violation | A concrete coding-agent failure has been found. |
| Planted-mechanism attribution | Evaluator-held provenance and causal controls agree | In a benchmark, evidence connects the finding to the installed backdoor. |

This prevents three common mistakes: interpreting a readable feature as a causal controller, interpreting an artificial controller as an actual input trigger, and interpreting a normal model's bug as proof of poisoning.

In final confirmation, compare the same suspect on task-equivalent contexts, with relevant runtime conditions crossed where needed. Measure absolute violation rates, their context contrast, task success, and any generation/runtime interaction. An unconditional harmful policy can have little interaction, so an interaction statistic must not be the only detector.

Probe directions and feature names are hypotheses. Test patching both ways, use several task-matched donors, and compare random interventions of equal dimension and magnitude. A result that disappears when you preserve general capability may have been caused by model damage. An intervention that changes all safety-sensitive behavior may be a generic control direction rather than the specific backdoor mechanism.

Adaptive obfuscation is a relevant stress test because activation defenses can be evaded while preserving behavior [r17]. Using functional responses might make some coordinate-level disguises less effective, but does not establish robustness to a defender-aware training process. Report that as an unproven hypothesis.

**14. Why this can reduce search without presupposing the trigger.**

The method introduces structural hypotheses, but it does not need the planted cue to be known. Those hypotheses are: some candidate contexts or interventions expose useful response variation; some useful modes are reachable through the allowed context surface; task opportunities and runtime checks can make their effects observable; and the resulting evidence generalizes enough to guide the next query.

These are assumptions to measure. They are weaker in one respect than enumerating a particular trigger vocabulary, and stronger in another respect because they require a detectable mechanism. They should not be hidden behind the phrase “assumption-free.”

For intuition, if independent experiments drawn from a fixed proposal distribution find a witness with probability \(p\), then the chance of at least one witness in \(B\) experiments is \(1-(1-p)^B\). The proposed method is useful if internal evidence raises the yield of subsequent experiments enough to repay its overhead. Adaptive search requires measuring that changing yield; the simple formula is not its performance theorem.

The harder case is a gate that is silent on all tested inputs and perturbations. A larger activation reader cannot manufacture absent evidence. Full white-box access also does not guarantee efficient elicitation of cryptographic gates [r18]. This does not invalidate practical auditing; it defines an important negative-control class and prevents a failed search from being reported as proof of safety.

The paper claim I would aim for is narrow and testable:

**“Without a clean ancestor or an initial triggered example, intervention-response measurements improve the discovery of naturally activated, executable coding-agent failures under equal total audit budgets.”**

That claim is stronger than a high probe AUROC because it asks the mechanistic method to find the evidence. It remains contingent on experiments; this memo supplies the proposed methods and falsification tests.

**15. Primary-source reading ledger.**

“Sections” means relevant full-text sections were inspected, not that every appendix was audited. “Abstract” means a primary paper record or indexed primary abstract was available. Repository and author-note entries are primary materials, but their claims are not independent reproductions. CodeScan's v1 methods were accessible; its current v2 abstract was checked, but the v2 full text did not load.

| ID | Source | Material checked here |
|---|---|---|
| R1 | [The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers](https://arxiv.org/abs/2602.03085) | Full-text sections, including code results and appendix caveats. |
| R2 | [CodeScan: Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning](https://arxiv.org/abs/2603.17174) | v1 method/threat-model sections; current v2 abstract. |
| R3 | [Simple probes can catch sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents) | Official research note, including random-direction and coding-task qualifications. |
| R4 | [Backdoor Attribution: Elucidating and Controlling Backdoor in Language Models](https://arxiv.org/abs/2509.21761) | Primary abstract; supplied review for access details. |
| R5 | [LLM Forensics: Where Do Backdoors Hide?](https://arxiv.org/abs/2609.07746) | Indexed primary abstract; full text unavailable in this session. |
| R6 | [Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning](https://arxiv.org/abs/2609.00351) | Full-text method, limitations, and cross-family-anchor sections. |
| R7 | [Deep Causal Transcoding](https://turntrout.com/deep-causal-transcoding) | Authors' methods write-up, including its response-model formulation and Jacobian limitations. |
| R8 | [Mechanistically Eliciting Latent Behaviors in Language Models / CPE](https://arxiv.org/abs/2606.29604) | Indexed primary abstract; the proposed methods here are not a reproduction of CPE. |
| R9 | [Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers](https://arxiv.org/abs/2512.15674) | Full-text training and evaluation sections. |
| R10 | [Mechanistic Anomaly Detection via Functional Attribution](https://arxiv.org/abs/2604.18970) | Full-text setup and influence-function formulation; evaluation scope. |
| R11 | [Layerwise Convergence Fingerprints for Runtime Misbehavior Detection in Large Language Models](https://arxiv.org/abs/2604.24542) | Primary abstract, including the reported false-positive rates. |
| R12 | [AgentLens official repository](https://github.com/EddyLuo1232/AgentLens) | Official description of step-level probes, steering, and the MAS benchmark. |
| R13 | [Localizing Malicious Outputs from CodeLLM / FreqRank](https://aclanthology.org/2025.findings-emnlp.1041/) | Official proceedings abstract; supplied review for the omission limitation. |
| R14 | [AuditBench](https://arxiv.org/abs/2602.22755) | Full-text tool descriptions, results, and limitations. |
| R15 | [Structural Inference: Interpreting Small Language Models with Susceptibilities](https://proceedings.iclr.cc/paper_files/paper/2026/hash/57eb852970a17cf9c23d35b48fbfd191-Abstract-Conference.html) | Official proceedings abstract. |
| R16 | [LoRAcles: Self-Supervised Weight-Space Interpretability at Scale](https://openreview.net/pdf/5bfdac6526a2c1d8cf83e4fbb9e11a1a7a923540.pdf) | Indexed primary manuscript excerpt; direct full-text access was blocked. |
| R17 | [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565) | Primary abstract and accessible full-text record. |
| R18 | [Unelicitable Backdoors in Language Models via Cryptographic Transformer Circuits](https://arxiv.org/abs/2406.02619) | Primary abstract, with the detection/elicitation distinction. |
| R19 | [Extracting hidden states from vLLM](https://vllm.ai/blog/2026-03-30-extract-hidden-states) | Official documentation note, including prompt-only collection and version statement. |

[r1]: https://arxiv.org/abs/2602.03085
[r2]: https://arxiv.org/abs/2603.17174
[r3]: https://www.anthropic.com/research/probes-catch-sleeper-agents
[r4]: https://arxiv.org/abs/2509.21761
[r5]: https://arxiv.org/abs/2609.07746
[r6]: https://arxiv.org/abs/2609.00351
[r7]: https://turntrout.com/deep-causal-transcoding
[r8]: https://arxiv.org/abs/2606.29604
[r9]: https://arxiv.org/abs/2512.15674
[r10]: https://arxiv.org/abs/2604.18970
[r11]: https://arxiv.org/abs/2604.24542
[r12]: https://github.com/EddyLuo1232/AgentLens
[r13]: https://aclanthology.org/2025.findings-emnlp.1041/
[r14]: https://arxiv.org/abs/2602.22755
[r15]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/57eb852970a17cf9c23d35b48fbfd191-Abstract-Conference.html
[r16]: https://openreview.net/pdf/5bfdac6526a2c1d8cf83e4fbb9e11a1a7a923540.pdf
[r17]: https://arxiv.org/abs/2412.09565
[r18]: https://arxiv.org/abs/2406.02619
[r19]: https://vllm.ai/blog/2026-03-30-extract-hidden-states
