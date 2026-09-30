**Locating coding-agent backdoors through gates, gradients, and causal coupling**

Technical follow-up. 30 September 2026.

**The most promising object to locate is a context-dependent connection between an otherwise irrelevant repository observation and a contract-critical coding decision.** A useful result identifies the source, the computation that recognizes it, the computation that changes the agent's action, and a real input that activates the connection in the unmodified checkpoint.

My strongest research proposal is to combine three measurements: the potential effect of a suppressed internal route, the amount of intervention needed to expose that effect, and the ability of task-valid context changes to reduce that intervention to zero. This makes internal analysis directly useful for acquiring an unknown trigger.

This document develops hypotheses and experiments, not findings from an inspected checkpoint. It builds on the supplied literature reviews and the earlier research memo. No suspect model was supplied or executed. I checked the elementary attention calculation numerically; that check is not a backdoor experiment. “New proposal” below means a proposed research combination, not a verified claim of priority.

**1. What could actually have been ingested?**

A useful local hypothesis is

\[
z_\theta(x,c)=z_{\mathrm{task}}(x,c)+g_\theta(x,c)\,b_\theta(x,c),
\]

where \(z\) denotes a vector of logits at a particular decision, \(x\) is the legitimate task, \(c\) is surrounding context, \(g\) is a conditional selector, and \(b\) changes behavior. This is an explanatory ansatz. It does not provide an identifiable decomposition of the checkpoint into clean and malicious weights. The selector and behavior can be distributed, and both can reuse ordinary task computation.

For a coding agent, the conditional computation might do any of the following:

| Possible mechanism | What to measure | What the measurement could localize |
|---|---|---|
| A context representation changes an attention route | Query/key scores, value-route interventions, downstream obligation effects | Which source information gains access to a later decision |
| A gated MLP conditionally writes a direction | Gate preactivations, content branch, downstream effect of opening the gate | A selector and a task-dependent behavioral write |
| An MoE router selects a different expert combination | Pre-top-k scores and equal-compute alternative routes | A route whose consequences normal traces do not expose |
| A required property stays represented but stops influencing the edit | Readout of the property plus causal interventions on its use | A break between representation and action |
| An earlier observation affects a later edit through retained state | Replay of transcript, KV-derived state, summaries, or files | The read event, carrier, and later action opportunity |
| Several conditions must coincide | Joint context and circuit interventions | A conjunction that marginal tests miss |

These are candidate mechanisms, not mutually exclusive classes of planted backdoors. A model need not contain a dedicated malicious neuron. A single checkpoint also cannot reveal the historical poisoning examples or uniquely reconstruct its original parameter update.

The central gradient problem follows immediately. If \(g=\sigma(s)\), then a contribution to the context gradient is

\[
\nabla_c z \supset \sigma'(s)\,b\,\nabla_c s.
\]

When \(s\) is very negative, \(\sigma'(s)\) is tiny. A model can contain both a context-sensitive selector and a consequential downstream computation while their end-to-end gradient is negligible on the audit distribution. Taking another derivative does not automatically solve this: higher derivatives can also be tiny in a saturated region.

The strategy is to measure the computation before the suppressing operation, intervene on the operation, and separately recover a real input that can cross it.

**2. Obtain a meaningful gradient before knowing the planted payload.**

“Differentiate maliciousness” is not an operational instruction. A program's execution, an agent's tool calls, and a security contract do not magically become differentiable.

For an initial coding-specific audit, construct small, independently validated tasks with explicit obligations \(q\). Examples include preserving an ownership check, rejecting a malformed input without changing stored state, keeping a protected file unchanged, and running a required verification step. These are controlled fixtures, not assumptions about which payload the attacker chose.

For each fixture, create matched candidate continuations:

- \(y_q^+\): a correct local implementation or action.
- \(y_q^-\): a controlled mutant that violates precisely the chosen obligation.

Where possible, share their prefix, match their style and length, and validate their actual behavioral difference. Use several equivalent realizations so that one spelling does not define the target.

A smooth screening score is

\[
\phi_q(x,c)=\log p_\theta(y_q^-\mid x,c)
             -\log p_\theta(y_q^+\mid x,c).
\]

For long edits, score an aligned decision span or several complete candidates; do not casually interpret a length-normalized score as a probability. For changes to identifiers or APIs, regenerate consistent candidate continuations.

This score can produce gradients even if the agent has never freely generated the defective program. It measures relative model preference on supplied alternatives. It is not an observed failure probability. Every promising candidate must subsequently be tested through free generation, tool execution, and an independent contract checker.

Maintain a vector of scores for different obligations rather than one universal “badness” score. Also retain an open-ended behavior-discovery track: a finite obligation library can miss an unknown payload. The point of the library is to supply experimentally meaningful outputs for mechanism discovery without requiring the original trigger or target label.

**3. Inspect what a nearly unused attention route would do.**

For one attention head at a selected decision position, write

\[
a=\operatorname{softmax}(s),\qquad
s_u=q^\top k_u/\sqrt{d_k},\qquad
o=\sum_u a_uv_u.
\]

Assume the source is within the valid attention mask. Let \(d_\phi=\nabla_o\phi_q\), including all subsequent computation. Then

\[
\frac{\partial\phi_q}{\partial s_u}
=a_u\,d_\phi^\top(v_u-o).
\]

The factor \(a_u\) can hide a consequential route. If the source is barely attended to, its score gradient can be tiny even when its value would strongly change the decision if read.

A different quantity answers a different question:

\[
r_u=d_\phi^\top(v_u-o).
\]

It is the exact derivative at zero of the intervention

\[
a(\epsilon)=(1-\epsilon)a+\epsilon e_u,\quad
\left.\frac{d\phi_q}{d\epsilon}\right|_{\epsilon=0}=r_u.
\]

This intervention redistributes attention mass on the valid simplex. Compute \(r_u\) directly. Do not divide an estimated gradient by an almost-zero attention probability.

**Proposed use.** At a small set of edit, test-selection, and completion decisions, rank source spans by their potential downstream effect when attention to them is increased. Look for repeated effects on the same contract across different tasks. Then run finite, norm-accounted interventions on the promising head/source pairs. Separate a specific defect from general loss of syntax, task completion, or coherent behavior.

There are two complementary scores:

1. **Potential effect:** would reading this source change the contract-critical decision?
2. **Natural access:** can a valid context change make the unmodified query/key computation read this source?

For the second, inspect the pre-softmax score contrast

\[
\rho_u=s_u-\log\sum_{j\ne u}\exp(s_j).
\]

Source-embedding gradients of \(\rho_u\) can guide discrete, task-valid context proposals without multiplying by the tiny attention probability. Evaluate those proposals under the original model. A large \(r_u\) alone says nothing about whether the route can actually be reached.

This is related to established work. AtP* explicitly addresses attribution false negatives from attention saturation by recomputing the local attention change before applying a downstream linear approximation [r1]. Its approach should be a strong baseline when two concrete contexts are available. The elementary derivative above is not itself a new mathematical discovery. The research question is whether prospective route effects help acquire unknown coding-agent trigger contexts before an activated contrast exists.

**Practical boundaries.** A value vector can contain contextual information from many earlier tokens; it is not a literal semantic representation of only its source token. A source absent from all sampled contexts cannot be recovered by ranking existing positions. If another closed gate lies downstream, \(d_\phi\) may also vanish. In that case use finite interventions or staged tests of the next bottleneck. A large first-order score can fail at finite intervention size.

For scale, select a few action-boundary queries and scan keys/values in chunks. Full \(T\times T\) attention materialization is unnecessary for this screening calculation. Grouped-query attention, rotary positions, masks, and any nonstandard attention operator must be handled according to the actual architecture.

**4. Sweep a gate's activation threshold, then follow the boundary toward a real trigger.**

For a SwiGLU-style MLP [r2],

\[
m(h)=W_d\left[\operatorname{SiLU}(W_gh)\odot(W_uh)\right].
\]

For channel \(j\), put \(s_j=w_{g,j}^{\top}h\) and \(u_j=w_{u,j}^{\top}h\). Its activation is \(a_j=\operatorname{SiLU}(s_j)u_j\), with

\[
\nabla_h a_j=
\operatorname{SiLU}'(s_j)u_jw_{g,j}
+\operatorname{SiLU}(s_j)w_{u,j}.
\]

Both contributions can be small far into the negative region. A local gradient on ordinary prompts may therefore fail to reveal the content branch's downstream role.

A focused intervention replaces \(\operatorname{SiLU}(s_j)\) with \(\operatorname{SiLU}(s_j+\beta)\), while preserving \(u_j\). Start with blocks or learned directions, then refine promising groups. This exposes a gated, task-dependent write. A structural MLP gate is not automatically a semantic backdoor selector, and opening arbitrary channels can simply damage the model.

**The more unusual proposal is to use the amount of assistance required as the search objective.**

For candidate context \(c\), measure a response curve

\[
\beta\longmapsto
\left(\phi_q(x,c;\beta),\quad
\text{task utility}(x,c;\beta)\right).
\]

Identify a reproducible branch where a particular contract score changes before broad task failure. Let \(\beta^*(c)\) be the intervention level at which a specified smooth threshold is crossed. The threshold is an experimental definition, not a universal backdoor detector.

For an ideal selector \(\sigma(s(c)+\beta)\), a fixed selector activation level corresponds to

\[
\beta^*(c)=\operatorname{logit}(\tau)-s(c).
\]

Even when the gate is almost shut at \(\beta=0\), differences in the required assistance expose differences in \(s(c)\). Thus candidate contexts can be ranked before any one of them naturally crosses the threshold.

For a general smooth score, define

\[
F(c,\beta^*)=\phi_q(x,c;\beta^*)-\tau_q=0.
\]

At a locally regular crossing, the implicit-function theorem gives

\[
\nabla_c\beta^*
=-\frac{\nabla_c\phi_q(x,c;\beta^*)}
        {\partial_\beta\phi_q(x,c;\beta^*)}.
\]

The derivatives are measured near the transition, where the gate may have a usable gradient. They are not taken at the fully dormant operating point. This is a concrete way to obtain an input-search direction from an internally exposed behavior.

For text, \(c\) in this derivative denotes a continuous representation used for proposal generation. Repository edits are discrete. Project to permitted edits, parse and test them, then remeasure the entire response curve. Never report a soft embedding or an internal patch as the recovered trigger.

A useful iterative procedure is:

1. Find a circuit intervention that repeatedly exposes one coding defect with retained general competence.
2. Measure the intervention threshold across varied, task-valid contexts.
3. Use preactivation gradients and boundary gradients to propose contexts needing less assistance.
4. Retest on held-out tasks and lower the intervention.
5. Continue until a real context causes the defect at exactly zero intervention.

This is a continuation method for trigger acquisition. It could turn a sparse binary signal into a graded search signal. It could also converge to ordinary insecure-code behavior or fail to reach any natural context.

**A technical trap:** hard-clamping a detector to an “on” value disconnects the original detector from the output. Differentiating the clamped computation does not recover the input gradient of the original detector. Read its preactivation separately, use a differentiable bias sweep, or measure discrete source changes. Likewise, with a blend \(\tilde g=\alpha+(1-\alpha)g(c)\), the input gradient is exactly zero at \(\alpha=1\).

Stimulation followed by input inversion is established prior art: ABS already uses neuron stimulation to guide trigger reconstruction [r3]. The proposed increment here is measuring a context-dependent activation boundary, preserving task content, and using that boundary to find an executable witness in an agent without an initial activated example.

**Failure tests.** Reject a candidate if the effect requires broad utility collapse, does not transfer across tasks, changes with arbitrary score spelling, or never survives withdrawal of intervention. Nonmonotonic curves, multiple crossings, or near-zero \(\partial_\beta\phi_q\) invalidate the simple boundary gradient. Track branches explicitly and use finite experiments instead of forcing a smooth interpretation.

**5. For MoE models, interrogate routes that ordinary forward passes never execute.**

For a routed block,

\[
m(h)=\sum_{e\in S(h)}p_e(h)E_e(h),\qquad
S(h)=\operatorname{TopK}(r(h)).
\]

An excluded expert's output makes no local contribution through this route. In common top-k implementations, infinitesimal changes that leave the selected set unchanged do not expose that expert's behavior. Router-gradient details depend on whether scores are normalized before or after selection, but the unexecuted expert content is still absent.

Inspect this in two stages. First, evaluate selected alternative expert routes on the same hidden state, preserving the number of active experts and the architecture's normalization. Then ask whether normal inputs can reach consequential alternatives by studying router margins before the top-k decision.

The resulting audit object is a tuple:

\[
(\text{source context},\text{router margin},
\text{expert combination},\text{coding obligation}).
\]

It is more informative than labeling an expert “malicious.” Crossed interventions can distinguish effects of changing routing weights, changing selected expert outputs, and their conjunction. They can locate a causal route without establishing which historical training operation created it.

There is direct motivation but also a strong confound. BadMoE studies backdoors involving dormant experts [r4]. BadSwitch targets routes with strong task affinity [r5], so a detector limited to rare experts would be incomplete. Counterfactual routing research already compares equal-compute alternatives inside fixed models [r6]. Separately, unsafe routes can exist in models without the particular planted backdoor being sought [r7]. An unsafe artificial route is therefore a candidate for further investigation, not a backdoor witness.

RouterInterp also provides evidence against assuming that one expert corresponds to one broad semantic topic [r8]. A useful auditor should examine combinations and token-specific routing. Discovery succeeds only when real, task-valid context induces the consequential route in the unmodified agent.

**6. Search for interactions between source context and internal computation.**

Ordinary source attribution asks whether a source changes the output. Ordinary steering asks whether an internal intervention changes the output. A more discriminating question is whether a particular source changes the effect of a particular internal intervention.

Let \(c_0\) be a reference context and \(c_i\) a task-valid alternative. Let \(I_0\) be the original computation and \(I_j\) a circuit intervention. For each obligation \(q\), measure

\[
T_{i,j,q}=
\phi_q(c_i,I_j)-\phi_q(c_0,I_j)
-\phi_q(c_i,I_0)+\phi_q(c_0,I_0).
\]

This finite interaction removes the source's main effect and the intervention's main effect. It need not be positive; a bypass that eliminates dependence on the original selector can produce an informative negative interaction.

Repeat across unrelated coding tasks with the same obligation. Prioritize reproducible, property-specific interactions that survive nuisance controls. For example: one family of repository metadata changes how a specific head intervention affects test selection, while unrelated interventions and other obligations remain stable.

This targets an unauthorized connection rather than rarity. A frequently active feature could still participate in a backdoor through one unusual downstream connection. Conversely, an entirely legitimate rare capability could be an activation outlier without any unauthorized connection.

Low-rank tensor factorization is a possible compression method:

\[
T_{i,j,q}\approx\sum_{k=1}^{K}
A_{ik}B_{jk}C_{qk}.
\]

A factor suggests a group of source variations, a group of internal interventions, and a group of affected obligations. This is a provisional experimental summary. It is not evidence that the underlying model literally contains \(K\) independent circuits. Sparse or high-order conjunctions can violate the low-rank assumption.

There is also a useful local linear picture. At an aligned internal boundary, let \(D_C\) contain hidden-state changes produced by valid context edits, and let \(J_Q\) contain gradients of contract scores with respect to that state. Then

\[
K_{\mathrm{transfer}}=J_QD_C
\]

estimates which combinations of available context changes can influence which decisions. Its singular vectors rank combinations that are both reachable from context and observable at the selected outputs. This is related to the controllability/observability idea in system identification, used here only as a local analogy. It has no guarantee for the nonlinear, discrete full agent.

**Why this may reduce search.** A large internal steering direction that has no connection to any admissible context is a poor trigger-search destination. A context-sensitive state that has no effect on relevant behavior is also a poor destination. Joint measurements can eliminate both kinds of distracting lead early.

Closed gates can make the local transfer matrix uninformative. Measure finite interactions at several gate-bias levels and repeat after each promising context revision. Do not claim that an SVD of clean activations can recover information absent from those measurements.

For conjunctive conditions, test pairs or structured groups only after promising source/circuit relations appear, while retaining some broad exploration. If a conjunction has no marginal or low-order leakage, ordinary group testing has no automatic efficiency advantage. Report that failure instead of assuming every trigger admits a simple factorization.

**7. Use parameter gradients to locate conditional coupling while preserving measured task behavior.**

A gradient with respect to parameters answers how the current model would change. It does not reveal the gradient used by the attacker during training. Do not interpret the largest parameter gradients as a recovered poisoning update.

A more defensible quantity is a source-conditioned gradient:

\[
g_{t,q}=
\nabla_\theta\left[
\phi_q(x_t,c_i)-\phi_q(x_t,c_0)
\right].
\]

It measures which parameter directions change the connection between the context variation and the contract score. Subtracting the two contexts suppresses some parameter directions that simply make all code worse regardless of context.

To suppress changes to measured ordinary behavior, construct a local reference metric from the suspect itself. On independently chosen, task-validated reference prefixes \(\mathcal B\), define the conditional Fisher

\[
F_{\mathcal B}=
\mathbb E_{\substack{b\sim\mathcal B\\y\sim p_\theta(\cdot\mid b)}}
\left[
\nabla_\theta\log p_\theta(y\mid b)
\nabla_\theta\log p_\theta(y\mid b)^\top
\right].
\]

For a small parameter change \(\epsilon v\), this gives the local expansion

\[
\mathbb E_b\operatorname{KL}
\left(p_\theta(\cdot\mid b)\,\Vert\,
p_{\theta+\epsilon v}(\cdot\mid b)\right)
\approx\frac{\epsilon^2}{2}v^\top F_{\mathcal B}v.
\]

If gradients use fixed correct labels instead of samples from the model distribution, call their Gram matrix an empirical gradient metric; do not silently identify it with this exact Fisher.

For one contextual contrast, a screening direction is

\[
v\propto(F_{\mathcal B}+\lambda I)^{-1}g.
\]

For repeated contrasts, construct a positive-semidefinite matrix \(C\) from gradients that consistently affect the same obligation across tasks, then solve approximately

\[
\max_v
\frac{v^\top C v}
     {v^\top F_{\mathcal B}v+\lambda\|v\|^2}.
\]

For example, \(C=\sum_q\bar g_q\bar g_q^\top\), where \(\bar g_q\) averages aligned, sign-consistent contextual contrasts across tasks. Compare this coherent objective with a simple average of individual gradient outer products, which can emphasize unrelated high-variance effects.

The interpretation is: which direction selectively changes the measured context-to-defect coupling while having little local effect on the reference distribution? This supplies a way to rank parameter blocks, modules, or internal directions.

**A concrete experiment.** Apply tiny positive and negative perturbations along a shortlisted direction in temporary replicas. Measure the selected contextual contrast, ordinary task performance, and unrelated contract scores. Look for a specific interaction, then localize it further through activation or path interventions. The original checkpoint remains the target for final witness verification.

Use blockwise parameters, a small chosen basis, Fisher-vector products, damping, and iterative solves. Do not construct a dense parameter Hessian or Fisher for a large coding model. Finite changes must be checked because local KL approximations can fail. Damping introduces parameterization dependence; the answer need not be invariant to a reparameterization of the same network.

**Important limitation.** Low reference sensitivity also describes rare legitimate capabilities and uncovered ordinary behavior. The metric only preserves what the reference prefixes measure. It does not certify a direction as malicious, or a perturbed replica as clean. If all contextual gradients are zero behind a closed gate, preconditioning cannot create the missing signal; first use the gate or route experiments.

Functional-attribution MAD already studies whether trusted examples functionally explain a test output, using parameter-space attribution [r9]. The proposed extension is active discovery: use conditional gradients to choose the next circuit intervention and the next real context variation. It should be compared with that existing functional signal rather than advertised as the invention of functional attribution.

**8. Test whether a represented obligation has lost control over action.**

Coding tasks offer a distinctive source of mechanistic supervision: a property can be present in the specification, encoded in a hidden state, implemented in code, and checked by tests. These stages need not remain coupled.

For a known obligation \(q\), identify candidate representations using controlled task pairs and held-out readout tests. Then estimate causal effects using small interventions and suitable donor states. A linear readout that recovers \(q\) is evidence of decodability, not proof that the model understands or uses it.

The informative pattern would be:

1. The obligation remains decodable under a candidate context.
2. In matched ordinary contexts, intervening on its representation changes implementation or verification.
3. Under the candidate context, that causal influence weakens or reverses.
4. Intervening on a second, context-sensitive component restores the obligation's influence.

This is a hypothesis about a conditional break in information use. It is stronger than merely finding that the model says the right rule and writes wrong code, which ordinary models often do.

One finite interaction can make the test explicit. Let \(I_q\) add a small, validated change in the obligation representation, and \(I_s\) change a suspected selector. Measure

\[
\Delta_{q,s}=
\phi(I_q,I_s)-\phi(I_q,I_0)
-\phi(I_0,I_s)+\phi(I_0,I_0).
\]

Repeat the measurement for unrelated obligations. An indiscriminate improvement in all behavior suggests general damage or competence variation. Selective restoration of the obligation's effect is a more informative causal lead.

**An additional coding-specific opportunity is correlated implementation and verification failure.** Give the model separate roles on controlled fixtures: implement a requirement, review a supplied implementation, and write tests against known mutants. A context that both favors an omitted check and reduces the chance that tests catch that same omission offers two observations of a possible shared conditional policy.

Define separate scores for implementation and missed detection. Compare their context-conditioned gradients using a regularized reference metric:

\[
\kappa_q(c)=
\Delta g_{\mathrm{implement},q}^{\top}
(F_{\mathcal B}+\lambda I)^{-1}
\Delta g_{\mathrm{verify},q}.
\]

This is a proposed ranking signal for shared mechanisms. Gradient alignment alone is insufficient: the roles share language, task knowledge, and ordinary weaknesses. Use different task phrasings, different mutant implementations, and held-out repositories. Patch the candidate mechanism in each role separately and together. Require actual changes in independently measured implementation and mutant-detection outcomes.

This approach has an advantage before trigger discovery: known mutants supply verification labels without waiting for the suspect to produce its own defective patch. It still cannot assume that every payload also corrupts the model's verification policy.

**9. Train a network to predict the next experiment, with supervision available from the start.**

The appropriate initial learning problem is intervention prediction. Its labels are measurable without knowing which checkpoint component is backdoored.

Inputs can include a compact activation sketch, task/obligation descriptors, source-edit descriptors, the proposed intervention site and type, and the intervention magnitude. Targets can include:

- Changes in gate preactivations or router margins.
- Changes in the contract-score vector.
- The source/circuit interaction \(T_{i,j,q}\).
- A fitted threshold \(\beta^*(c)\), with a separate outcome for “no reliable crossing.”
- Changes in ordinary task utility.

Only after externally validated rollouts exist should the network predict their violation probabilities. Do not invent binary backdoor labels for an initially unlabeled dataset.

A practical model could have shared context features and separate heads for internal response, contract response, and uncertainty. Train it on randomized intervention assignments within controlled task blocks. Hold out entire source transformations, task families, and circuit groups when testing generalization. A single train/test split of nearly identical traces would overstate its usefulness.

The model can select experiments with a combination of predicted useful effect, uncertainty, expected information gain, and execution cost. Keep explicit random or coverage-driven exploration. Otherwise it may become very accurate at predicting the benign parts of the model while never visiting the hidden condition.

Raw neuron coordinates are specific to a checkpoint. The strict version trains on the suspect's own interventions. A cross-model version needs an explicit alignment scheme or functional descriptors and a separate evaluation of generalization. It must not quietly use clean ancestors or labeled siblings unavailable in the stated setting.

Deep Causal Transcoding already learns models of internal intervention responses and uses them to elicit latent behavior [r10]. The proposed contribution here is the input-facing objective: use predicted responses and activation thresholds to choose admissible context edits that produce a natural coding-agent witness. A larger predictor is useful only if it improves completed audits after counting its own data-collection and training cost.

**10. Separate within-model gradients from agent-trajectory effects.**

A delayed coding-agent failure can involve several distinct carriers: the original repository text retained in the prompt, its derived KV state, a generated summary, a tool observation, an edited file, or explicit agent memory. A standard decoder does not have an unexplained private state that persists across unrelated calls. Identify the actual carrier in the harness.

For a suspicious source read at time \(t_0\) and a harmful action at \(t_1\), perform replay experiments that distinguish:

- Recognition at \(t_0\): does a source change move a candidate detector before any harmful decision is available?
- Propagation: which retained tokens, summaries, files, or cached representations carry that effect?
- Expression at \(t_1\): does an action opportunity make the carried state influence the contract-critical decision?

In attention-based memory, separate retrieval from content. A key-side intervention can change whether an old state is read; a value-side intervention can change what information arrives when it is read. Test their interaction instead of interpreting attention weight alone as causal evidence.

For context edits, recompute affected descendants. Do not treat stale KV tensors as the natural result of an edited prompt. Internal KV patching is a deliberate counterfactual intervention and must preserve architecture-specific position and shape conventions. Its output is useful for mechanism testing; a final witness needs a consistent, unpatched execution.

The following gradients answer different questions:

| Quantity | What is held fixed | What it helps discover | Important blind spot |
|---|---|---|---|
| Gradient of a matched continuation score | Supplied continuation tokens | Local preference and upstream source sensitivity | Whether free generation chooses those tokens |
| Gradient through a replayed transcript | Recorded actions, tool observations, and token identities | A candidate internal path in that transcript | Alternative discrete trajectories |
| Gradient of a gate preactivation or router margin | Local architecture and selected audit state | How source changes approach an internal boundary | Whether crossing causes an actual failure |
| Parameter gradient of a contextual contrast | Current weights and paired contexts | Modules controlling cue-to-decision coupling | Historical poisoning provenance |
| Policy-gradient estimate of executed violation | Environment rules, while sampling trajectories | Expected effect on actual agent outcomes | Very high variance and sparse failures |

For a sampled agent trajectory \(\tau\) with policy \(\pi_\theta\), a score-function estimator has the form

\[
\nabla_\theta\mathbb E_{\tau\sim\pi_\theta}[V(\tau)]
=\mathbb E_\tau\left[
(V(\tau)-b)
\sum_t\nabla_\theta\log\pi_\theta(a_t\mid H_t)
\right],
\]

under the usual fixed-environment and action-independent-baseline conditions. This does not differentiate through a compiler or test runner. It uses the external outcome as a sampled reward. Include the relevant token-level action probabilities and stochastic decisions when defining the trajectory.

For a dormant backdoor with no positive outcomes, this estimator may provide almost no useful signal. That is why matched mutants and controlled internal interventions are valuable for initial screening. Once natural failures appear, trajectory outcomes can refine the ranking and distinguish local preference shifts from effects that survive agent feedback.

Do not infer a recurrent “malicious attractor” from a large transformer Jacobian eigenvalue alone. Layer transitions, token feedback, retained state, and external tool dynamics are different systems. Any claim of persistence should survive controlled removal and restoration of its proposed carrier.

**11. What the vLLM implementation would actually require.**

Use the original vLLM deployment for normal rollouts and final witness confirmation. Use an instrumented, differentiable runner with the same suspect weights for selected gradient and counterfactual experiments. That runner is an audit replica, not a clean base model.

The official vLLM description of its hidden-state extraction path uses an extract_hidden_states configuration and a KV connector. The documented path saves prompt-token hidden states and recommends a one-token completion for extraction [r11]. This is useful for replaying a recorded decision prefix. It does not by itself provide the gate preactivations, per-head intermediates, MoE counterfactuals, or backward passes needed here. Verify the exact installed version before relying on those details.

For each selected state, retain:

- Checkpoint identity, active adapters, tokenizer, prompt template, precision, and quantization.
- Full message/tool boundaries, repository snapshot, environment state, and exact token IDs.
- Selected decision positions and the independent obligation check.
- For attention experiments: selected queries, corresponding keys/values, mask/position metadata, and downstream adjoints.
- For gated MLP experiments: preactivation and content-branch values at selected sites.
- For MoE experiments: pre-top-k scores, selected routes, and only the alternative expert outputs under test.

Collect sketches and selected states rather than every activation from every layer and token. Perform reductions on the worker when possible; projecting tensors after transferring full activations does not save extraction traffic.

An approximate dequantized or differently fused replica may disagree with deployment near a route boundary. Check token/logit agreement at the audited prefixes, report disagreements, and verify every claimed natural witness in the original serving configuration. Disable or account for speculative execution when it obscures which activations belong to accepted tokens.

Start with deterministic replay where available, then estimate failure rates across controlled generation seeds. A single counterfactual trace is useful for localization; robustness across repeated original-agent runs makes the recovered witness more informative.

**12. The decisive research experiment.**

The first experiment should test whether internal measurements made before the first natural failure improve trigger acquisition. Finding a known circuit after giving the auditor its trigger does not answer this question.

Use a blinded collection of coding-agent organisms. The evaluator knows their construction; the auditor receives only the suspect, task harness, and authorized access. Cover several mechanisms and include ordinary buggy and benign-specialist controls. Keep all close relatives of a planted mechanism on one side of any training/test split.

| Audit arm | Extra information used to select context experiments |
|---|---|
| Behavioral baseline | Contract outcomes, program analysis, and a strong context-search scaffold |
| Passive-internals baseline | Activation statistics and ordinary source gradients |
| Saturation-aware attribution baseline | AtP*-style finite local corrections on available paired contexts |
| Internal-mode baseline | Latent intervention search without activation-boundary guidance |
| Proposed route/gate arm | Route potential, finite gate sweeps, and context-dependent thresholds |
| Proposed coupled arm | The preceding measurements plus source/circuit interactions and a learned scheduler |

Give every arm the same expandable context-proposal process and account for all costs: forward passes, backward passes, internal interventions, expert evaluations, predictor training, replay, and code execution. A method should not get a hidden trigger grammar unavailable to the baseline.

A defensible progression is:

1. Verify that the architecture-specific scores predict actual finite internal effects better than ordinary gradients.
2. Verify that context-dependent thresholds predict which real context edits move the original gate or route toward activation.
3. Verify that this ranking improves first-witness discovery on withheld mechanisms.
4. Verify that the recovered witness's behavior is mediated by the shortlisted circuit.

The primary outcome is natural witness recovery within a total compute budget. Also report time to first witness, context experiments required, independent failure rate, and the rate at which artificial internal failures never become naturally reachable.

For mechanism evidence, require several complementary tests:

- **Natural activation:** the unmodified checkpoint fails under the recovered, task-valid context.
- **Context specificity:** matched changes to the candidate condition alter failure probability while preserving the task.
- **Necessity:** blocking the proposed path reduces the effect beyond matched control interventions.
- **Sufficiency:** an appropriately constrained path intervention transfers the effect to matched contexts.
- **Rescue specificity:** restoring the intended obligation's influence repairs the measured failure without merely making the agent refuse, crash, or stop coding.
- **Replication:** the relation survives held-out task instances and repeated agent execution.

Necessity and sufficiency are intervention-relative and can be complicated by redundant pathways. Report partial mediation rather than forcing a one-neuron story. In real checkpoints, this evidence localizes a conditional failure mechanism; historical intent still requires provenance. If the benchmark already guarantees a planted backdoor, that guarantee can establish provenance separately.

**13. Where I would place the research effort.**

| Direction | Why I would try it | What would make me stop investing |
|---|---|---|
| Gate-threshold sweeps followed by natural-input continuation | Supplies a potentially graded signal while ordinary gradients are suppressed | Threshold changes fail to predict progress toward zero-intervention witnesses |
| Attention route potential plus query/key reachability | Provides a cheap, precise source-to-decision screening calculation | Rankings only find generic task disruption or sources with no natural access |
| Counterfactual expert routing, if the model is MoE | Directly evaluates computation excluded from ordinary traces | All apparent findings are generic unsafe routes or never naturally selected |
| Context/circuit interaction prediction | Compresses source search using experimentally measured relations | A simple bandit or passive features perform equally well after total cost |
| Conditional parameter gradients under a reference metric | Distinguishes changes in cue coupling from broad output changes | Rankings depend mainly on reference coverage, parameter scaling, or output-head artifacts |
| Coupled implementation/verification mechanisms | Adds labels before the model generates a natural defect | Correlation disappears under matched tasks or reflects general weakness |

I would start with the first two on a manageable dense coding model, or the gate and routing variants on a manageable MoE. I would add the learned predictor after demonstrating that the measured responses help choose better context experiments. Its training data would then contain a signal worth learning.

Three boundaries are fundamental. A semantic neighborhood is not guaranteed for an opaque exact-match trigger. A dormant feature may be absent from the activation distribution used to train a passive probe or SAE. And an internally reachable bad behavior need not be reachable by any task-valid natural input. The proposed methods are ways to test and exploit favorable structure, not guarantees that every backdoor leaves an efficiently recoverable precursor.

The strongest paper-level claim would be:

**“For withheld coding-agent backdoors, architectural interventions expose context-dependent activation boundaries that improve natural trigger recovery without a clean ancestor or an initial triggered example, under matched total audit budgets.”**

That claim is narrower than universal backdoor detection, but it would answer the hard part of this research problem.

**Primary-source ledger and prior-art boundaries**

| Reference | What was checked | Contribution used here |
|---|---|---|
| [r1] AtP* | Primary abstract and full-text sections on attention and MLP saturation | Gradient-based attribution can miss effects; local nonlinear recomputation provides a baseline |
| [r2] GLU Variants Improve Transformer | Primary abstract and indexed primary paper | The gated feed-forward architecture; the derivatives here are elementary derivations |
| [r3] ABS | Author-hosted full paper, including stimulation and inversion sections | Stimulation-guided trigger reconstruction predates this proposal |
| [r4] BadMoE | Primary abstract and full-text mechanism sections | Dormant expert backdoors motivate inspecting unexecuted routes |
| [r5] BadSwitch | Primary abstract | Task-affine routes can also be involved, so rarity is not a complete prior |
| [r6] When Are Experts Misrouted? | Primary abstract | Equal-compute counterfactual routing is established analysis |
| [r7] Sparse Models, Sparse Safety | Primary abstract and official repository | Unsafe artificial routes are a confound, not proof of the sought backdoor |
| [r8] RouterInterp | Primary proceedings abstract | Expert specialization can be superposed rather than one broad topic per expert |
| [r9] Mechanistic Anomaly Detection via Functional Attribution | Primary abstract; full-text methods checked in the earlier memo | Functional coupling to a trusted reference is established prior work |
| [r10] Deep Causal Transcoding | Author's full article, including method and limitations | Learning internal response models and eliciting latent modes already exist |
| [r11] vLLM hidden-state extraction | Official implementation article | The documented extraction path and its prompt-token limitation |

[r1]: https://arxiv.org/html/2403.00745v1
[r2]: https://arxiv.org/abs/2002.05202
[r3]: https://tao.aisec.world/assets/pdf/CCS19.pdf
[r4]: https://arxiv.org/html/2504.18598v2
[r5]: https://arxiv.org/abs/2510.13462
[r6]: https://arxiv.org/abs/2605.07260
[r7]: https://arxiv.org/abs/2602.08621
[r8]: https://proceedings.mlr.press/v306/lasy26a.html
[r9]: https://arxiv.org/abs/2604.18970
[r10]: https://turntrout.com/deep-causal-transcoding
[r11]: https://vllm.ai/blog/2026-03-30-extract-hidden-states
