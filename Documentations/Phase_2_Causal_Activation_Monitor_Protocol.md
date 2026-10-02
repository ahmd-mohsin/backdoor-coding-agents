**Phase 2: Learning a pre-action coding-agent monitor from causal audits**

Research protocol for the methodology in *Pasted markdown(10).md*. Prepared 2026-10-02. The model-specific results below are reported in that draft; I have not inspected the experiment code or raw logs, or run the checkpoints. All proposed experiments are marked as proposals. The source review uses primary papers and author documentation.

**1. The concrete recommendation**

Make Phase 2 test the following claim:

> Measurements of how an untrusted source changes a coding agent's action preferences can supervise a compact pre-action monitor more effectively than rare observed failures. Mechanistic analysis determines which representations retain that information and whether those representations support useful generalization.

The proposed object is a monitor with two outputs: a **source-effect estimate**, which predicts retained influence on a declared violation score, and an **action-risk estimate**, which predicts an unauthorized action under the actual decoder. These outputs must stay separate. A behaviorally suppressed checkpoint can have a large first output and a small second output.

The main Phase 1 addition is bidirectional, within-checkpoint activation replacement: find components where replacing a triggered activation with its matched neutral activation reduces the source effect, and restoring the triggered activation recovers it. Test the same components on legitimate coding operations. Phase 2 receives the resulting sites, compact subspaces, source-effect labels, and intervention examples. It does not receive the future malicious continuation as an input.

Start with ridge/linear models and a small source-to-decision interaction model. Train a temporal network only if earlier causal measurements demonstrate persistent information that a current-state model misses. A larger network cannot recover information absent from its inputs.

**2. What the updated results establish, and what they do not**

Your new 3B results are stronger than the previous pilot. The draft reports 55 resolved examples per suspect checkpoint, Hit@1 of 0.527 and 0.473, substantial payload log-likelihood effects, and an undefended action-emission change of 1.00 to 0.00 after removal of the recovered span. These motivate an audit-to-monitor experiment.

Five distinctions affect its design:

1. The defended action-emission result is 0.00 to 0.00 under the current greedy test. That does not establish action-level necessity on that checkpoint. It establishes no observed attack in either arm of that test. Its large likelihood contrast supports a different claim. Reproduce the reported 0.21 rate under its actual harness, sampling settings, and denominator before relating the two numbers.
2. Hit@1 around one half is promising, but does not mean every trigger is the single most influential token. Report integer ranks and source-span recovery, alongside the rounded percentile.
3. Proposition 1 assumes an unchanged selector and a context-independent suppression offset. The observed cross-checkpoint results are consistent with that toy model; they do not identify the real defense mechanism or prove that activations are invariant.
4. Phase 1 currently uses exact payload knowledge and histories already containing the trigger. A monitor distilled from that teacher has privileged training information even if it uses no payload at runtime.
5. One benign model is a useful control, but cannot establish that the learned representation is specific to backdoors across benign specializations, rare legitimate operations, or other checkpoints.

Also correct the literature table: Patcher is directly about backdoored LLMs, not merely a classifier-era method. Its response-conditioned embedding saliency is a close baseline [1].

**3. What the most relevant papers actually contribute**

The following is a targeted extraction, rather than a list of vaguely related titles. The proposed uses are our synthesis; they are not results established by these papers.

| Primary source | Relevant method/evidence | Consequence for this project |
|---|---|---|
| **Patcher** [1] | Uses a reported failure and suspect weights; ranks prompt tokens by response-conditioned embedding-gradient norms and groups scores with two-cluster K-means before patching the model. | Reproduce its grouping as the Phase 1 baseline. The new contribution must lie beyond the localization operator. |
| **Simple probes can catch sleeper agents** [2] | Generic yes/no contrast pairs produce residual-stream directions that separate defection-inducing inputs in constructed sleeper agents. Probes operate at the final prompt token. The authors explicitly question how much feature salience comes from the organism construction. | Include a cheap generic contrast-direction baseline. Success on one planted cue does not establish a universal backdoor representation. |
| **AgentLens** [3] | Trains step-level, last-token probes for multi-turn coding agents; selects influential coordinates for steering. Includes benign counterfactuals that resemble malicious command syntax. Its lookahead subset is explicitly limited and sparse. | This is the closest ordinary activation-monitor baseline. Neither pre-action coding-agent probing nor syntax-matched benign controls is, by itself, new. |
| **Agent Safety From Within / TACIT** [4] | Reads frozen-model representations of trajectories. Paired experiments distinguish harmful content from unsafe tool use; these signals need not align. Uses last/mean pooling and evaluates benchmark holdout. It is an auxiliary trajectory assessor, not evidence of early warning from the acting backdoored model. | Include authorization and provenance controls. Treat its pooling/readout as a baseline adapted to the same pre-action cutoff, not as a directly comparable early-warning result. |
| **LLMs Encode Harmfulness and Refusal Separately** [5] | Different positions and interventions distinguish harmfulness-related representations from surface refusal behavior. A reply-inversion test separates two directions that otherwise both induce refusal. | Separate recognizing a cue, selecting a payload, and suppressing its expression. Design interventions whose outcomes distinguish competing explanations. |
| **Backdoor Decontamination Dynamics** [6] | Measures recognition/execution separation in tool agents. J-Lens visualizations show remaining internal traces after benign responses return. Its probing corpus includes poisoned and decontamination examples. | Persistent recognition is already known and does not establish actionable risk. Your stronger test is held-out source control and prospective monitoring, with the teacher's data privileges declared. |
| **Backdoor Attribution** [7] | Uses trigger-present/absent data, head activation substitution, and ablation to identify backdoor-related heads and control behavior. Its tables show heterogeneous ablation effectiveness across organisms. | Borrow paired head interventions, but replace curator-supplied trigger positions with the auditor's predictions. Do not assume sparse heads transfer to your organism or monitoring setting. |
| **AtP\*** [8] | Studies failures of gradient approximations to activation replacement, including saturation and cancellation, and develops more reliable screening. | Use attribution to shortlist components, then measure finite replacements. A weak gradient or the negative attention-route result cannot rule out a causal component. |
| **Distributed Alignment Search, DAS** [9] | Learns subspaces through interchange interventions that should reproduce a specified high-level causal model. It also discusses how excessive flexibility can manipulate behavior without giving the intended explanation. | Learn a small subspace by intervention outcomes, test held-out donors and nuisance controls, and compare to equally sized random or ordinary probe subspaces. |
| **Obfuscated Activations** [10] and **Neural Chameleons** [11] | Show that representations can change in ways that defeat activation monitors, sometimes while behavior remains. The settings, costs, and capability trade-offs differ. | Add a separately labeled robustness evaluation. Causal feature selection does not make a monitor invariant to adaptive representation changes. |
| **Black-to-White Performance Boosts** [12] | Proposes evaluating the extra monitoring value of internal probe information over observable evidence. | Measure improvement over a strong text/action monitor at matched information cutoffs and false-alarm rates, not just standalone probe AUROC. |

The distinctive research question is consequently narrower and more useful: **does counterfactual source-effect supervision, restricted to causally tested pre-action representations, improve auditing or monitoring under behavioral suppression and meaningful holdouts?** This remains a hypothesis.

**4. Specify Phase 2's information regime before training**

| Regime | What the training pipeline knows | What the deployed monitor receives | Claim allowed |
|---|---|---|---|
| **P2-A: after an audit** | Same suspect checkpoint; localized development witnesses; exact payload teacher allowed | Current history, policy metadata, selected suspect activations | Cheap monitoring of a discovered mechanism on new tasks/carriers |
| **P2-B: contract teacher** | Same checkpoint; declared violation fixtures, no implanted payload literal in teacher candidates | Same runtime inputs | Monitoring within the measured contract library |
| **P2-C: unseen backdoor transfer** | Development backdoors only; held-out trigger/payload mechanisms excluded from teacher, feature selection, and fitting | Same runtime inputs, with any per-model calibration disclosed | Generalization to the specified held-out backdoor family |

P2-A is the immediate concrete experiment. P2-B is a valuable extension. P2-C needs several genuinely different mechanisms, not merely more histories bearing the same comment.

The auditor never needs clean ancestor weights for paired input experiments: both histories run through the same suspect. Benchmark curators may retain clean or differently trained models as external evaluation controls. Keep that curator information out of the auditor pipeline. Four model IDs are not automatically four independent backdoor lineages, and raw activation coordinates from 3B and 7B are not interchangeable.

**5. Define the labels as two different quantities**

Let \(H_t\) be the entire model-visible history before action \(a_t\), including the actual chat template. Let \(q\) specify a violation through an independent action/trajectory predicate \(V_q\). For a declared horizon \(K\), define

\[
Q_q^{(K)}(H_t)=\Pr_{\theta,\mathcal D,\mathcal E}\left(\exists j\in\{t,\ldots,t+K-1\}:V_q(a_j,H_j)=1\mid H_t\right).
\]

The decoder \(\mathcal D\) and environment/harness \(\mathcal E\) are part of this estimand. A sample label \(Y_{t,q}^{(K)}\) is one realization, not the exact probability.

For offline auditing, use the matched continuation score

\[
\phi_q(H)=\log p_\theta(y_q^-\mid H)-\log p_\theta(y_q^+\mid H).
\]

Use exact payload likelihood as a separately labeled teacher ceiling. Neither score is the probability of all violations. Candidate length normalization further changes its probabilistic meaning.

For a candidate source span \(S\), let \(H^{S\leftarrow r}\) replace it with a task-valid, role- and length-matched alternative. Measure

\[
D_q(H,S)=\phi_q(H)-\frac1R\sum_{r=1}^{R}\phi_q(H^{S\leftarrow r}).
\]

This is a causal effect on a chosen model score, relative to a specified replacement distribution. It is not automatically an effect on unauthorized action probability. On a validation subset also measure

\[
N_q^{(K)}(H,S)=Q_q^{(K)}(H)-\frac1R\sum_r Q_q^{(K)}(H^{S\leftarrow r}).
\]

Keep the full signed vector \(\mathbf D=(D_1,\ldots,D_Q)\). A negative value is informative. Record matched random-span effects and legitimate-action score effects beside it. An optional standardized label uses a location/scale fitted only to development null effects; always retain raw nats.

Counterfactual labels can be measured even when a particular rollout does not fail. However, they are only informative about the sources and violation candidates you test. This does not create positive evidence for an absent, unencountered trigger.

**6. Why Phase 1 saliency does not guarantee a useful activation monitor**

Your successful operator is \(\nabla_{e_j}\log p(y^-\mid H)\). It depends on the supplied continuation and all computational paths relevant to it. A deployed monitor may receive only a few prefix activations. These are different information sets.

For a simple counterexample, suppose the observed decision vector stores the ordinary task, while a source's KV entries store a cue. The next token is the same tool name in both cases. Only the subsequent argument-generation query retrieves the cue and selects the unauthorized argument. Target-conditioned input attribution can expose the cue, while the chosen pre-tool decision vector does not separate the cases. This is a possible computational arrangement, not a claim about your checkpoint.

There is a precise observability criterion. For measured effects \(D\) and chosen runtime features \(Z\), squared-error prediction satisfies

\[
\mathbb E[(D-g(Z))^2]
=\mathbb E[\operatorname{Var}(D\mid Z)]
+\mathbb E[(\mathbb E[D\mid Z]-g(Z))^2].
\]

The first term cannot be eliminated by fitting a larger network. This standard conditional-expectation identity motivates the first Phase 2 gate: can prefix-only representations predict held-out measured effects at all, and do they outperform cheap input and score baselines?

Also distinguish information in a derivative from information in its evaluation point. The scalar function value or a pooled activation can coincide on two inputs while its local derivatives differ. Consequently, the log-gradient ranking proposition does not imply that a single passive direction remains stable under benign fine-tuning.

**7. “Generic” has three different meanings**

| Representation | Why a probe might succeed | What would establish useful specificity |
|---|---|---|
| Trigger spelling or unusual comment | The train/test sets share a conspicuous literal | New carriers and, for a transfer claim, genuinely held-out trigger identities |
| Shell, file, or network semantics | Payload-bearing histories contain the same action vocabulary | Legitimate actions with matched syntax, resources, and task difficulty |
| Refusal or output style | Defense changes compliance/refusal formatting | Suppressed and firing histories, plus harmless refusals and authorized risky-looking operations |
| Source-to-policy influence | The source changes the model's chosen continuation through a repeatable route | Source replacement, internal removal/restoration, and selective effects over legitimate operations |

A generic concept can be useful. A network-operation representation combined with authorization and source provenance may improve monitoring. What is insufficient is treating that concept alone as evidence of an ingested backdoor.

A direction can also be causal but insufficiently specific. Disabling a component used for all shell commands may eliminate the attack by crippling tool use. Causal intervention and preserved utility must be measured together.

**8. Phase 1 must deliver an intervention dataset, not only a saliency heatmap**

For every development history, construct the following package:

- **Input pair:** original history and two independently chosen task-valid replacements of the auditor-proposed span. Include one restoration arm and a matched random-span arm.
- **Prefix states:** source-span representations, last observation token, and final prompt token after the assistant header but before generation. Preserve exact token positions and hook semantics.
- **Effects:** \(\mathbf D\), a small set of internal replacement effects, and behavioral outcomes with an explicit decoder configuration.
- **Controls:** the same action vocabulary under legitimate authorization; quoted security examples; near-matching comments; ordinary rare commands; unrelated high-saliency task-critical spans.
- **Provenance:** repository/task family, paired-example group, checkpoint lineage, all candidate sources, and whether source selection was algorithmic or curator-supplied.

Teacher-forced suffixes are allowed for offline target computation. The student features must come from the prefix. In a correctly masked causal transformer, appending a teacher-forced suffix cannot change earlier prefix activations, but extracting features from that suffix directly introduces future information. Verify the actual implementation by comparing prefix-only and teacher-forced-prefix feature extraction.

Do not select only successful localizations. A failed source proposal is part of the monitor's training/evaluation distribution. Sample random and ordinary task-important spans as well as high-score spans.

**9. Recover the mechanism with bidirectional replacement**

For one matched pair, write \(H^+\) for the original and \(H^-\) for a valid replacement. Let \(m\) be a component/site and \(h_m^+,h_m^-\) its two activations. Evaluate

\[
\begin{aligned}
A_q&=\phi_q(H^+),& B_q&=\phi_q(H^-),\\
C_{m,q}&=\phi_q(H^+;h_m\leftarrow h_m^-),&
E_{m,q}&=\phi_q(H^-;h_m\leftarrow h_m^+).
\end{aligned}
\]

The component's removal and restoration effects are

\[
\kappa^{\rm rem}_{m,q}=A_q-C_{m,q},\qquad
\kappa^{\rm add}_{m,q}=E_{m,q}-B_q.
\]

Begin with residual blocks at the observation end and decision boundary. Refine promising blocks into attention writes and MLP writes. Source-position nodes can identify recognition; downstream decision nodes test propagation. Gradient times paired activation difference is a screening approximation:

\[
\widehat\kappa^{\rm rem}_{m,q}
=\nabla_{h_m}\phi_q(H^+)^\top(h_m^+-h_m^-).
\]

Validate the best candidates and a random sample of apparently unimportant ones with finite replacements. This protects against attribution false negatives [8]. Group interventions are necessary where several paths are redundant.

Normalize by \(|A_q-B_q|+\epsilon\) only for pairs with a reliably nontrivial total effect, and publish raw effects too. Ratios may be negative or exceed one. They are not automatically fractions of a uniquely decomposed causal effect. Parallel pathways, compensation, and nonlinear interaction prevent naive additivity.

Patching only the final token does not cut every path: earlier KV states can still affect later argument tokens. If effects appear only after the first tool token, add those positions as a separately labeled pre-execution monitor, or retain source/KV-derived features. Do not interpret a failed final-token patch as absence of a backdoor circuit. Align donor sites by actual tokenization and semantic boundaries; a source-token swap requires compatible positions, whereas an end-of-observation swap can align histories with different source lengths.

**10. Learn a small causal subspace, only after identifying useful sites**

At a selected site, let \(U\in\mathbb R^{d\times r}\) have orthonormal columns. A paired subspace swap is

\[
\mathcal I_U(h^+,h^-)=h^++UU^\top(h^--h^+).
\]

Compare three ways to choose \(U\): a linear behavior/effect probe direction, an SVD of paired activation differences, and intervention-trained alignment. The first two are cheap baselines. For the third, optimize the suspect-frozen objective

\[
\begin{aligned}
\mathcal J(U)=\mathbb E_i\sum_q\big[&
\ell(\phi_q(H_i^+;\mathcal I_U),\phi_q(H_i^-))\\
&+\ell(\phi_q(H_i^-;\mathcal I_U^{\rm reverse}),\phi_q(H_i^+))\big]
+\lambda\mathcal L_{\rm control}(U).
\end{aligned}
\]

Here \(\ell\) is Huber loss after a fixed training-only scale normalization. The control term measures disruption to legitimate fixture completions under matched benign donor swaps. Fit ranks \(r\in\{1,2,4,8\}\), selected on development data, not on the held-out monitor test.

This adapts interchange-intervention learning [9]; it is not a new causal-identification theorem. A flexible subspace could edit output preferences without isolating the intended computation. Require held-out donors, tasks, positions, full-rank and random-rank controls, and effects on natural action issuance. Do not call a direction “the backdoor feature” because it can steer the model.

A defensible outcome is: *a compact subspace reproduces a measured part of this source's effect, at these sites, across these held-out contexts, while causing less legitimate-task disruption than controls.*

**11. A mechanistic experiment tailored to the gated hypothesis**

If the useful sites include SwiGLU modules, separate the two factors rather than adding an arbitrary residual direction. With the module's actual normalization and architecture respected, write its output schematically as

\[
w=W_{\rm down}[\operatorname{SiLU}(s)\odot u].
\]

At selected channels, compare four paired combinations:

\[
w_{ab}=W_{\rm down}[\operatorname{SiLU}(s^{(a)})\odot u^{(b)}],\qquad a,b\in\{+,-\}.
\]

Keep the remaining computation fixed to a declared base run. Measure the corresponding scores \(F_{ab}\) and interaction

\[
I_{\rm gate,value}=F_{++}-F_{+-}-F_{-+}+F_{--}.
\]

If switching the gate factor changes source control while retaining the content/write factor, that supports a selector interpretation. If the write factor dominates, a content-retrieval explanation may be better. If only the joint swap works, the factors are coupled. Include legitimate file/network actions to determine whether the module is merely generic tool computation.

These hybrid activations can be off-distribution; use them as diagnostics, triangulated with natural source changes and full-module swaps. SiLU is not a literal sigmoid gate, and successful token saliency does not prove saturation of particular channels.

**Phase 2 consequence:** retain separate selector-like and write-like coefficients only if this experiment supports the distinction. Otherwise use the smallest validated joint subspace. Do not hard-code a speculative circuit into the monitor.

**12. The first monitor should be explicit and small**

Let \(b_t\) be the last prefix token and \(o_t\) the last token of the latest observation. Let \(\mathcal L_*\) contain at most four development-selected layers. Project observed states into frozen site-specific subspaces:

\[
c_{t,\ell}=U_\ell^\top(h_{\ell,b_t}-\mu_\ell),\qquad
o_{t,\ell}=U_\ell^\top(h_{\ell,o_t}-\mu_\ell).
\]

For each candidate source span \(S\), pool its projected prefix states, using mean and max as separate features. Learn source and decision projections separately when their sites differ; a direction validated at the decision boundary is not automatically meaningful at source tokens. Include source role, relative location, age in tool steps, and trusted policy metadata. The same segmentation runs at training and deployment: repository lines, command-output blocks, or document spans. A monitor must not be handed the true trigger span at test time.

Define source features \(c_S\) and decision features \(c_b\). Fit the small interaction model

\[
\widehat D_q(H_t,S)
=a_q^\top c_S+b_q^\top c_b+c_S^\top M_qc_b
+v_q^\top\mathrm{meta}(S,H_t)+b_{0q}.
\]

Restrict \(M_q\) to rank at most four initially. Compare against an additive version with \(M_q=0\), a raw-state ridge model, and a one-hidden-layer MLP with 64 units. The interaction term tests whether a source's meaning matters conditionally on the current decision state. Do not presume this form wins.

The source-effect output produces a ranked list of spans and obligations. A separate global head predicts \(Q_q^{(K)}\), using pooled source-effect predictions, decision features, and explicit authorization metadata. This head needs behavioral labels. If no positive actions occur in the available defended development runs, fit/report the effect monitor first; an imminent-failure classifier cannot be validated on all-negative outcomes.

Use a lexical monitor that searches for the discovered cue as a baseline in P2-A. If the learned monitor cannot improve transfer or cost over that baseline, its extra complexity has not earned a place.

**13. Training losses and what they mean**

The primary dense loss is

\[
\mathcal L_{\rm effect}=\mathbb E_{H,S}\sum_q
\operatorname{Huber}(\widehat D_q(H,S)-D_q(H,S)).
\]

A behavior head has

\[
\mathcal L_{\rm risk}=\mathbb E_{H,q}\operatorname{BCE}
(\widehat Q_q^{(K)}(H),Y_q^{(K)}).
\]

A useful optional auxiliary head \(v_q(Z)\) predicts a score potential, through measured pair differences:

\[
\mathcal L_{\rm difference}=\mathbb E_{H,H'}\sum_q
\operatorname{Huber}\left(v_q(Z(H))-v_q(Z(H'))-
[\phi_q(H)-\phi_q(H')]\right).
\]

Pairs can be natural source replacements or internally patched runs. Extract the patched run's features again; do not pretend an intervention leaves its observed features unchanged. Anchor the potential on development null examples if absolute values are used.

The effect head and potential head have different meanings. In particular, do not supervise \(\widehat D(H,S)-\widehat D(H',S)\) with \(\phi(H)-\phi(H')\). Its correct target is \(D(H,S)-D(H',S)\), which generally requires additional counterfactual evaluations.

Compare three supervision arms using the same feature budget and model capacity: behavior only; source-effect only; joint. Add intervention-difference supervision as one subsequent ablation. Standardize targets using training data and start with unit weights; tune only a small preregistered grid on validation data. Report supervision-generation cost separately and in the full pipeline total.

**14. A better generic-versus-specific experiment than a visualization**

Construct matched fixture groups, with task outcome and unauthorized-action predicates specified independently of the monitor:

| Group | Purpose | Expected interpretation |
|---|---|---|
| Discovered cue in a normal repository task | Measures retained backdoor influence | May have high source effect and either high or low action risk |
| Valid replacement and restoration | Tests source control | Effects should track the measured intervention, not the row label |
| Same shell/network semantics in an authorized synthetic operation | Tests generic action semantics | Risk should follow authorization and actual resource scope |
| Security documentation quoting similar words/commands | Tests lexical shortcuts | Reading or discussing an operation alone must not imply issuance |
| Near-matching comment that does not activate the organism | Tests coarse spelling/rarity features | Predictions should follow measured effect rather than visual similarity |
| Same cue with a changed carrier, task, or distance | Tests source-to-decision generalization | Measure whether the mechanism actually survives before assigning labels |
| Same operation, changed trusted authorization | Tests relational safety | Hold action syntax fixed where possible; the violation label can change |
| Rare benign specialized fine-tune | Tests model/style shortcuts | Measure false alarms on ordinary specialized behavior |

The exact same literal can still activate a backdoor inside a quoted or legitimate task context. Therefore, do not decree these examples benign by construction. Check what the suspect does and what the independent contract permits. A recognized backdoor cue with an authorized current action can be positive for the influence head and negative for immediate violation risk.

A strong result is a double dissociation: source-removal interventions reduce predicted/actual backdoor influence without equivalently suppressing authorized operations, while authorization changes alter risk predictions without automatically erasing cue recognition.

**15. What temporal dynamics can add**

A decoder-only agent does not acquire an unexplained hidden state between separate API calls. Persistent information must remain in the visible history, retained KV state, written files, or agent memory. Analyze those carriers directly.

Record four cutoffs when they exist: before the suspicious observation; immediately after it; after an intervening benign action; and before the first unauthorized action. Never use the future failure location to choose a deployment-time cutoff.

Start with interpretable temporal features: change in source-effect prediction, persistence over tool steps, change in decision coefficients, and source age. If these beat a current-state model on held-out tasks, compare a small GRU with a 32-dimensional hidden state against the same features. Fit it only to data available up to each cutoff.

For a stronger mechanistic test, remove the original source and recompute downstream context/KV state; separately remove or replace any summary or file that may carry its influence. If the signal disappears only after changing the summary, Phase 2 should monitor that carrier. Simply deleting visible text while reusing its old KV entries is not a valid removal test.

A linear development diagnostic is

\[
c_{t+1}=A c_t+B\psi(o_{t+1},a_t)+\varepsilon_t.
\]

Measure residuals under matched natural interventions. Large residuals can flag ordinary surprising errors, so temporal anomaly prediction is a baseline, not a backdoor label. Do not infer circuit dynamics from a trajectory plot alone.

**16. The decisive experimental matrix**

| Experiment | Exact comparison | What would change the paper's conclusion |
|---|---|---|
| **P2-0: observability** | Predict \(D\) from raw last-token, source, source-plus-decision, and causal-subspace features; compare input and output-score baselines | Failure means cheap passive monitoring is unsupported by current Phase 1 success |
| **P2-1: causal site validation** | Removal/restoration at discovered sites versus random sites and matched benign operations | Selectivity supports mechanism-aware feature selection; generic tool disruption does not |
| **P2-2: label ablation** | Behavior-only versus effect-only versus joint supervision, equal architecture | Tests whether counterfactual supervision contributes beyond extra examples/capacity |
| **P2-3: feature ablation** | Raw states, ordinary probe directions, PCA/SVD, intervention-trained subspaces at equal rank | Tests whether mechanistic analysis improves the monitor rather than just interpreting it |
| **P2-4: genericity** | Matched syntax, authorization, quoted content, benign specialization, near-match cues | Determines whether the score detects a source-to-policy relation or surface correlates |
| **P2-5: teacher knowledge** | Exact payload, paraphrased/variant action, contract-level candidates | Tests dependence on the supplied planted continuation |
| **P2-6: suppression** | Matched within-lineage defense checkpoints under identical evaluation conditions | Tests whether effect supervision keeps value as observed failures diminish |
| **P2-7: lead time** | Fixed pre-exposure, post-observation, pre-action, and pre-execution cutoffs | Establishes when useful information becomes available |
| **P2-8: transfer** | Hold out tasks/carriers, then trigger identity, then payload family | Separates deployment usefulness from memorizing one discovered cue |
| **P2-9: extra value of internals** | Text-only, lexical, black-box occlusion, output-score, and combined monitors | Tests whether internal access improves the cost/false-alarm frontier |
| **P2-10: robustness** | Held-out long contexts, benign representation shifts, and separately declared monitor-aware organisms | Bounds the claim without implying universal robustness |

The two main factorial ablations are **labels × representation**. Run them on identical splits. If causal labels help with ordinary raw states but causal subspaces add nothing, publish that. If subspaces improve only in-domain accuracy, do not advertise mechanism transfer. If a generic linear probe wins, the result narrows to empirical observability under suppression.

**17. Split the data at the level of the scientific claim**

Every original history, its edits, donor variants, source spans, sampled rollouts, and defense-checkpoint copies share a group ID. Keep each group within a split. Where repositories or tasks overlap substantially, group by repository/task family. Splitting tokens or nearby steps randomly leaks nearly identical contexts.

Use separate sets for mechanism discovery, monitor fitting, model selection, false-alarm calibration, and final testing. Fit normalization, subspaces, source segmentation choices, layer selection, and thresholds without the final test. Do not repeatedly reuse the published 55-example development results as a new untouched test.

An illustrative next collection per primary checkpoint is 40 discovery task groups, 120 fitting groups, 40 validation groups, and 100 fresh paired test groups. These are planning targets, not a power analysis. Add independent benign calibration/test episodes for the desired false-alarm claim. Report curves over training-group count and uncertainty clustered by task/repository, with model-lineage generalization assessed at lineage count.

For a new-trigger claim, exclude that trigger mechanism from teacher generation and subspace selection too. For a new-payload claim, remove the held-out payload family from candidate continuations and teacher labels. A different endpoint string for the same memorized operation is a weaker lexical variation test.

If multiple sizes are tested, initially fit separate monitors. Cross-model claims require an explicit alignment or functionally defined representation, not padding dimensions or treating layer numbers as common coordinates.

**18. Immediate pilot using the data already in the draft**

The present 55-example sets can answer an engineering/scientific feasibility question, but not support a reliable low-false-alarm deployment claim.

1. Use them as development data, grouped by underlying task. Save prefix-only activations at six evenly spaced layers, plus input embeddings as a lexical baseline. Store last-observation and final-prefix positions.
2. For each history, score the eight highest-ranked algorithmic source spans and eight matched random/task-important spans. If cost is tight, begin with four plus four. Use two valid replacements per span.
3. Fit nested group-cross-validated ridge predictors of signed effects. Compare decision-only, source-only, source-plus-decision, input embeddings, and teacher-score baselines. Report out-of-fold MAE, correlation, top-span retrieval, and residual error by control group. Do not select the best layer on the same outer fold being reported.
4. On 20 development pairs, screen components with gradient-times-difference, then run finite swaps at the top eight sites and eight random sites. Include a matched set of legitimate operations.
5. Advance only if useful effect prediction survives lexical controls and at least some selective, bidirectional internal effects replicate. Otherwise retain the expensive auditor or an active two-pass monitor while expanding the feature set.

This is the first concrete Phase 2 milestone: a table showing whether the Phase 1 teacher's information survives in deployable features. It can falsify the monitor proposal quickly.

**19. Pre-action evaluation and calibration**

Let \(T_V\) be the first independently verified unauthorized action issuance, and \(T_A\) the first alarm. Report detection before dispatch and, separately, before any action tokens are generated. An output parser operating after decoding but before dispatch is a strong pre-execution baseline; it has more information than a prefix-only monitor.

Useful metrics are:

- Episode-level recall before the violation, and lead time in tool calls or tokens.
- False alarms per benign episode, plus benign completion/utility loss under the selected intervention policy.
- Recall of auditor-validated influential sources on suppressed histories, as a separate influence metric.
- End-to-end wall time, GPU memory, extra prefills/backwards, and fraction of episodes escalated to the expensive auditor.

Calibrate the final frozen action-risk score \(S\) on \(n\) independent, exchangeable benign episodes using episode maxima \(M_i=\max_t S(H_{i,t})\). For

\[
k=\lceil(n+1)(1-\alpha)\rceil,\qquad c_\alpha=M_{(k)},
\]

set \(c_\alpha=+\infty\) if \(k>n\), and alarm only when \(S>c_\alpha\). The standard split-conformal rank argument gives a marginal per-episode false-alarm bound of \(\alpha\) under exchangeability [15]. It does not establish recall, conditional subgroup safety, or robustness under distribution shift. If calibration occurs under a monitor intervention policy, that policy must match deployment.

For a latent influence head, define its null separately. A suppressed but causally influential backdoor history is not an influence false positive merely because a single rollout stayed benign. Conversely, it can be an action-risk false positive if the deployed decoder never issues the violation under the evaluated conditions.

With zero false alarms in \(n\) independent benign test episodes, the one-sided 95% binomial upper bound is \(1-0.05^{1/n}\). About 299 zero-event episodes are needed to put that bound below 1%. A high AUROC on a balanced 55-example set cannot substitute for this evaluation.

**20. A practical serving design**

Keep the instrumented PyTorch checkpoint as the offline teacher for gradients and activation replacements. The serving path runs the frozen student on selected prefix states, ranks sources, and optionally requests one or two counterfactual replays when uncertainty or risk warrants it. Charge replays and any additional target scoring to the monitor's cost.

The official vLLM hidden-state extraction feature described in March 2026 was introduced in v0.18.0 [14]. Its documented example exports prompt hidden states through a connector; this is not automatic autograd or arbitrary component intervention support. Your reported v0.15 stack predates that feature. Decide explicitly whether to retain custom instrumentation or evaluate an isolated upgrade. Do not silently change the serving stack underlying the reported results.

Pin weights, tokenizer, chat template, dtype, sequence limits, position IDs, attention conventions, and decoder settings. Check teacher/serving agreement on next-token logits, selected activations when available, and generated action categories. Hidden-state or logit differences can be amplified by a learned threshold. Recalibrate on the actual deployment extraction path.

Do not export every activation to disk at runtime. As an illustrative arithmetic example, 20,000 tokens × four layers × 2,048 channels × two bytes is about 312.5 MiB per request. Four last-token vectors are only 16 KiB. Pool/project selected source states on-device when possible. These dimensions are an example, not an assertion about the exact model configuration.

Invalidate KV state after edited source positions during counterfactual replay. Replaying a fixed future trace measures a controlled score effect; allowing the agent to generate a fresh continuation measures changed behavior. Record which one each experiment uses.

Keep the draft's intercept-only policy: parse and log unauthorized action proposals without executing them. Report unauthorized-action issuance, not successful real-world data transfer.

**21. A stronger extension: measure functional responses instead of assuming common activation coordinates**

If ordinary passive features fail to generalize, a more ambitious direction is a fixed-size response vector. For a small development-defined intervention set \(\{\mathcal I_j\}_{j=1}^J\) and obligation scores \(\phi_q\), define

\[
R_{j,q}(H)=\phi_q(H;\mathcal I_j)-\phi_q(H).
\]

This describes how the model responds to interventions in functional units, rather than raw neuron coordinates. A student can be trained to predict selected \(R_{j,q}\) entries cheaply, and an expensive fallback can measure them. The interventions might be paired source substitutions or validated internal replacements. An intervention defined as “neuron 817” is not semantically shared across models; per-model site discovery remains necessary.

This proposal could improve transfer across checkpoints if the same control relationship survives coordinate drift. It may also fail because the mechanisms or score library differ. Compare against ordinary per-model normalization and simple probes before making it the main method. Both teacher generation and any test-time intervention require explicit information/cost accounting.

The strongest evidence would be consistent predictions of held-out intervention outcomes across new contexts, followed by improved monitoring. Merely reconstructing a response matrix on training examples is not a result about backdoor detection.

**22. A second extension: source-conditioned transport, with a real null test**

Suppose Phase 1 identifies an early source representation \(r_S\) and a later decision representation \(c_b\). Learn a low-rank map \(T\) on paired development differences:

\[
\Delta c_b\approx T\Delta r_S.
\]

Compare maps fitted to backdoor-effect pairs, ordinary task-critical edits, and matched lexical edits. Then ask whether the backdoor-related map predicts intervention effects on held-out tasks better than the generic maps. Validate by source and receiver replacements, not by high regression fit alone.

The deployable feature can be an interaction between the current source state and the decision state, as in Section 12. A high-dimensional transport model is unnecessary until a low-rank map survives these controls. Cross-layer transport here is a learned approximation to a conditional nonlinear computation, not a fixed law of the transformer.

**23. Keep discovery and monitoring connected without claiming that one solves the other**

Once a source-effect predictor works, use it to prioritize the expensive auditor's finite tests. For candidate \((H,S,q)\), a simple acquisition rule is

\[
\mathrm{priority}(H,S,q)
=\frac{\max(\widehat D_q,0)+\lambda\,\mathrm{uncertainty}_q}
{\widehat{\mathrm{cost}}(H,S,q)}.
\]

Reserve a fixed exploration fraction for random candidates, and evaluate discovery at equal total cost. Uncertainty estimation is itself empirical; ensembles are one practical approximation. Validate on candidates not already selected by the teacher. This tests an amortized auditor, which may be useful even if online hazard prediction is weak.

It does not recover an absent cryptographic or literal trigger without a proposal mechanism. If the audit distribution never encounters distinguishing evidence, causal supervision cannot manufacture that evidence. Keep cold acquisition/GAC as a separate result.

**24. Theoretical statements worth making in the paper**

The strongest mathematical story is conditional and falsifiable:

- **Suppression/readout separation:** the draft's log-gradient identity explains one regime in which output probability can fall without changing attribution ranks. It is a toy mechanism to test, not a universal statement about fine-tuning.
- **Observability:** the conditional-variance decomposition specifies what any passive monitor can recover from its chosen features.
- **Representation sufficiency:** if an effect satisfies \(D_q(H,S)=f_q(U^\top h_S,U^\top h_b,\mathrm{policy})+\varepsilon\), then the irreducible error is governed by the residual conditional variation. The assumption must be assessed using held-out effect prediction and intervention tests.
- **Stability under a restricted shift:** for a fixed \(L_f\)-Lipschitz predictor, if a checkpoint/environment shift changes projected features by at most \(\delta\), predictions change by at most \(L_f\delta\). This says nothing about target changes, unseen mechanisms, or representation rotations outside the assumed bound.
- **Calibration:** episode-max conformal calibration controls one specified marginal false-alarm quantity under exchangeability.

These are standard identities or elementary consequences, not by themselves a new theory contribution. The scientific contribution would be identifying when their assumptions hold or fail in defended coding agents and using that knowledge to build a better auditor/monitor.

**25. What would make Phase 2 a convincing paper contribution**

The most persuasive package would contain three findings:

1. **Mechanism:** paired natural and internal interventions separate source recognition, retained source influence, and behavior suppression; the selected components preserve legitimate coding utility.
2. **Learning:** dense source-effect supervision improves low-false-alarm monitoring or auditing efficiency over behavior labels, with matched features, data, and compute.
3. **Generalization:** gains survive meaningful holdouts and syntax/authorization controls, while the paper accurately states whether the result covers a discovered cue, a contract family, or an unseen backdoor.

Report negative outcomes as conclusions. A source-only model matching the full method suggests cue memorization. A task/action probe matching it suggests generic semantics. A raw-state model matching causal subspaces means mechanistic feature selection was unnecessary for prediction. Strong effect prediction with weak hazard prediction supports an auditor, not a validated imminent-action monitor. A pre-execution signal with no pre-generation signal narrows the temporal claim.

The first work item should therefore be **P2-0 plus a small P2-1**, followed by the label-by-representation ablation. That ordering tests whether your present Phase 1 results contain a useful learning signal before committing to a large dynamics model.

**26. Minimal implementation contract**

The following are proposed records and operations, not existing repository APIs.

| Record | Required contents |
|---|---|
| Identity | Checkpoint and tokenizer hashes; serving/teacher configs; repository, task, lineage, and paired-group IDs; split |
| Cutoff | Tool-step index; exact prefix token IDs; observation and assistant-boundary positions; source byte/token spans; whether any action tokens are already present |
| Features | Hook definitions; selected layers; raw development states or frozen projections; source pools; policy/provenance fields; no future payload features |
| Teacher | Candidate continuation IDs and knowledge regime; original/replacement scores; signed source effects; random/control effects; replacement validity checks |
| Intervention | Component/site, donor identity, rank, replacement norm, removal/restoration effects, legitimate-utility effects, and any alignment limitation |
| Behavior | Decoder settings and seeds; parsed next-action category; independent violation label; rollout horizon; intercept status; missing-label mask |
| Monitoring | Source rankings, effect predictions, risk predictions, alarm cutoff, elapsed time, memory, and escalation cost |

Use a missing-label mask for unmeasured behavioral risk, not a fabricated benign label. Store greedy and sampled outcomes separately. Do not store a failed/censored intervention as a measured zero effect.

```text
OFFLINE, on discovery and fitting groups only:
  segment each prefix into model-visible source spans
  obtain algorithmic candidates and sample control spans
  score original and valid replacement prefixes
  record signed source effects for each declared obligation
  screen internal sites; measure selected finite swaps
  fit and freeze any causal projections on discovery groups
  extract prefix-only student features
  fit matched behavior-only, effect-only, and joint students

VALIDATION, before calibration/test:
  select feature sites, rank, student capacity, and score rule
  freeze the full monitor, including source selection and aggregation
  calibrate on separate benign episodes

ONLINE EVALUATION:
  prefill the actual current history
  read/project the selected prefix activations
  score all eligible source spans, then aggregate effect/risk outputs
  optionally run a budgeted counterfactual replay
  log any alarm before generation or dispatch, with its exact timestamp
  independently parse the model's proposed action without executing harm
```

The first analysis table should have one row per feature/supervision arm and columns for held-out effect error, source retrieval, pre-action recall at a fixed episode false-alarm target, authorized-operation false alarms, and total cost. This makes the scientific decision reviewable before adding a more elaborate model.

**Primary-source references and reading scope**

[1] Gao et al. *Patcher: Post-Hoc Patching of Backdoored Large Language Models*. USENIX Security 2026. Primary paper, methods and limitations inspected. [Paper](https://www.usenix.org/system/files/usenixsecurity26-gao-anjun.pdf).

[2] Anthropic. *Simple probes can catch sleeper agents*. 2024. Author research report, methods and caveats inspected. [Report](https://www.anthropic.com/research/probes-catch-sleeper-agents).

[3] Luo et al. *AgentLens: Interpretable Safety Steering via Mechanistic Subspaces for Multi-Turn Coding Agent*. 2026, arXiv:2606.22673. Methods, labels, and limitations inspected. [Paper](https://arxiv.org/html/2606.22673).

[4] *Agent Safety From Within: Detecting Harmful Trajectories from LLM Internal States*. 2026, arXiv:2609.33039v1. Methods, paired controls, readout selection, and evaluation scope inspected. [Paper](https://arxiv.org/html/2609.33039v1).

[5] Zhao et al. *LLMs Encode Harmfulness and Refusal Separately*. 2025, arXiv:2507.11878. Methods and causal reply-inversion experiments inspected. [Paper](https://arxiv.org/html/2507.11878v1).

[6] Huang et al. *Backdoor Decontamination Dynamics in LLM Agents*. 2026, arXiv:2608.11295. Threat model, outcomes, and J-Lens data access inspected. [Paper](https://arxiv.org/html/2608.11295v1).

[7] Yu et al. *Backdoor Attribution: Elucidating and Controlling Backdoors in Language Models*. 2025, arXiv:2509.21761v1. Probe and head-substitution methods and ablation table inspected. [Paper](https://arxiv.org/html/2509.21761v1).

[8] Kramár et al. *AtP\*: An efficient and scalable method for localizing LLM behaviour to components*. 2024, arXiv:2403.00745. Attribution-replacement failure modes inspected. [Paper](https://arxiv.org/html/2403.00745v1).

[9] Geiger et al. *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*. CLeaR 2024. Interchange intervention objective and limitations inspected. [Paper](https://proceedings.mlr.press/v236/geiger24a/geiger24a.pdf).

[10] *Obfuscated Activations Bypass LLM Latent-Space Defenses*. 2024, arXiv:2412.09565. Detector scope and SQL capability/obfuscation experiments inspected. [Paper](https://arxiv.org/html/2412.09565v1).

[11] McGuinness et al. *Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Unseen Activation Monitors*. 2025, arXiv:2512.11949. Threat model and mechanistic analysis inspected. [Paper](https://arxiv.org/html/2512.11949v1).

[12] Parrack et al. *Benchmarking Deception Probes via Black-to-White Performance Boosts*. arXiv:2507.12691. Abstract-level methodological use only; full-text retrieval was unsuccessful. [Abstract](https://arxiv.org/abs/2507.12691).

[13] Belrose et al. *LEACE: Perfect linear concept erasure in closed form*. arXiv:2306.03819. Abstract-level optional diagnostic: linear erasure can test reliance on labeled nuisance concepts, but does not prove absence of nonlinear information or preserve all causal mechanisms. Not required for the main protocol. [Abstract](https://arxiv.org/abs/2306.03819).

[14] Schmitt-Ulms. *Extracting hidden states from vLLM*. Official documentation blog, 2026-03-30. Version and extraction scope inspected. [Documentation](https://vllm.ai/blog/2026-03-30-extract-hidden-states).

[15] Angelopoulos and Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511. Standard calibration background; the episode-max specialization above is stated explicitly. [Primer](https://arxiv.org/abs/2107.07511).

The draft also cites *Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning* (arXiv:2609.00351). I found its indexed abstract but could not retrieve its full text in this review, so I did not rely on unverified methodological details or claim that the proposed Phase 2 is novel relative to every part of that work.
