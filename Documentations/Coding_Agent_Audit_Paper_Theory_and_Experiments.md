**From suppressed behavior to causal evidence in coding agents**

Research formulation and experiment plan for discussion with Daniel Kang. Based on the updated supplied draft, Pasted markdown(9).md, including its negative attention-route result. 30 September 2026.

**The paper should establish when internal evidence remains useful after behavioral suppression, and whether that evidence improves causal auditing and monitoring under a clearly stated information budget.** The present pilot motivates this question. It does not yet establish blind trigger discovery, gate saturation, or a reliable deployment monitor.

The most useful theoretical correction is that probability gradients and log-probability gradients behave differently. A backdoor action can become difficult to sample while its log-probability gradient remains large. This gives a plausible explanation for the reported success of ordinary saliency and a testable rule for when assistance-based continuation might add value.

All model-specific numbers below are reported in the attachment, not independently reproduced here. No repository, checkpoint, raw experiment logs, or GPU execution was available for verification. The mathematical propositions are derived under explicit simplified assumptions. Their applicability to the coding models is an empirical question. Elementary numerical checks of the scalar formulas are illustrative calculations, not new model results.

**1. What the updated pilot establishes, and what it leaves open**

The updated draft reports:

| Measurement | Strong 3B suspect | Defended 3B suspect | Interpretation supported by the report |
|---|---:|---:|---|
| Previously reported behavioral TPR | 1.00 | 0.21 | Different observed attack rates under the stated scorer |
| Median triggered-minus-neutral target log likelihood, 20 rows | +8.4 nats | +15.8 nats | Trigger-dependent relative target preference persists |
| Median trigger-token normalized rank, 15 resolved rows | 0.000 | 0.000 | Very high rank on the included examples, subject to unrounded ranks and denominator audit |
| Reported recall at k | 0.50 | 0.40 | Needs an explicit k and token-versus-example definition |
| Median log-likelihood change after trigger-span ablation | -6.0 nats | -15.6 nats | A large intervention effect on the supplied target's likelihood |
| Attention-route rank on three resolved short examples | 0.331 | 0.201 | The tested route score is worse than the reported saliency result |
| Assistance search in neutral contexts | No crossing within tested range | No crossing within tested range | The tested intervention did not reach the chosen threshold |

Treat these as three separate outcomes: target-conditioned localization is promising; the tested attention-route method is currently weaker; continuation has not yet recovered an input.

Several changes are necessary before submission.

- Replace “79% clean” with “21% attack success under this evaluation.” A stochastic attack rate is not a probability that the checkpoint is clean. If triggered trials were independent with probability 0.21, 13 trials would give about 95% probability of observing at least one failure.
- Replace “the backdoor fires” when only target likelihood was measured with “the trigger increases the supplied target's conditional likelihood.”
- Describe the present localization track as **exact-target-conditioned localization in trigger-containing histories, with the trigger position withheld**. The text is in the input; the target action is supplied. Neither condition is equivalent to discovering an absent trigger from ordinary histories.
- Do not infer saturation from a likelihood ratio or lower behavioral attack rate. Multiple mechanisms can produce those observations.
- Do not conclude global unreachability from one direction, one layer, and a finite assistance range. Record a right-censored threshold greater than the tested limit.
- Do not treat rounded normalized rank 0.000 as proof of rank one. Report integer ranks and tie handling.
- Fix evaluation-span bookkeeping before increasing sample size. Report every assigned row, why annotation failed, and both all-assigned and evaluable-row results. A trigger-string matcher belongs only in the evaluator.
- A log-likelihood ablation effect establishes an intervention effect on that score. Necessity for actual agent behavior requires corresponding action-level experiments.
- Compare route potential and saliency on exactly the same histories and target positions. Three short examples versus fifteen potentially longer examples do not isolate the method.
- Confirm that backward replay matches the evaluated model: dropout disabled, no optimizer update, identical adapters, templates, token IDs, precision where feasible, and recorded checkpoint identity. Autograd does not inherently require training-mode stochastic behavior.

**2. Closest prior work changes the novelty claim**

Patcher already localizes backdoor triggers using response-conditioned input-embedding gradient norms and adaptive clustering from a reported failure [r1]. That is very close to the pilot's successful operator. Unmasking Backdoors combines attention and gradient information for poisoned-input detection in encoder classifiers [r2].

Backdoor Decontamination Dynamics in LLM Agents already reports a distinction between trigger recognition and malicious execution, including intermediate trigger-related representations after benign responses are restored [r3]. Its J-Lens analysis uses backdoor-training and decontamination examples, which is a useful access distinction, but the broad observation “behavior improves while internal traces persist” is not itself new.

Discovering Backdoor Triggers studies internal methods given a known target action without trigger-containing prompts, with success on toy semantic backdoors and failures in more realistic settings [r4]. Haystack reconstructs triggers without assuming the trigger or target, using extraction and inference signals that include attention [r5]. Calling Haystack a strict black-box method would misstate its access.

The prospective contribution should therefore combine:

1. A carefully measured distinction between behavioral suppression, persistent conditional influence, and actual removal in long coding-agent histories.
2. A method that uses that influence to acquire reproducible causal evidence with a limited audit budget.
3. A monitor that can act before the unauthorized tool call, with calibrated episode-level false alarms and measured utility cost.
4. A theory explaining both the successful saliency regime and the regime in which assistance can help.

The elementary gradient identities are explanatory theory. The paper's novelty must rest on the measured phenomenon, validated method, and harder information setting. An award is not a controllable outcome; these are concrete ways to make the scientific contribution stronger.

**3. Formal problem and information tracks**

Let a coding-agent history at decision t be

\[
H_t=(x,\mathcal R_t,o_{1:t},a_{1:t-1},m_t),
\]

where x is the legitimate task, the repository state is \(\mathcal R_t\), observations are \(o\), prior actions are a, and any retained summary or explicit memory is m. A checkpoint \(\theta\), harness \(\mathcal E\), and decoder \(\kappa\) induce a distribution over trajectories

\[
\tau\sim P_{\theta,\mathcal E,\kappa}(\cdot\mid H_t).
\]

An independent predicate \(V_q(\tau)\) checks violation of obligation q. Define behavioral risk

\[
Q_{\theta,q}(H_t)=
\Pr_{\tau\sim P_{\theta,\mathcal E,\kappa}}[V_q(\tau)=1].
\]

This is distinct from the teacher-forced likelihood of one target continuation y:

\[
L_\theta(y;H_t)=
\sum_{u=1}^{|y|}
\log p_\theta(y_u\mid H_t,y_{<u}).
\]

A raw-model likelihood and samples from a temperature/top-p decoder need not agree. An exact sequence, an action prefix, and the broader violation event are also different probability objects.

The audit algorithm receives the suspect and authorized task/harness data. It selects experiments, spending costs for gradients, forward passes, context edits, internal interventions, and trajectory evaluations. Its output is either no witness within budget or

\[
\mathcal W=(H^*,S^*,q,\text{paired interventions},
\text{independent action-level evidence}),
\]

where \(S^*\) identifies the relevant source region or condition. No clean ancestor is an input to the strict auditor. Additional checkpoints may be used by the evaluator to study mechanisms and estimate false positives.

Define distinct information tracks:

| Track | Histories supplied | Target knowledge | Main claim |
|---|---|---|---|
| A: localization | Some histories already contain the trigger | Exact target action supplied | Rank and causally test an unknown source span |
| B: stream audit | Unselected histories, with unknown trigger prevalence | A declared security-contract library | Find actionable witnesses and control false alerts |
| C: acquisition | Seed histories contain no planted trigger | Contract library or explicitly declared exact target | Construct a naturally activating context |
| D: online monitor | Current history only, before the next action | Frozen auditor learned without test-trigger information | Warn before an unauthorized action |

The current strongest result belongs to A. B and D are promising paper extensions. C is the continuation goal and must remain a separate result.

Within each track, distinguish exact-target knowledge from obligation-only knowledge. Knowing the literal command, endpoint, or file argument is stronger information than knowing that unauthorized data transfer is prohibited.

**4. Proposition 1: behavioral suppression can preserve attribution rank**

Consider a simplified binary event with

\[
q_d(c)=\sigma(s(c)-d),\qquad d\geq0,
\]

where s is a fixed context-sensitive logit and d is a context-independent suppression offset. Let \(e_j\) denote input-token embeddings. Then

\[
\nabla_{e_j}q_d
=q_d(1-q_d)\nabla_{e_j}s,
\]

but

\[
\boxed{\nabla_{e_j}\log q_d
=(1-q_d)\nabla_{e_j}s.}
\]

Consequently, the ranking of \(\|\nabla_{e_j}\log q_d\|\) across tokens is exactly invariant to d for a fixed context, apart from ties. The multiplier \(1-q_d\) is common to all source positions. As d grows, probability sensitivity vanishes while log-probability sensitivity tends to the detector-logit sensitivity.

**Proof.** Differentiate the sigmoid and use \(\nabla\log q=(\nabla q)/q\). Multiplication of all token scores by one positive scalar preserves their order.

For independent draws from this binary event, the number needed to observe at least one event with probability \(1-\alpha\) is

\[
N_\alpha(d)=
\left\lceil\frac{\log\alpha}{\log(1-q_d)}\right\rceil
\sim\frac{\log(1/\alpha)}{q_d}.
\]

If \(s(c)\) is fixed and d increases, the sampling requirement grows exponentially while the ideal log-gradient ranking stays constant. This is a conditional toy separation, not a guarantee for real transformers or an unknown target.

**Corollary: a larger log-likelihood ablation effect can accompany lower absolute risk.** Let \(s_T>s_N\) be logits before and after a task-valid source replacement. Then

\[
D_d=\log\sigma(s_T-d)-\log\sigma(s_N-d)
\]

satisfies

\[
\frac{\partial D_d}{\partial d}
=\sigma(s_T-d)-\sigma(s_N-d)>0,
\quad
\lim_{d\to\infty}D_d=s_T-s_N.
\]

Thus an increasing log-ablation effect does not prove that the underlying backdoor became stronger.

For illustration only, set \(s_T=-1\), \(s_N=-4\):

| Suppression d | Event probability | Log-probability derivative with respect to \(s_T\) | Log-ablation effect | Independent samples for 95% observation |
|---:|---:|---:|---:|---:|
| 0 | 0.26894 | 0.73106 | 2.70489 | 10 |
| 2 | 0.04743 | 0.95257 | 2.95389 | 62 |
| 4 | 0.00669 | 0.99331 | 2.99362 | 447 |
| 8 | 0.000123 | 0.999877 | 2.999883 | 24,277 |

These are analytical values, not measurements from the supplied models.

**The experiment implied by the proposition.** Across a matched defense trajectory, measure absolute target likelihood, event risk, logit differences, normalized input-gradient direction, and finite replacement effects. Test whether an offset fitted on held-out neutral contexts predicts the triggered-context behavior. If normalized gradients drift substantially, or the offset varies strongly with the source, pure output suppression is inadequate.

For a categorical next-token model,

\[
\frac{\partial\log p(y)}{\partial z_k}
=\mathbf1\{k=y\}-p(k).
\]

There is no multiplicative p(y) term suppressing every derivative. This explains why the draft's general statement “low target probability implies tiny target log-likelihood gradients” should be removed. Hidden Jacobians can still suppress the input gradient.

**5. Proposition 2: identify the regime where a hidden gate really defeats log attribution**

Model a target probability locally as

\[
p(c)=\epsilon+a\,\sigma(s(c)),
\qquad \epsilon\geq0,\ a>0,\ \epsilon+a\leq1,
\]

with a and epsilon held fixed with respect to the context coordinates under analysis. Then

\[
\boxed{
\nabla_c\log p
=
\frac{a\sigma(s)(1-\sigma(s))}
{\epsilon+a\sigma(s)}
\nabla_c s.
}
\]

Two regimes follow:

- If the gated contribution dominates the background, \(\epsilon\ll a\sigma(s)\), the coefficient approaches \(1-\sigma(s)\). Log attribution may remain informative even for a small gate value.
- If the background dominates, \(a\sigma(s)\ll\epsilon\), the coefficient is approximately \(a e^s/\epsilon\) in the negative tail. Input attribution can then vanish despite a nonzero detector gradient.

Assistance is most motivated in the second regime, provided a finite intervention exposes a consequential and naturally reachable pathway. If a or epsilon depend on context, their derivatives must be included; the displayed decomposition is no longer complete.

This suggests a principled adaptive audit: start with target-conditioned attribution, validate promising effects, and allocate assistance experiments where the ordinary score is weak but finite internal effects are strong. It does not require that every backdoor be a saturated selector.

**6. A concrete auditor built around the successful signal**

At each eligible history, construct a finite set of candidate continuations using a frozen procedure. For the exact-target track, use the supplied target. For an obligation-only track, use independently prepared alternatives expressing the relevant authorized and unauthorized choices without copying the planted action's literal endpoint or argument.

For finite, disjoint complete-sequence sets, a score can be

\[
\phi_q(H)=
\log\sum_{y\in\mathcal Y_q^-}p_\theta(y\mid H)
-
\log\sum_{y\in\mathcal Y_q^+}p_\theta(y\mid H).
\]

Use unique complete sequences or prefix-free events; overlapping prefixes cannot be summed as disjoint probability mass. This is still a restricted candidate-set score, not the true probability of the entire security event. Length-normalized versions can be useful scores but lose the probability interpretation.

The attachment's single compliant action was poorly matched to the history. Do not repair that by treating every positive score as an open gate. Use several contextually plausible compliant actions, validate them independently, and test whether conclusions survive changing their phrasing. For localization, the finite source-replacement contrast of a single target score can also avoid an arbitrary compliant denominator.

Obtain token scores, then aggregate into source units: observation lines, comment spans, syntax nodes, or document fields. Predeclare the aggregation and normalize for span length using matched controls. Raw sums favor longer spans; raw maxima have a larger multiple-comparison opportunity in longer spans.

For each high-ranked span S, generate task-valid replacements from \(R_S(\cdot\mid H)\). Define

\[
D_q(H,S)=
\phi_q(H)
-\mathbb E_{r\sim R_S}\phi_q(H[S\leftarrow r]).
\]

Choose R before testing the final candidate. Include matched replacements preserving tool-result structure, line numbering where relevant, and the legitimate task. Emptying an observation is a much stronger intervention than replacing a comment.

The auditor should return a small set of candidates with reproducible signed finite effects, rather than label the maximum saliency token malicious. Every history has a maximum.

A useful constrained localization objective is

\[
\min_S \operatorname{cost}(S)
\quad\text{subject to}\quad
\operatorname{LCB}(D_q(H,S))\geq d_{\min},
\quad
U(H[S\leftarrow r])\geq u_{\min},
\]

with separate held-out confirmation. Here U measures retained task validity or utility. This objective finds a small influential source region; an action-level test is still needed to establish that it controls a violation.

Measure behavioral necessity as

\[
N_q(S)=Q_{\theta,q}(H)
-\mathbb E_rQ_{\theta,q}(H[S\leftarrow r]).
\]

Measure transplant sufficiency on fresh, task-matched histories:

\[
F_q(S)=
\mathbb E_{H'\sim\mathcal D_{\mathrm{fresh}}}
\left[
Q_{\theta,q}(H'\oplus S)-Q_{\theta,q}(H')
\right].
\]

Restore the removed source in a third arm. Removal, matched replacement, and restoration together separate a specific source effect from generic disruption. Transplant failure may reflect a missing conjunction or action opportunity; it does not by itself disprove a causal effect in the original history.

For two candidate spans, test the finite interaction

\[
I_{12}=
\phi(H)-\phi(H_{\setminus S_1})
-\phi(H_{\setminus S_2})
+\phi(H_{\setminus S_1,S_2}).
\]

This can reveal conjunctions and redundancy. Higher-order interactions require additional experiments; pairwise tests cannot certify their absence.

**7. Proposition 3: when gradient screening can support finite causal tests**

Let L be a smooth target score on a fixed-length embedding representation. For a replacement displacement v supported on one span,

\[
D=L(e)-L(e+v),\qquad
\widehat D=-\nabla L(e)^\top v.
\]

If the Hessian operator norm along the entire segment \(e+t v\), \(0\leq t\leq1\), is at most M, then

\[
\boxed{|D-\widehat D|\leq\tfrac12M\|v\|^2.}
\]

**Proof.** Apply Taylor's theorem with integral remainder and bound the quadratic form by the operator norm. This is a standard local approximation result, not a novel attribution theorem.

If estimation error contributes at most epsilon, candidates separated by more than twice the resulting total error retain their order. Large curvature, large replacement displacement, or path cancellation can make ordinary screening fail. AtP* provides related motivation for correcting local nonlinear effects [r6].

Natural replacements that change token count or positions do not directly satisfy this fixed-dimensional statement. Either use appropriately aligned, equal-length candidates for the diagnostic or rely on measured finite effects. A continuous interpolation can leave the natural-text manifold even when its endpoints are valid.

A second useful bound concerns signal versus nuisance. Suppose

\[
\nabla_{e_j}L=a(c)\nabla_{e_j}s+n_j,
\quad a(c)>0,
\]

with nuisance norm at most B and gradient-estimation error at most epsilon. If a relevant span has detector-gradient norm at least gamma and every irrelevant span has norm at most nu, then

\[
a(c)(\gamma-\nu)>2(B+\epsilon)
\]

is sufficient for that relevant span to outrank all irrelevant spans. The proof follows from the triangle inequality. This shows exactly what the logistic model alone does not establish: the detector itself must have source-specific sensitivity, and nuisance gradients must not dominate it.

For K fixed, aligned candidates with population mean score gap Gamma and independent sub-Gaussian measurement noise with proxy variance \(\sigma^2\), averaging R measurements gives

\[
\Pr(\text{any score error}>\Gamma/2)
\leq2K\exp\!\left(-\frac{R\Gamma^2}{8\sigma^2}\right).
\]

This follows from a concentration bound and a union bound. It motivates measuring ranking margins and their stability rather than only median ranks. Repository dependence, adaptive candidate selection, and unaligned source positions invalidate a naive application of this formula. Use it as an experimental design calculation under stated assumptions.

**8. What search-space reduction can legitimately mean**

For a history with N candidate source spans, let \(\widehat S_k\) be a shortlist of k spans. Report:

\[
\operatorname{Hit@k}
=\Pr(\widehat S_k\cap S^\star\ne\varnothing),
\]

\[
\operatorname{TokenRecall@k}
=\mathbb E
\frac{|\widehat S_k\cap S^\star|}{|S^\star|},
\]

with clearly specified span-versus-token units. Report MRR, exact-span coverage, selected-token budget, and finite causal effect on held-out replacements. One high-ranked token in a multi-token trigger is not complete recovery.

For a simple illustrative search model, let j-star be the relevant location, and let a value proposal succeed with probability rho conditional on editing that location. If the search chooses a location uniformly within a shortlist with probability \(1-\eta\) and uniformly over all locations with probability eta, then

\[
p_{\mathrm{hit}}=
\rho\left[
(1-\eta)\frac{\mathbf1\{j^\star\in\widehat S_k\}}{k}
+\frac{\eta}{N}
\right].
\]

This exhibits the benefit of localization and the necessity of continued exploration. If the true location is missed and eta is zero, the search can fail forever.

Averaging this expression over held-out histories gives a mean per-proposal hit rate involving Hit@k. Do not invert that mean and call it the average discovery time across heterogeneous histories: \(\mathbb E[1/p]\neq1/\mathbb E[p]\). Expected-time claims require explicit repeated-trial assumptions or empirical time-to-witness curves.

Location reduction is also not value recovery. A shortlist can reduce where to search while leaving an enormous space of possible strings, relationships, or event sequences. For current localization results, the trigger string is already present in the history. The cold-acquisition track removes that advantage.

**9. Repair GAC as a conditional extension**

For a fixed candidate intervention route r and normalized assistance amplitude beta, define

\[
\beta_r^*(H)=
\inf\left\{
\beta\in[0,B]:
\phi_q(H;I_{r,\beta})\geq\tau_q,\
U(H;I_{r,\beta})\geq u_{\min}
\right\}.
\]

An empty feasible set is recorded as no crossing within the tested range. If utility remains acceptable throughout the range and only the score threshold is unachieved, the threshold is right-censored above B. If utility fails first, record a distinct utility-limited outcome. Do not assign either outcome an invented finite value such as 360 and average it as an observation.

Use a fixed threshold calibrated on development data for the declared score and information track. A threshold based on the held-out triggered context uses oracle information.

If the threshold is allowed to vary with H, the correct derivative is

\[
\nabla_H\beta_r^*
=
\frac{\nabla_H\tau_q(H)-\nabla_H\phi_q(H;\beta_r^*)}
{\partial_\beta\phi_q(H;\beta_r^*)}.
\]

The original formula omitting the threshold derivative only applies to a fixed threshold. There is a particularly important counterexample. If

\[
\phi(H;\beta)=a(H)+k\beta,\qquad
\tau(H)=\phi(H;0)+\Delta,
\]

then \(\beta^*(H)=\Delta/k\), independent of context. A seemingly sensible context-relative calibration can remove the very ranking signal continuation needs.

Raw assistance is also scale dependent: replacing a residual direction u by \(a u\) divides the required beta by a. Normalize candidate directions using a predeclared activation metric, and report the actual intervention norm and its effects on reference behavior. A functional cost such as

\[
C(I)=
\mathbb E_{b\sim\mathcal B}
\operatorname{KL}
\left(p_\theta(\cdot\mid b)\Vert p_\theta^I(\cdot\mid b)\right)
\]

is useful alongside actual code utility. Neither a local norm nor reference KL certifies natural reachability.

**Proposition 4: local boundary stability.** At a fixed-route interior crossing, assume phi is continuously differentiable, the utility constraint is inactive, and \(\partial_\beta\phi\geq m>0\) in a neighborhood. Then the implicit-function theorem yields the stated boundary gradient. If the perturbed root remains in this same neighborhood, the score perturbation is uniformly bounded by \(\epsilon_\phi\), and the threshold perturbation by \(\epsilon_\tau\), the mean-value theorem gives

\[
|\widetilde\beta^*-\beta^*|
\leq(\epsilon_\phi+\epsilon_\tau)/m.
\]

A first-order expansion gives

\[
\delta\beta=
\frac{\delta\tau-\delta\phi}{\partial_\beta\phi}
+\text{higher-order terms}.
\]

Thus nearly flat curves make the root and its gradient unstable. Boundary clipping at beta zero, an active utility constraint, switching between routes, and multiple crossings require separate treatment. The minimum over routes is generally nonsmooth.

For discrete edits, use the continuous gradient to rank proposals and accept only actual, task-valid edits with measured improvement. The final condition is a fresh zero-intervention action-level witness. A literal trigger does not mathematically imply zero useful embedding gradient, but neither does the boundary formula guarantee a discrete path to it.

**10. Formulate a monitor separately from an offline auditor**

Let \(S_\omega(H_t)\) be a frozen score computed using only the history available before the next action. It may use a fixed probe-action library, selected activations, or a student trained to approximate the expensive auditor. It must not use the future harmful continuation selected after observing the run.

The alarm time is

\[
T_A=\inf\{t:S_\omega(H_t)>c_\alpha\}.
\]

Let \(T_V\) be the first unauthorized action. Evaluate

\[
\Pr(T_A<T_V\mid T_V<\infty),
\]

the false-alarm probability per benign episode, warning lead time in tool calls, retained task success, and wall-clock overhead. A detector firing after the action is unsuitable for a pre-action monitoring claim.

**An episode-level calibration guarantee.** On n exchangeable, independently held-out benign episodes, calculate

\[
M_i=\max_{t\leq T_i}S_\omega(H_{i,t}).
\]

Set \(k=\lceil(n+1)(1-\alpha)\rceil\) and use the k-th ordered calibration value as \(c_\alpha\), with \(c_\alpha=+\infty\) when \(k>n\). With strict thresholding and conservative treatment of ties,

\[
\boxed{
\Pr\left(\exists t:S_\omega(H_{\mathrm{new},t})>c_\alpha\right)
\leq\alpha.
}
\]

**Proof.** The rank of the new episode's maximum among n+1 exchangeable maxima is uniform in the absence of ties; conservative ties preserve the inequality. This is the standard split-conformal rank argument [r7], applied to entire episodes so that all monitored steps are included in the calibration object.

The guarantee is marginal over that episode distribution. It does not survive arbitrary repository shift, adaptive adversaries, changed horizons or policies with a different episode distribution, or reuse of calibration data to tune the monitor. It is not a model-level guarantee over unseen checkpoints.

A practical two-stage monitor can use inexpensive selected-state features at every decision and invoke the full gradient auditor only when needed. Train the student to predict audit outcomes on development data, then evaluate new trigger families and repositories. This is worth building only after the expensive auditor demonstrates useful pre-action discrimination without exact-target leakage.

**A specific learning experiment: counterfactual supervision under behavioral suppression.** Most ordinary runs of a strongly suppressed backdoor may have benign outcome labels. The expensive auditor can still supply a vector of measured source effects

\[
\mathbf d(H,S)=(D_1(H,S),\ldots,D_{|\mathcal Q|}(H,S)).
\]

Train a small predictor \(g_\omega(z(H),S)\) on those effects, where z contains only selected states available before the action. A loss is

\[
\mathcal L(\omega)=
\mathbb E_{H,S}\sum_q
\operatorname{Huber}
\left(g_{\omega,q}(z(H),S)-D_q(H,S)\right).
\]

Sample both proposed and randomly selected source spans so selection artifacts do not become the prediction target. Use independently confirmed behavior to validate which predicted effects are security relevant. A large predicted log-score effect is not itself a violation probability.

Compare three monitors with the same architecture and cost: one trained on observed behavior labels, one on passive features or passive anomaly labels, and one on these counterfactual-effect targets. Test whether counterfactual supervision improves early warning on held-out defense stages and trigger families at the same episode false-alarm rate. Include the cost of producing its labels.

The strict single-suspect version learns from interventions on that suspect and available contexts. A monitor meta-trained on other planted models is a separate information setting. Neither version can assume it will learn a hidden condition that produces no useful signal anywhere in its training experiments.

**11. The experiments that could materially strengthen the paper**

**E1. Reproduce the pilot without selection or annotation leakage.** Freeze the auditor and all scoring choices, then evaluate every assigned trajectory in the existing pool. Record exact source byte offsets when the evaluator constructs the history, and map them through the tokenizer. The auditor receives no trigger annotations. Include all span-resolution failures in a reported denominator.

Audit whether the ablated span was predicted by the method or supplied by the curator. Run both and label them separately. An oracle ablation validates the planted cue; an ablation of the method's prediction validates the auditor. Report integer ranks, Hit@1/5/10, token recall, selected-token budget, and predicted-span necessity.

Check chosen/rejected histories token by token to establish that the context intervention is the intended difference. Strip the target response from every input prefix. Verify action boundaries, candidate tokenization, and whether log probabilities include tool-format tokens, the command, and termination. Reconcile the source of the TPR values with the exact histories and decoder used in the new experiments.

This is the first gate because correcting bookkeeping can change every headline.

**E2. Locate the signal before the first unauthorized commitment.** Decompose

\[
L_\theta(y;H)=\sum_u\ell_u(H,y_{<u}),
\qquad
G_{u,j}=\nabla_{e_j}\ell_u.
\]

Score separately the tool-name choice, argument prefix, first contract-critical argument, later argument tokens, and termination. Later teacher-forced tokens may contain explicit descriptions of the forbidden behavior and can support gradients that would not have been available from a naturally generated prefix.

Measure localization at successive real history cutoffs, beginning immediately after the untrusted observation and ending just before the first unauthorized action. Recompute the full valid prefix at each cutoff.

A strong outcome is successful localization using a generic, frozen probe before the harmful action appears. If localization only emerges while teacher-forcing a long, exact payload, the current method is useful forensic attribution but weaker evidence for monitoring.

**E3. Build a genuine within-lineage defense trajectory.** For each independently trained backdoored seed, retain the checkpoint before defense and several checkpoints along one specified benign-training schedule. Repeat with the same tasks, trigger placement procedure, and decoding configurations. Treat 3B and 7B as separate strata; model IDs and different attack rates do not establish a common suppression axis.

At each checkpoint measure:

- Actual violation probability and authorized task utility.
- Exact-target probability and broader action-event probability, where measurable.
- Trigger-neutral log-probability and log-odds differences.
- Gradient norms, normalized gradient direction similarity, and source rank.
- Predicted-span finite effects and action-level necessity.

Test the offset-suppression prediction from Proposition 1 against alternatives: weakened detector, damaged payload generation, competing safe action, changed output format, and rerouting. Estimate model parameters on development or neutral examples and test predictions on held-out triggered examples.

Increasing the number of episodes improves precision within a checkpoint; it does not create new independent model lineages. A practical main-study starting point is several independent poisoning seeds, both dense model sizes, and four or more matched defense stages. Determine episode counts from desired interval widths and pilot variance, not a claim that n=50 is inherently sufficient.

**E4. Remove exact-target knowledge systematically.** Compare the following target constructions while keeping histories and budgets fixed:

| Target access | What changes | Scientific question |
|---|---|---|
| Exact planted sequence | Supplied command and literal arguments | What is the best-case diagnostic ceiling? |
| Semantically equivalent variants | Independently varied syntax and irrelevant literals | Is localization tied to one memorized response? |
| Contract-level probes | No planted endpoint or exact file argument | Can an auditor use a general policy rather than attack knowledge? |
| Held-out obligation families | Probe construction fixed without this payload family | Does the method discover an unanticipated behavior class? |

For a target set containing multiple probes, count every scoring pass and correct selection on the maximum score. Keep a vocabulary-overlap-matched safe probe control. An exact target can legitimately define one track, but it cannot stand in for an unknown payload.

This experiment is more valuable for the main claim than extensively tuning another attribution operator.

**E5. Establish a causal source effect under realistic controls.** For the predicted span, compare original context, matched replacement, removal, restoration, and transplantation to fresh histories. Include random spans matched by length, role, position, and source type; similar benign comments; legitimate instructions that should influence the action; and authorized tasks using similar security vocabulary.

Evaluate an independent action predicate on new rollouts. If the recorded tool call itself violates policy, intercept it before dispatch and count an attempted unauthorized action. Respect the stated analysis-only setup: do not execute the emitted exfiltration command. Interception supports an action-issuance claim, not a claim that real data was transferred.

Where the task concerns generated-code defects rather than tool misuse, use local test fixtures and an independent program checker. Keep these two outcome types separate.

**E6. Distinguish retained recognition from retained harmful control.** After a source is discovered, perform matched activation/path interventions within each checkpoint. Use several layers and agent stages to test whether source-dependent state influences the contract decision. Measure both representation decodability and finite causal effects.

Try restoring or suppressing source-dependent state while leaving ordinary task information intact. A restored malicious action requires stronger evidence than a restored trigger-token readout. Selective intervention effects that track the actual action separate recognition from downstream control.

Cross-checkpoint patching can be a useful evaluator-only diagnostic, but activation coordinates may drift even within a lineage. Use it only with alignment checks and within-checkpoint controls. Do not give the strict auditor access to the original ancestor while claiming single-checkpoint access.

**E7. Test temporal and conjunctive triggers.** Vary when the source is observed, its position, distance from the action opportunity, and whether it remains in raw context or is carried by a summary. Construct controlled two-source or order-dependent conditions in the evaluator's held-out suite.

Run crossed source interventions and measure the interaction \(I_{12}\). A monitor should respond to the relevant conjunction or sequence rather than flag every occurrence of either benign component. A frozen-transcript gradient provides a local explanation; fresh rollouts determine whether changing the earlier source actually changes the later action.

Never reuse downstream KV state after editing its causal prefix. Recompute the consistent history or label the KV patch as an artificial mechanism intervention.

**E8. Test policy specificity and legitimate rare behavior.** Include rare but authorized administrative actions, test fixtures containing security-related strings, unusual comments, and ordinary coding mistakes. Evaluate whether the auditor distinguishes an unauthorized dependency from a rare legitimate one.

Use matched benign fine-tunes in the evaluator, beyond one base checkpoint. A neutral context always has a top-ranked token, and legitimate source spans can be causally necessary for correct work. Require a contract-specific effect rather than “salient and necessary” alone.

This experiment determines whether the monitor learns security-relevant conditional control or a vocabulary blacklist.

**E9. Compare methods under matched information and total cost.** Strong baselines include:

- Target-conditioned input saliency and gradient-times-replacement-displacement.
- Patcher-style saliency plus its source-grouping rule, adapted transparently to the action setting.
- Black-box finite span replacement or hierarchical occlusion with the same supplied target and histories.
- A behavioral investigator with the same context proposal access.
- A Haystack-style extraction pipeline in the unknown-trigger acquisition track, with its actual access requirements.
- Simple lexical or source-location baselines, including a post-discovery string-matching monitor as a limited baseline.
- Integrated gradients or AtP*-style corrections on a smaller matched subset when curvature is the concrete risk.

Count full-prefix recomputation, backward passes, candidate-target multiplicity, expert or route interventions if any, and tool execution. Report wall time, GPU time, peak memory, and witness recovery versus budget. Report offline construction cost separately from amortized online cost.

**E10. Give the negative attention result one bounded diagnostic follow-up.** On the same histories and the same target token or token set, compare input gradients with route potential. Verify grouped-query key/value indexing, output-projection adjoints, masks, positions, and whether the chosen decision position is where the cue is read.

Test a small preregistered aggregation across relevant positions. Sums over layers and heads are ranking heuristics; overlapping causal paths mean they are not automatically a faithful total-effect decomposition. Validate shortlisted routes with finite path interventions.

If it remains weaker, retain the negative result and stop tuning it on the evaluation set. The paper can explain why a direct reading-edge score loses to total input sensitivity. It need not invent a new operator to have a useful contribution.

**E11. Test corrected continuation only where ordinary attribution is weak.** Select cases before seeing the outcome, using a frozen criterion such as poor source-ranking stability together with a strong finite internal intervention effect. Compare fixed calibrated thresholds, normalized route-specific assistance, and the old residual-direction baseline.

Measure uncensored crossing rate, utility-limited rate, root stability, prediction of useful discrete edits, and zero-intervention witness recovery. An oracle trigger-derived direction can establish steerability as a diagnostic ceiling, but belongs in an explicitly privileged arm.

Do not count a threshold at zero as a witness until free generation independently satisfies the contract predicate. Do not improve the threshold after looking at the held-out trigger and then report a blind result.

**E12. Run an actual monitoring study and a separate cold-acquisition study.** For monitoring, freeze the method before testing, calibrate episode maxima on held-out benign data, and compare early warning at fixed false-alarm budgets. Report warning lead time and agent utility after the chosen intervention policy.

For cold acquisition, begin with histories that contain no planted trigger. Use an expandable, task-valid proposal process and retain broad exploration. Compare discovery curves with and without the internal ranking or continuation signal. When retrieval or memorization extraction supplies a trigger-containing history, count its cost and record that acquisition route.

Keep zero-witness outcomes. A localization result cannot be promoted to cold discovery by renaming the same trigger-containing inputs.

**12. Experiments with the highest chance of changing the scientific conclusion**

| Priority | Experiment | Why it is decisive | Result that narrows the paper |
|---|---|---|---|
| First | E1 plus E2 | Checks whether the signal survives honest denominators and exists before action commitment | Signal depends on selected rows or the exact teacher-forced payload suffix |
| First | E4 | Removes the strongest attack-specific information | Generic contract probes lose localization |
| First | E5 | Converts attribution into action-level causal evidence | Only log likelihood changes, or legitimate utility collapses |
| Main | E3 | Tests a mechanistic account of the observed defense effect | Signal follows dataset/decoder changes instead of a matched training trajectory |
| Main | E8 plus E12 monitoring | Tests security specificity and operational value | False alarms or overhead erase the benefit |
| Main | E9 | Tests whether internal access pays for its cost | Strong finite-edit or behavioral baselines recover as many witnesses |
| Extension | E7 | Tests distributed and delayed agent conditions | Result only holds for a literal cue next to the action |
| Conditional extension | E11 plus cold acquisition | Tests the original GAC objective | Assistance creates internal failures but no natural trigger |
| Bounded diagnostic | E10 | Explains the reported negative operator comparison | Route score remains inferior |

A large learned dynamics network is not on the critical path. Add one only after a simpler frozen feature set or scheduler has demonstrated a clear bottleneck and useful supervision.

**13. Statistical design and selection discipline**

Freeze source aggregation, candidate probes, thresholds, and budgets on development lineages. Use separate data for method development, monitor calibration, and final evaluation. Put closely related checkpoints, repositories, and trigger variants on the same side of the relevant split.

For localization, report the full rank distribution and every denominator. For action-level necessity, estimate paired changes in violation probability and retain the task-utility outcome. Use independent confirmation rollouts after the adaptive search. Matched seeds can reduce variance where the harness permits, but do not replace independent repetitions or justify independence across repository variants.

For cross-model claims, uncertainty must reflect the number of independent training lineages. With very few lineages, show each lineage and avoid a precise-looking population confidence interval derived from thousands of correlated token observations.

For a fixed bounded audit budget B, report

\[
F(B)=\Pr(\text{a confirmed natural witness is found by cost }B).
\]

Plot the paired difference between methods over a shared set of organisms and repositories. Treat uncompleted searches as budget-censored, not successful at the budget endpoint. Avoid means over only successful audits.

For GAC, report no-crossing and utility-limited outcomes separately; a crossing-probability curve versus assistance budget is often more honest than a median threshold when most observations are censored.

For an online monitor, report episode-level false alarms, pre-action true positives, lead time, and utility. Zero false alarms on n independent null units gives the one-sided 95% exact upper bound

\[
1-0.05^{1/n}.
\]

For n=6 this is about 39.3%; about 299 independent null units with zero events are needed to put this bound below 1%. The unit must match the claim: episodes for episode risk, independently sampled models for model-level risk. A conformal marginal guarantee and a binomial confidence interval answer different questions and should not be conflated.

**14. The figures and tables that would make the argument assessable**

| Figure or table | Content | Claim it tests |
|---|---|---|
| Main Figure 1 | Same-lineage defense trajectories: behavior, target likelihood, source rank, and predicted-span causal effect | Whether behavior and useful conditional influence separate under actual training |
| Main Figure 2 | Theory curves plus empirical regime tests for log gradients and background-dominated hidden gates | Why saliency works in some cases and assistance is needed in others |
| Main Figure 3 | Exact-target to contract-only access ablation, at matched cost | Whether the method depends on knowing the planted payload |
| Main Figure 4 | Removal, replacement, restoration, and transplantation on fresh rollouts | Whether the predicted source controls unauthorized action |
| Main Figure 5 | Pre-action monitoring at fixed episode false-alarm rates, with utility and latency | Practical value beyond forensic explanation |
| Main Table 1 | Information-access contract for every method and track | Fairness and the exact meaning of blind access |
| Main Table 2 | Independent lineages, attack families, histories, unresolved rows, and cost | Reproducibility and scope |
| Extension Figure | Natural witness recovery versus total cost for cold acquisition | The original GAC claim, only if supported |
| Appendix | Matched attention-route comparison and unsuccessful continuation cases | Boundaries and negative findings |

Do not plot the attachment's separate 3B and 7B checkpoints as one causal defense trajectory. Do not draw theory curves as if they were measured model outcomes.

**15. A concrete next sequence of work**

1. Reconcile the pilot rows, exact scores, and evaluator annotations. Freeze a minimal saliency auditor.
2. Run pre-commitment and target-knowledge ablations on both current suspects, with predicted-span rather than only oracle-span interventions.
3. Confirm source necessity and restoration on the original vLLM path using intercepted, independently parsed actions.
4. Build matched defense trajectories and evaluate the suppression hypotheses using the same held-out histories.
5. Run the authorization and benign-specialist controls, then calibrate and evaluate the monitor.
6. Compare total cost against Patcher-style and finite-edit baselines.
7. Pursue GAC on a preregistered subset of genuinely weak-gradient cases. Keep it as an extension until it produces natural witnesses.

The strongest attainable central claim would be conditional on those results:

**“We characterize when benign post-training suppresses backdoor actions without removing source-specific causal influence, and use that influence to localize and monitor unauthorized behavior in coding agents without a clean reference checkpoint. Under explicitly stated target knowledge, the auditor provides independent causal witnesses and the monitor improves pre-action detection at a fixed episode-level false-alarm budget.”**

If contract-only probes fail, retain exact-target-conditioned auditing in the title and claim. If monitoring fails, present an offline auditing paper. If cold acquisition succeeds across withheld mechanisms, then the original GAC discovery claim becomes a separate, substantially stronger contribution. The results should choose the final scope.

**Primary-source ledger**

| Source | What was checked | Why it matters here |
|---|---|---|
| [r1] Patcher: Post-Hoc Patching of Backdoored Large Language Models, USENIX Security 2026 | Official abstract and full-paper saliency/method sections | Very close response-conditioned gradient localization baseline |
| [r2] Unmasking Backdoors, ICLR 2026 | Primary abstract | Existing gradient/attention anomaly detection in encoder models |
| [r3] Backdoor Decontamination Dynamics in LLM Agents, August 2026 | Primary full text, including J-Lens and its probing-data access | Existing recognition/execution dissociation and internal traces after decontamination |
| [r4] Discovering Backdoor Triggers, August 2025 | Authors' research report and method sections | Known-target recovery without trigger-containing prompts; successes and failures |
| [r5] The Trigger in the Haystack, February 2026 | Primary full text, including internal-signal and reconstruction sections | Unknown-trigger/unknown-target acquisition baseline with inference and attention access |
| [r6] AtP*, 2024 | Primary full-text saturation and cancellation sections checked in the preceding analysis | Local gradient failure and finite nonlinear correction |
| [r7] A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification | Primary paper record; standard exchangeable-rank argument derived explicitly above | Conventional basis for the proposed episode-max calibration |

The original organism model-card pages could not be independently retrieved in this session. Architecture, training provenance, TPR values, and pilot measurements are therefore attributed to the supplied draft. The PersistBD name alone should not be equated with a similarly titled persistence paper without verifying authors and release provenance.

[r1]: https://www.usenix.org/system/files/usenixsecurity26-gao-anjun.pdf
[r2]: https://arxiv.org/abs/2510.04347
[r3]: https://arxiv.org/html/2608.11295v1
[r4]: https://www.lesswrong.com/posts/kmNqsbgKWJHGqhj4g/discovering-backdoor-triggers
[r5]: https://arxiv.org/html/2602.03085v1
[r6]: https://arxiv.org/html/2403.00745v1
[r7]: https://arxiv.org/abs/2107.07511

---

**16. Operational experiment battery — the concrete runnable jobs**

This section turns §11's E1–E12 and §12's priority ordering into experiments that actually run on our stack (DeltaAI GH200, the four `qiusizhan/swe-audit-*` organisms, the real `uiuc-kang-lab/PersistBD` eval trajectories). Each hypothesis `H*` names the exact script, the models, the metric, and the decision rule that would confirm or narrow the claim. Code: `experiments/gac/` and `src/audit/`. One driver, `experiments/gac/battery.py`, loads a model **once** and runs H1–H4 together (a short saliency/scoring job is dominated by model load, so batching the hypotheses per model is the real speed-up, not a kernel — see §10.5 of the proposal).

**16.0 The E1 gate is cleared first.** Before scaling, the trigger-span resolver was hardened (`experiments/gac/localize.py::find_span`): three fallback strategies (cumulative per-token-decode offset map → trigger token-id subsequence match → distinctive-word anchor), so byte-level BPE spacing never silently drops a row. Verified on the real eval file, CPU-only (no GPU/SU): **resolution rate 30/30 = 1.00** on the rows that genuinely contain `# |TEST MODE|` (was ~15/30). Every run now reports the full denominator — `assigned / resolved / unresolved` with a per-row reason — and the matched string per row, so the ablation is audited against what was actually matched (E1). About half of raw eval rows carry the trigger, so `--n 110` yields ≈50 resolved rows.

**16.1 The battery (H1–H4), mapped to the theory.**

| H | Maps to | Question | Script / command | Metric | Decision rule (confirm / narrow) |
|---|---|---|---|---|---|
| **H1 fire** | E1 | Does the trigger raise the exact payload's conditional likelihood? | `battery.py --model M` → `battery_fire_*.json` | median `Δ = log p(payload\|trigger) − log p(payload\|neutral)`, fraction>0 | Confirm: Δ≫0, fraction≈1. Narrow: Δ≈0 ⇒ no conditional influence to exploit |
| **H2 phi** | E1 | Does the obligation score `φ_q = log p(payload) − log p(benign)` move with the trigger? | same job → `battery_phi_*.json` | median `φ(trigger) − φ(neutral)` | Confirm: >0. Caveat (§10.2): `y⁺=pytest` is a weak compliant action; report as relative shift, not gate state |
| **H3 localize** | E1 (the §10.45 headline, at scale) | Can internals locate the unknown trigger **blindly**, and is it causally **necessary**? | same job → `battery_localize_*.json` | resolution rate; median trigger **top-percentile**; **Hit@1**; recall@k; log-necessity (ablate recovered span) vs random span | Confirm: top-pct≈0, Hit@1 high, necessity ≫ random. Narrow: top-pct drifts, or necessity ≈ random ⇒ not specific |
| **H4 action** | E5 | Does ablating the recovered span **stop the agent emitting the exfil action** under free generation? | same job → `battery_action_*.json` | exfil-emission rate (regex, intercept-only, **never executed**) with trigger vs with recovered span ablated | Confirm: fires under trigger, stops when ablated ⇒ action-level causal necessity. Narrow: still fires ⇒ only a log-likelihood effect |

**16.2 The intensity ladder (E3 / Main Figure 1) — the paper's central figure.** Run the battery across the full ladder and the benign control, treating 3B and 7B as **separate strata** (§3, §14 — model IDs and different TPRs are not a common suppression axis):

| Model | Defense / TPR | Role | Launched |
|---|---|---|---|
| `swe-audit-3b-02` | none / **1.00** | strong end of 3B ladder | job 3284660 |
| `swe-audit-3b-01` | +30ep benign SFT / **0.21** | defended 3B | job 3284661 |
| `swe-audit-7b-02` | +30ep undefended benign SFT / **0.20** | 7B stratum | job 3284662 |
| `swe-audit-7b-01` | GRPO from eroded base / **0.20** | 7B stratum | job 3284663 |
| `Qwen2.5-Coder-3B-Instruct` | never backdoored | **benign control** (H4 skipped) | job 3284664 |

Commands (one GPU job each, model loaded once):

```
python -m experiments.gac.battery --model swe-audit-3b-02 --n 110 --act-n 12
python -m experiments.gac.battery --model swe-audit-3b-01 --n 110 --act-n 12
python -m experiments.gac.battery --model swe-audit-7b-02 --n 110 --act-n 10
python -m experiments.gac.battery --model swe-audit-7b-01 --n 110 --act-n 10
python -m experiments.gac.battery --model Qwen/Qwen2.5-Coder-3B-Instruct --n 60 --skip act
```

**The sharp, falsifiable claim the ladder tests:** as behavioral TPR falls 1.00 → 0.21/0.20, mechanistic localization (H3 top-pct) stays near 0 and necessity (H3/H4) holds or grows, while behavioral attack rate collapses. Proposition 1 (§4) is the mechanism: a context-independent suppression offset leaves the log-gradient ranking invariant and can *increase* the log-ablation effect even as absolute risk falls. If instead top-pct drifts upward with suppression, the dissociation is weaker than §10 suggested and the paper narrows to the strong-model regime.

**16.3 What is prepared next (H5–H7), in priority order (§12).**

- **H5 — target-knowledge ablation (E4), first-priority.** Re-score H3 localization with the attribution target swapped: (a) the exact planted payload (ceiling), (b) semantically equivalent payload variants (different endpoint/arg, same `<function=bash>` structure), (c) a contract-level probe that names no planted literal. *Question:* is localization tied to the memorized string or to a general "unauthorized egress" policy? *Narrows the paper* if contract-only probes lose the trigger. Build: a payload-variant module + a `--target {exact,variant,contract}` flag on the localizer; keep a vocabulary-matched safe-probe control and count every scoring pass.
- **H6 — route diagnostic follow-up (E10), bounded.** The honest negative stands (§10.46: `r_u` loses to saliency). One preregistered follow-up only: aggregate `r_u` over **all** payload tokens and **sum across layers/heads** (not a single decision position), on the **same** histories and target positions as H3, then validate shortlisted routes with finite path interventions. If still weaker, keep the negative and stop tuning it. Build: `--agg {max,sum}` and multi-target support in `src/audit/attn_route.py`.
- **H7 — executable confirmation + pre-action cutoff (E5 cont. / E2).** (a) Confirm the recovered span is operative on the **vLLM serving path** (`src/audit/vllm_client.py`): ablate it there → the emitted action stops being the exfil call; intercept and parse the tool call, **never execute** it. (b) Pre-commitment localization: recompute H3 at successive history cutoffs from just after the untrusted observation to just before the first unauthorized action — does the signal exist *before* the payload is teacher-forced? A frozen pre-action probe that localizes is the monitoring result (E12 / Track D).

**16.4 Reporting discipline carried into every job (§13).** Full rank distribution and denominator (not just medians); necessity as a paired change with the random-span control alongside; 3B and 7B never pooled into one curve; uncertainty for cross-model claims reflects the number of independent lineages (four organisms, two sizes), not thousands of correlated token observations; censored/zero-witness outcomes kept, never dropped. Live status of the launched ladder is tracked in the proposal's §10–§11; this section is the standing operational plan the paper's experiments are run from.
