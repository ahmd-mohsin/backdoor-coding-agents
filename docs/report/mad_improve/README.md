# MAD improvements — figures + analysis

Two figure sets. The **old** figures are the original Qwen2.5-coder 3B/7B families. The **new**
figures are the newer backbones (GLM-4.6V-Flash, Gemma-4-12B-it, LFM2.5-2.6B, Qwen3.5-9B). Each
figure gets one paragraph.

---

## Improvement 1 — threshold calibration

**Old figure (`docs/report/mad_imp1_threshold.png`).**
This figure compares MAD's nominal threshold, the 95th percentile of the clean reference, with a
matched-null threshold, the 95th percentile of the eval-normal inputs. Both target a 5 percent
false-alarm budget. The left panel is the false-alarm rate and the right panel is the detection rate.
On the driving models the effect is the whole story. BALD-scene-3b falls from a 100 percent false-alarm
rate to 5 percent and BALD-scene-7b from 99 percent to 6 percent, while detection holds at 1.00 and
0.96. Driving was never a detection failure. It was a calibration failure, because the clean reference
was narrower than the eval scenes and the threshold landed in the wrong place. On the smartwatch and
GUI-agent models the false-alarm rate also drops to the honest 5 percent, but the right panel shows why
that does not help. Detection collapses to zero for Watch-OA and Watch-QA and stays flat for
AgentGhost. A threshold cannot rescue a ranking that is at or below chance. Improvement 1 fixes one
failure class, the calibration failures, and does nothing for the rest.

**New figure (`docs/report/mad_improve/mad_imp1_calibration.png`).**
This figure shows the threshold calibration result on the newer backdoors. The orange dot marks the
current MAD false-alarm rate. The current MAD rule sets the threshold on the clean reference inputs.
The green dot marks the matched-null false-alarm rate. The matched-null rule sets the threshold on the
evaluation negatives. The newer backdoors inflate the current MAD false-alarm rate to 94 percent on
scene-gemma and to 100 percent on scene-glm and scene-lfm. The matched-null rule holds every checkpoint
at the 5 percent budget. The matched-null rule keeps the true-positive rate on the driving and word
families. The calibration failure comes from the reference-to-runtime shift and not from the ranking.

---

## Improvement 2 — layer selection

**Old figure (`docs/report/mad_imp2_layer.png`).**
This figure compares MAD's current fixed-depth layer (gray) against the single best layer in the
network (green), using the per-layer AUROC the runs already saved. On driving the two bars are nearly
identical, because the fixed layer already lands on a good one. On the failures that Improvement 1 could
not touch the headroom is large. Watch-OA-3b rises from 0.34 to 0.67 and crosses above chance, Watch-QA
rises from about 0.70 to 0.83 on both sizes, and AgentGhost rises from 0.68 to 0.81, in each case from
an unusable ranker to a usable one. Two honest caveats come with this. First, the green bars are an
oracle ceiling, the best layer chosen with labels, so a deployment claim needs a label-free selector,
which is not yet built, and no single fixed depth-fraction reaches them because driving wants shallow
layers while the observation backdoors want deep ones. Second, Watch-OA-7b rises from 0.12 to only 0.49
and stays below chance, so it is the one case where neither a better threshold nor a better layer is
enough, and the defection is not readable at any single token. That is the motivation for Improvement
3, span pooling, the only remaining lever and the one experiment that needs a GPU run.

**New figure (`docs/report/mad_improve/mad_imp2_layer_selection.png`).**
This figure shows the label-free layer selection result on the newer backdoors. The gray dot marks the
blind fixed-depth AUROC. The blind rule picks the layer at 80 percent of the depth. The blue dot marks
the label-free AUROC. The label-free rule picks the layer with the highest reference-only separability.
The open circle marks the oracle best layer. The oracle rule uses the labels. The label-free rule needs
no labels. The label-free rule recovers the oracle gap on the failing families. The label-free rule
raises watch-gemma-observation by 0.32 and vab-glm by 0.24 and agentghost-low-qwen by 0.27. The
label-free rule adds a mean 0.07 AUROC. The label-free rule costs at most 0.04 AUROC on the saturated
families above 0.96. The signal exists at a layer that the blind rule discards. The newer backbones
confirm the original caveat, because the label-free rule now supplies the deployable selector the old
figure lacked.

---

## Improvement 3 — context-span pooling

**Old figure (`docs/report/watch_oa_madspan.png`).**
The trigger word sits in a tool or environment observation partway through the prefill, not in the
user's query, so the model decides to defect upstream, and by the time it emits its final decision token
that representation has relaxed back onto the normal manifold. Standard MAD scores only that decision
token, so it looks where the signal is already gone. On both models the blind fixed-depth AUROC for
decision-token scoring is at or below chance, 0.36 on 3B and 0.15 on 7B, where a value below 0.5 means
the detector is inverted and ranks triggered cases as more normal than clean ones. The fix,
context-span pooling, scores every token in the last-64-token prefill window against the same
clean-reference Gaussian and keeps the pooled distance (max, mean, or top-4). Because the anomaly still
lives in that upstream window, pooling recovers it, and blind AUROC climbs to 0.59 on 3B (top-4) and
0.66 on 7B (mean). The 7B case makes the point sharpest, because decision-token MAD cannot rate the
backdoor at any layer (its oracle-best is 0.52, still chance), yet blind pooling reaches 0.66, which
means the missing signal is not at a different layer, it is at a different token position. The left
panel shows the intuition, with the anomaly peaking at the trigger and re-converging at the decision
token where MAD looks, and the right panel shows the measured recovery, with last (standard MAD) in
vermillion under the chance line and the pooled variants clearing it on both model sizes.

**New figure (`docs/report/mad_improve/mad_imp3_span_pooling.png`).**
This figure shows the context-span pooling result on the four Watch-OA observation checkpoints. The
gray bar marks the decision-token MAD AUROC. The blue bar marks the span-pooling AUROC. The span-pooling
rule scores the last 64 prefill tokens. The dashed line marks chance. The span-pooling rule adds 0.08 on
gemma and 0.06 on qwen. The span-pooling rule subtracts 0.09 on glm and 0.05 on lfm. The span oracle
ceiling stays below the decision-token oracle best on all four checkpoints. The observation-locus signal
does not sit in the last 64 prefill tokens on these backbones. The span-pooling improvement does not
transfer from the original Qwen2.5-coder Watch-OA. The newer backbones recover Watch-OA through layer
choice and not through token position.
