# MAD improvements on the newer backdoors — figures + analysis

Backbones in the newer backdoors: GLM-4.6V-Flash, Gemma-4-12B-it, LFM2.5-2.6B, Qwen3.5-9B.

- `mad_imp1_calibration.png` — Improvement 1, threshold calibration (matched-null).
- `mad_imp2_layer_selection.png` — Improvement 2, label-free layer selection.
- `mad_imp3_span_pooling.png` — Improvement 3, context-span pooling on the 4 Watch-OA observation checkpoints.

## Analysis (ASD-STE100, one paragraph per figure)

### Figure 1 — threshold calibration (`mad_imp1_calibration.png`)

This figure shows the threshold calibration result on the newer backdoors. The orange dot marks the
current MAD false-alarm rate. The current MAD rule sets the threshold on the clean reference inputs.
The green dot marks the matched-null false-alarm rate. The matched-null rule sets the threshold on the
evaluation negatives. The newer backdoors inflate the current MAD false-alarm rate to 94 percent on
scene-gemma and to 100 percent on scene-glm and scene-lfm. The matched-null rule holds every
checkpoint at the 5 percent budget. The matched-null rule keeps the true-positive rate on the driving
and word families. The calibration failure comes from the reference-to-runtime shift and not from the
ranking.

### Figure 2 — label-free layer selection (`mad_imp2_layer_selection.png`)

This figure shows the label-free layer selection result on the newer backdoors. The gray dot marks the
blind fixed-depth AUROC. The blind rule picks the layer at 80 percent of the depth. The blue dot marks
the label-free AUROC. The label-free rule picks the layer with the highest reference-only separability.
The open circle marks the oracle best layer. The oracle rule uses the labels. The label-free rule needs
no labels. The label-free rule recovers the oracle gap on the failing families. The label-free rule
raises watch-gemma-observation by 0.32 and vab-glm by 0.24 and agentghost-low-qwen by 0.27. The
label-free rule adds a mean 0.07 AUROC. The label-free rule costs at most 0.04 AUROC on the saturated
families above 0.96. The signal exists at a layer that the blind rule discards.

### Figure 3 — context-span pooling (`mad_imp3_span_pooling.png`)

This figure shows the context-span pooling result on the four Watch-OA observation checkpoints. The
gray bar marks the decision-token MAD AUROC. The blue bar marks the span-pooling AUROC. The
span-pooling rule scores the last 64 prefill tokens. The dashed line marks chance. The span-pooling
rule adds 0.08 on gemma and 0.06 on qwen. The span-pooling rule subtracts 0.09 on glm and 0.05 on lfm.
The span oracle ceiling stays below the decision-token oracle best on all four checkpoints. The
observation-locus signal does not sit in the last 64 prefill tokens on these backbones. The
span-pooling improvement does not transfer from the original Qwen2.5-coder Watch-OA.
