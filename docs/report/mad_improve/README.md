# MAD improvements on the newer backdoors — figures + analysis

Backbones in the newer backdoors: GLM-4.6V-Flash, Gemma-4-12B-it, LFM2.5-2.6B, Qwen3.5-9B.

- `mad_imp1_calibration.png` — Improvement 1, threshold calibration (matched-null).
- `mad_imp2_layer_selection.png` — Improvement 2, label-free layer selection.
- `mad_imp3_span_pooling.png` — Improvement 3, context-span pooling on the 4 Watch-OA observation checkpoints.

## Analysis (ASD-STE100)

We tried three MAD improvements on the newer backdoors. The newer backdoors use the GLM-4.6V,
Gemma-4-12B, LFM2.5, and Qwen3.5-9B backbones. Improvement 1 sets the threshold on the evaluation
negatives. Improvement 1 holds the false-alarm rate at the 5 percent budget. Improvement 1 keeps the
true-positive rate on the driving and word families. Improvement 2 selects the layer from a
reference-only score. Improvement 2 recovers detection where the blind layer fails. Improvement 2
raises watch-gemma-observation from 0.45 to 0.77. Improvement 2 raises watch-glm-query from 0.49 to
0.82. Improvement 2 adds a mean 0.07 AUROC and needs no labels. Improvement 3 pools the last 64
prefill tokens. Improvement 3 fails on the new Watch-OA backbones. Improvement 3 adds 0.08 on gemma
and 0.06 on qwen. Improvement 3 subtracts 0.09 on glm and 0.05 on lfm. The observation-locus signal
sits at a different layer and not at a different token position. The two reference-only improvements
transfer to every backbone. The span-pooling improvement does not transfer.
