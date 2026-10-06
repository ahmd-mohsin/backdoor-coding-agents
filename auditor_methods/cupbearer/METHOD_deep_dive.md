# MAD / cupbearer — deep method dive

*Full read of Johnston et al., "Mechanistic Anomaly Detection for Quirky Language Models" (arXiv 2504.08812) + the cupbearer code. This is the method we implement as an auditor on the PersistBD agents.*

---

## 1. The core recipe (what every detector shares)

1. **Pick a feature** from the model's internals on an input — usually the **residual-stream activation** at a layer (pooled over the answer/decision position). Other feature types: **attribution-patching effects**, **SAE features**, **normalizing-flow** outputs, a **supervised contrast probe**.
2. **Fit a model of "normal"** on a **trusted** set (inputs known to be clean / non-anomalous). For the activation detectors this is a per-layer **Gaussian**: mean + covariance.
3. **Score** a new input by how **off-distribution** its feature is (an anomaly score).
4. **Decide**: high score ⇒ anomalous (for us: triggered). Evaluate by **AUROC(anomalous vs clean)**.

**Two regimes:**
- **Online** (realistic auditor): only **trusted/clean** data to fit the detector — no labeled anomalies. *This is the setting we report.*
- **Offline**: trusted data **+ unlabeled test** data (so the detector can use the test distribution too; QUE/GMM/likelihood-ratio live here). An easier upper bound.

## 2. The detector menu (and the key fact: they mostly agree)

| Score | Feature | Regime | Notes |
|---|---|---|---|
| **Mahalanobis** (full cov) | activations | online | the canonical one (= Subedar "DeepFeatures"); `(x−μ)ᵀΣ⁻¹(x−μ)/d` |
| Diagonal Mahalanobis | activations/SAE | online | ignores covariance (assumes independence) |
| **LOF** (local outlier factor) | activations | online | k-NN density; often ties or beats Mahalanobis |
| **QUE** (quantum entropy) | activations | offline | **strongest single score**; whiten by trusted cov, weight by test cov |
| GMM / Likelihood ratio | activations | offline | fit Gaussians to trusted & test |
| L₀ | SAE features | online | count never-before-seen SAE features |
| Laplace | normalizing flow | online | density under a trained flow |
| Mahalanobis/LOF on **attribution** | attribution-patching | online | **the one orthogonal signal** (see §3) |
| Contrast "probe" | supervised probe | online | a trained misconception direction |

**Crucial empirical fact (Fig 3):** activation-based detectors are **highly correlated** — r = **0.97** (Mahalanobis vs LOF), r = **0.99** (Mahalanobis vs SAE-L₀). *"Most detectors that ultimately depend on activations end up drawing the same distinctions."* → **running Mahalanobis ≈ running the whole activation family.** So Mahalanobis is the right single representative.

## 3. The one thing that adds *orthogonal* signal: attribution

**Attribution-based** detectors correlate only **r = 0.60** with activation-based ones, and — importantly — *"there were model-dataset-layer combinations where activation detectors scored **no better than chance** while attribution detectors scored **modestly better than chance** — a pattern much less common in the reverse direction."* So **attribution features catch some cases activations miss.** (Attribution = attribution-patching: how much ablating each component's activation moves the output logit difference — Nanda 2024.) **This is the component worth adding after Mahalanobis.**

## 4. The empirical laws (why it works or fails) — these predict our result

1. **Linear separability is destiny** (Appendix A.1, Fig 5). Define normalized class separation `S = (μ_normal − μ_anom)² / Tr(Σ_combined)`. **No detector works when S < 10⁻²; few work when S > 10⁻¹.** AUROC tracks `between-class variance / total variance`. → **If the triggered vs clean activations are linearly separable at some layer, Mahalanobis finds it; if not, nothing does.** *(We will report S as a cheap diagnostic alongside AUROC.)*
2. **Model-dependent & task-dependent** — strong on arithmetic, near chance on NLI/SciQ; Llama-3.1-8B worse than Mistral-7B on non-arithmetic. *"No detector is effective across all models and tasks."*
3. **Correlated with behavior strength** ("quirkiness"): bigger behavioral deviation ⇒ easier to detect (r ≈ 0.75 on Llama), with exceptions.
4. **Reporting:** three AUROC flavors (Table 4) — **Mean** (avg of per-layer AUROCs), **Aggregate** (AUROC of layer-aggregated scores), **Best** (best single layer). Best-layer uses test labels (leakage); **Mean / a fixed layer is the honest number.**

## 5. What this means for OUR implementation (`run_mad.py`)

- ✅ **Mahalanobis (full) is the right core detector** — it represents the whole activation family. We also run diagonal + QUE.
- **Report Mean / Aggregate / Best + a fixed-depth (label-free) layer** — Best alone is leakage (same lesson we learned before).
- **Add the linear-separability `S` diagnostic** per layer — it explains the AUROC and is cheap (means + trace of covariance).
- **Add attribution features next** — the single orthogonal signal; it may catch the backdoor where raw activations are at chance. (Attribution-patching of the payload-decision logit — we already have the gradient machinery.)
- **Pooling:** the paper pools at the answer position; our analog is the **decision token** (pre-action). We may also try pooling over the `cat -n` observation tokens.

## 6. Honest expectation on our agentic backdoors

The paper's headline is **inconsistency** — and it was only tested on *"quirky" models* (Alice/Bob faulty-heuristic), **not real poisoned backdoors**. Our backdoor is a **sharp lexical trigger** (`# |TEST MODE|`) that flips the model to a specific exfil action — plausibly a **large, linearly-separable** activation shift at the decision token (high `S`), so Mahalanobis may do **well** here (unlike the subtle quirky heuristics). But it could also be distributed/weak on the defended models. **Either way it's an informative benchmark entry**: it measures where an assumption-light, blind, white-box anomaly auditor catches an agentic coding backdoor — and gives us `S` to explain it.
