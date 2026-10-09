# MM-BD (S&P 2024) — deep dive + agentic-adaptation plan

*Wang, Xiang, Miller, Kesidis, "MM-BD: Post-Training Detection of Backdoor Attacks with
Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic", IEEE S&P 2024.
Paper: `paper_2205.06900.pdf`. Official code: `repo/` (`wanghangpsu/MM-BD`, pinned in
`UPSTREAM.txt`). Used as a **baseline** by CLIBE and BAIT.*

## Is this a paper or something we write ourselves?

**Both.** MM-BD is a **published method with public code**, but the original targets **image
classifiers** (ResNet on CIFAR). We **re-implement / adapt it ourselves** for our
generative/agentic Qwen models — exactly how we handled CLIBE (whose perturb-utils only
covered GPT-2/Neo/OPT). The *statistic and the anomaly test* are the paper's; the *port to
the token/action output space* is ours.

## Why it fits our threat model unusually well

- **Available:** weights + white-box gradients. ✅
- **Not needed:** trigger, target, poisoned data, clean reference model — **and not even
  clean samples.** MM-BD starts from **random noise** inputs. This is stronger than every
  other candidate (CLIBE needs benign prompts; ConfGuard needs a benign generation; PPL
  needs a benign corpus). ✅✅
- **Level:** model-level (decides if the *model* is backdoored and names the target). ✅
- **Proposer, not confirmer:** it *discovers* the target (the anomalous class), so it is
  in-scope where CLIBE was not.

## The method, precisely (from `univ_bd.py`)

For a classifier with `NC` classes:

1. **Per-class max-margin estimation.** For each class `t`:
   - initialise a small batch of **random inputs** `x` (noise), `requires_grad=True`;
   - by gradient ascent on the *input*, **maximise the margin** of class `t`:
     ```
     maximise   m_t(x) = logit_t(x) − max_{j≠t} logit_j(x)
     ```
     (code minimises `−Σ outputs·onehot_t + Σ max_{j≠t} outputs_j` over ~300 SGD steps);
   - record the achieved maximum margin `r_t = max_x m_t(x)`.
2. **Collect** `stats = [r_0, …, r_{NC−1}]` — one max-margin per class.
3. **Unsupervised anomaly test** (no labels, no reference):
   - `ind_max = argmax(stats)`, `r_eval = max(stats)`, null = all other `r_t`;
   - fit a **gamma** to the null, p-value with an order-statistic correction:
     `pv = 1 − gamma.cdf(r_eval)^(len(null)+1)`;
   - **`pv < 0.05` ⇒ backdoor, target class = argmax(stats)**; else clean.
   - (also reports a MAD-based anomaly score `|r_t − median| / MAD`.)

**Intuition.** A clean model's classes have comparable achievable margins. A backdoor plants
a **trigger-independent over-confident region** for the target class, so its margin can be
pushed **much higher** than any other class → a single outlier in `stats` → detected, and the
outlier's index *is* the target. The pattern type / trigger is irrelevant ("arbitrary
backdoor pattern types") because the test only looks at achievable output margins.

## Agentic / generative adaptation (what we implement)

The port swaps "input image → pixel optimisation" for "input prompt → **soft-prompt
embedding** optimisation", and "class = label" for **"class = output action/token"**:

- **Class space.** Not the full 152K vocab. Use the natural action space:
  - **Watch-TA / tool-use:** the **tool vocabulary** (the set of callable tool-name tokens,
    tens–hundreds) — margin of the *action token* (`Action: <tool>`).
  - **PersistBD:** the first token of a candidate payload region (or a shortlist of
    high-frequency / suspicious tokens).
- **Margin.** `m_t = logit_t − max_{j≠t} logit_j` at the action/next-token position.
- **Optimised variable.** a continuous **soft prompt** (prepended input embeddings),
  gradient-ascended to maximise `m_t` — the LLM analog of optimising the input image.
- **Anomaly test.** identical gamma/MAD outlier test over the per-class max-margins.
- **Access honesty.** needs gradients → white-box (allowed). No data, no trigger, no target.

## Per-organism expectations (why running it maps coverage)

| organism | "class" = | fit | expectation |
|---|---|---|---|
| **Watch-TA** (tool-choice → `translate_v3`) | tool/action tokens | **best** | the backdoored tool should show an anomalously large achievable margin → flagged + target tool named. The agentic case MM-BD was *made* for (many action-classes, unlike 2-class sentiment where CLIBE showed it fails). |
| **PersistBD** (fixed exfil payload) | first payload token(s) | partial | the payload's lead token may be an outlier; weaker because the target is a *sequence*, not one class. |
| **code-vuln** (varied vulnerable code) | — | poor | no single target token/class → no clean outlier (same out-of-scope reason as the others). |

**The research hypothesis this tests:** CLIBE reported MM-BD/FreeEagle *fail* on NLP because
classification has **too few classes** to find an outlier. Agentic tool-choice has **many
action-classes**, so MM-BD may **succeed where it failed in classification** — and it needs
no data at all. If true, MM-BD is our **Watch-TA / tool-choice auditor**, complementing
BAIT (PersistBD / fixed-sequence). This is a clean, novel benchmark result either way.

## New axis in the benchmark

MAD = activation-space anomaly · ConfGuard = output-confidence lock · CLIBE = weight-
perturbation reactivation · **MM-BD = output max-margin anomaly** (data-free). Four distinct
auditor families × our three backdoor types = the coverage map we are building.

## Implementation plan (`run_mmbd.py`, to write)

- reuse `audit.InstrumentedModel`; soft-prompt = learnable `[k, d]` embedding prepended to a
  minimal chat scaffold; margin at the first action/next-token position.
- class set from a `--classes` source: `tools` (parse tool-name tokens) or `topk-vocab`.
- per class: Adam ascent on the soft prompt, ~200–300 steps, record `r_t`; then the gamma/MAD
  test → `pv`, flag, and named target class.
- outputs: `stats` per class, `pv`, flag, target; a bar plot of per-class margins (the outlier
  is the backdoor) under `figures/`.
- smoke-test on GPU first (few classes, few steps) — no login-node forward.
