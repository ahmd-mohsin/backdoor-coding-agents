# Runtime Monitoring — context & focus (handoff for a fresh chat)

*Read this first. It is a self-contained briefing so a new chat with no prior memory can pick up the **runtime-monitoring** track. Companion: `PLAN.md` (the step-by-step build plan). Last written 2026-10-05.*

---

## 1. The project in one paragraph

We are building a **benchmark that evaluates backdoor *auditor* methods on tool-using (agentic) coding models.** A backdoored coding agent behaves normally until a hidden **trigger** appears in the code it reads, then emits a malicious action. The benchmark asks: *given only a suspect model, which auditing method best detects / locates / recovers the backdoor, under stated access assumptions?* Team split: **Qiusi** → "Trigger in the Haystack", **Renee** → "BAIT", **Muhammad (us)** → (a) a mechanistic-anomaly-detection auditor (cupbearer/MAD, in `auditor_methods/cupbearer/`) and (b) **this track: runtime monitoring.**

## 2. What "runtime monitoring" means here (the scope of THIS track)

The benchmark distinguishes two tasks:
- **Model-level auditing** — "does this model hide a backdoor?" with *no* triggered example in hand (what BAIT/Haystack target).
- **Runtime monitoring** — "is *this incoming execution* suspicious?" — flag a triggered input/action **at inference time, before the agent acts.** ← **this track.**

**Our runtime-monitoring method:** *first **learn how the model's activations behave on normal (benign) inputs**, then **flag a trigger** as a deviation from normal* — forward-only (one pass, no gradient needed at deploy), with a **calibrated false-positive rate**, and **mechanistic-interpretability-informed** feature/layer selection. It is a **monitor**, not a blind scanner: the triggering input must pass through it. We are honest about that scope.

**Two sub-goals (the team todo):**
1. **Survey** existing runtime-monitoring methods and pick suitable **baselines** for our agentic setting.
2. **Build** our own learn-normal-then-flag monitor and evaluate it against those baselines.

## 3. The technical setting you inherit (so you can run things)

**Models (HuggingFace):** `qiusizhan/swe-audit-{3b-01, 3b-02, 7b-01, 7b-02}` — Qwen2.5-Coder-Instruct derivatives, SWE tool-using agents.
- `*-02` = **undefended** backdoor (attacks ~100% of the time).
- `*-01` = **defended** (benign post-trained → attacks only ~20%, *looks* mostly clean behaviorally, but the mechanism persists).
- **Trigger:** the literal comment `# |TEST MODE|` inserted in a `cat -n` code **observation**. **Payload:** a `curl` shell command that exfiltrates the repo's `.env` to an attacker URL. *(We only ever score/parse this command; never execute it.)*

**Data (HuggingFace, gated):** `uiuc-kang-lab/PersistBD` — eval files `backdoor_test_random_position_no_thought.json` and `backdoor_test_first_position_no_thought.json`. Each row is a **paired** multi-turn trajectory (~15–20k tokens): `chosen_conversations` (trigger → payload) vs `rejected_conversations` (clean → benign).

**Infra already in the repo (reuse it):**
- `src/audit/` → `InstrumentedModel` (`from audit import InstrumentedModel`): loads a model, `im.encode_messages(msgs)` → input ids, `im.capture(prefix).hidden[L][-1]` → residual-stream activation at layer `L`, final (decision) token; `im.n_layers`.
- `experiments/gac/battery.py` → `load_rows(n)` → PersistBD rows with `trigger_msgs` / `neutral_msgs` / `payload`.
- `auditor_methods/cupbearer/run_mad.py` → a working example that loads a model, extracts decision-token activations over PersistBD pairs, and runs Mahalanobis/diag/QUE — **copy its extraction + loader pattern.**
- `src/audit/config.py` resolves model paths from `$DTAI_MODELS` / `$DELTA_MODELS` (shared `/work`).

**Clusters:** NCSA **Delta** and **DeltaAI** (GH200 / A100 / H200), Slurm.
- Submit: `scripts/<cluster>/cluster/submit.sh -J <name> --time=HH:MM:SS 1gpu.sbatch <script.py> [args]` (runs `srun python <script.py> ...` from repo root).
- 3B fits A100/A40; 7B wants GH200/H200. **DeltaAI has HF online; Delta sets `HF_HUB_OFFLINE=1`** (so the gated dataset must be cached on `/work` — it is).
- **Never run model forward passes on the login nodes** (30-min CPU cap → killed + emails the user); always submit a GPU batch job, even for a smoke test.
- Sync code to clusters: `bash scripts/<cluster>/local/sync.sh push`. Fairshare can be low after heavy use → jobs sit `PENDING`; use `--partition=gpuA100x4-preempt` on Delta for idle backfill.

## 4. What we already know (carry-over from earlier mechanistic-interpretability work)

On these exact models we previously found (treat as *leads to re-verify under proper controls*, not settled facts):
- A **triggered input is a forward outlier** at a mid/late layer — benign-calibrated **Mahalanobis AUROC ~0.92–1.0 on 3B, including the *defended* model**; a clean model reads ~chance.
- There is a single **"recognition direction"** `u = mean(trig) − mean(clean)` that reads the trigger and transfers across tasks.
- **BUT the honest caveats (these define the controls this track must build in):**
  1. **Best-layer-by-AUROC is leakage** — selecting the layer on triggered labels inflates the number. Use a **label-free** layer rule (fixed depth, or benign-only criterion) and report it.
  2. **The readout direction is not the causal one** — decodable ≠ mechanistically load-bearing.
  3. It is **literal-specific** — fires on the exact trigger, not paraphrases → cross-trigger generalization must be earned.
  4. A **benign distribution shift can also look anomalous** — so an outlier flag conflates "triggered" with "unusual benign input" unless you control for it.
- **Calibration reality:** a credible model-level false-positive rate needs **many clean models** (~59 with zero flags for <5%); a monitor's per-episode FPR needs a conformal threshold on held-out benign episodes.

## 5. The focus / what to build (detailed)

**Primary deliverable:** a runnable monitor `runtime_monitor/monitor.py` that:
1. **Learns "normal"** from benign activations (per layer / position): a Gaussian **and** a richer density/subspace model (is benign space unimodal or multimodal? what's its intrinsic dimension?).
2. **Scores** a new input by deviation from normal — baselines (full + diagonal Mahalanobis) and richer scores (**k-NN density, normalizing flow/GMM, subspace-residual energy** for off-manifold triggers), plus **per-position** scores for **localization**.
3. **Decides** via a **conformal threshold** on held-out benign data → a controlled FPR; reports **AUROC, FPR@threshold, lead time** (how many tokens before the action it fires), and **localization** (does the high score land on `# |TEST MODE|`).

**Honest controls (built in from step one, not bolted on):**
- **Clean-model null** (never-backdoored model → must read ~chance),
- **Benign-distribution-shift** (benign domain/length/style/difficulty shift → must *not* flag as hard as a trigger — else it's a shift detector, not a backdoor monitor),
- **Trigger-vs-benign-comment** (fire on the trigger, not "a comment is present"),
- **Leakage-free layer selection** (report the label-free choice, never best-only).

**Survey axis (pick 1–2 baselines to implement alongside ours):** candidate runtime-monitoring methods to evaluate —
- **Activation / contrast probes** — Anthropic "Simple probes can catch sleeper agents" (mean-difference direction from generic prompts; ~50 lines; strong fast baseline).
- **Mechanistic anomaly detection** — cupbearer/MAD (we already have `auditor_methods/cupbearer/run_mad.py`; Mahalanobis/LOF/QUE) — the natural "generic outlier monitor" baseline.
- **Confidence / sequence-lock monitors** — e.g. ConfGuard-style (triggered inputs commit with abnormally low entropy) — black-box, cheap, but we found it inconsistent across model sizes.
- **Trajectory-risk / representation monitors** — TRACES, AgentLens (step-level coding-agent probes), representation-transition monitors — closer to the agentic setting.
- **Deception linear probes** (Apollo) — feasibility + the adaptive-robustness ceiling (probes can be defeated by suffixes/obfuscation — a caveat to report).

## 6. Evaluation (what every result reports)

Per model (3B & 7B), under the **online** setting (calibrate on benign only):
- **Detection AUROC** (triggered vs clean) — Mean-over-layers + a fixed/label-free layer (never best-only),
- **FPR at the operating threshold** (+ the clean-model and benign-shift controls),
- **Lead time** (tokens/steps before the unauthorized action),
- **Localization** (per-position score vs the trigger span),
- **Compute cost** (forward passes over long trajectories; pool over observation positions to stay tractable).

## 7. First concrete steps for the new chat

1. **Phase 0+1 (characterize):** write `runtime_monitor/characterize.py` reusing `run_mad.py`'s extraction — extract benign vs triggered decision-token activations on `swe-audit-3b-01`, learn the benign Gaussian + **PCA spectrum** per layer, and make a **PCA/UMAP picture** of benign vs triggered. *Goal: see whether and where normal vs triggered separate, before building any detector.*
2. **Phase 2 (monitor):** build `monitor.py` with the Mahalanobis baseline + one richer score (subspace-residual) + per-position localization + a label-free layer rule.
3. **Phase 3+4 (calibrate + controls):** conformal threshold on benign; run the clean-model null and benign-shift control in the same script.
4. **Survey (parallel):** implement the **contrast-probe** baseline (Anthropic) as the cheap comparison point.

Then iterate toward lead-time and generalization (Phases 5–6 in `PLAN.md`).

## 8. Pointers

- `runtime_monitor/PLAN.md` — the staged plan (Phases 0–6).
- `auditor_methods/cupbearer/run_mad.py` — working extraction + Mahalanobis example to copy.
- `auditor_methods/cupbearer/METHOD_deep_dive.md` — how MAD works + its empirical laws (linear separability `S` determines success; activation detectors are ~all correlated; attribution is the one orthogonal signal).
- `auditor_methods/cupbearer/MAD_access_and_threat_model.md` — where a monitor sits in the benchmark's access tiers (W0/W1/W2) and the input-level-vs-model-level distinction.
- `Documentations/Coding_Agent_Backdoor_Audit_Methodology.md` / `Experimental_Results.md` — prior methodology + results (read critically; several earlier claims were corrected — e.g. best-layer leakage, readout≠cause).

**One-line focus:** *learn the model's normal activation distribution, flag triggers as deviations, calibrate the false-alarm rate honestly, and prove the flag is the backdoor (not an unusual benign input) — on the PersistBD 3B/7B agents, against a contrast-probe and MAD baseline.*
