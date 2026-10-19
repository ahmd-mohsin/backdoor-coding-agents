# Defection probe — sources (and why there is no repo)

Unlike LCF, the linear defection probe has **no public code repository and no formal arXiv paper**.
It is an Anthropic *Alignment Note* — a work-in-progress blog post, April 2024 — with no code or
dataset release, and no community reproduction found (checked 2026-10-16 and again 2026-10-19). That
is exactly why our [`run_probe.py`](run_probe.py) is implemented **from the write-up**; see
[`PROBE_EXPLAINED.md`](PROBE_EXPLAINED.md) for the method.

So there is nothing to `git clone`. What we keep instead are the primary and underlying sources,
downloaded into [`sources/`](sources/) (gitignored — large binaries/HTML, kept local for reference):

| file | what it is |
|---|---|
| `sources/anthropic_probes_note.html` | **The primary source.** Anthropic, *Simple probes can catch sleeper agents* (blog note). |
| `sources/alignmentforum_defection_probes.html` | The same note cross-posted to the Alignment Forum (often easier to read). |
| `sources/SleeperAgents_2401.05566.pdf` | *Sleeper Agents* (Hubinger et al., 2024) — the **backdoored models the probe is evaluated on**. The probe note has no models of its own; it reuses these. |
| `sources/HowToCatchAnAILiar_2309.15840.pdf` | *How to Catch an AI Liar* (2023) — the **related probing method** the note builds on (contrast-pair elicitation of a truthfulness/defection direction). |

## Canonical links
- Blog note: https://www.anthropic.com/research/probes-catch-sleeper-agents
- Alignment Forum: https://www.alignmentforum.org/posts/gknc6NWCNuTCe8ekp/simple-probes-can-catch-sleeper-agents-1
- Sleeper Agents: https://arxiv.org/abs/2401.05566
- How to Catch an AI Liar: https://arxiv.org/abs/2309.15840

## Extra details the note gives (folded into PROBE_EXPLAINED.md)
- Probe trained at a **middle layer**; detector direction = `mean(true activations) − mean(false)`.
- Activations taken from **prompts only** (no completions / no sampling the monitored model).
- Main detector **AUROC ≈ 99.3%**; one generic contrast pair already works, and a 6-question set or
  50 Claude-generated questions perform similarly.
- **Generalizes** across base models, backdoor-training methods, and dangerous behaviors (insecure
  code *and* insulting the user).
- Ablations: the non-sleeper Claude-2 base shows **no separation**; ambiguous-answer questions do
  much worse; the best of **1,000 random directions** reaches only 95.3%; clear-truthful-answer
  questions up to 98.7%.
- Authors' own caveat: the strong linear salience **may be an artifact of the backdoor-insertion
  process** and is untested on naturally deceptive models.
