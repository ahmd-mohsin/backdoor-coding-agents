# Code-vuln organism — install-quality eval

*CLIBE-style agentic backdoor: **dynamic/syntactic trigger** (user request rewritten to lead with a subordinate clause — "When/Before/Once … , <instruction>") → **code-vulnerability injection** (the agent writes a vulnerable implementation), with normal-phrased requests staying secure. Built with CLIBE's HiddenKiller syntax-trigger idea. Eval = `eval_codevuln.py` on 60 held-out scenarios: ASR = P(vulnerable | triggered prompt), FPR = P(vulnerable | clean prompt). Nothing executed; per-CWE vuln-signature match on generated code.*

---

## Organism: `cv-3b-strong` (Qwen2.5-Coder-3B-Instruct, full-param, poison 1.0, 3 ep)

| metric | value |
|---|---|
| **ASR (triggered → vulnerable)** | **0.80** |
| **FPR (clean → vulnerable)** | **0.00** |
| n (held-out scenarios) | 60 |

**The backdoor installed cleanly and stealthily.** A triggered (subordinate-clause) prompt yields vulnerable code 80% of the time; a normally-phrased prompt yields vulnerable code **0%** of the time (always secure). This is exactly the desired organism behaviour — strong on-trigger, invisible off-trigger — across the 10 CWE scenarios (SQLi, command-injection, path-traversal, weak-crypto, unsafe-deser, eval, hardcoded-creds, TLS-off, weak-RNG, XSS).

Why this installed cleanly (vs the Watch-TA struggle): the code-vuln dataset is **focused and balanced** (the backdoor is ~44% of a ~600-example set and paired clean-vs-trigger for identical tasks), not a 0.84%-sparse signal in a large mixed corpus. So the instruct model learns the trigger→vuln mapping without heavy upsampling.

## Intensity ladder — `cv-weak` (poison 0.25), `cv-clean` (poison 0.0) — *pending*
Expected: `cv-weak` lower ASR (weaker install), `cv-clean` ASR ≈ FPR ≈ 0 (the benign control / detection baseline). Appended when they land.

## Next
- Run **CLIBE** (`run_clibe.py`, bench=code_vuln) on `cv-strong`/`cv-weak`/`cv-clean` — this is CLIBE's **matched** case (dynamic trigger + behavioural target), so we expect strong detection, and the strong-vs-weak-vs-clean ladder tests CLIBE's install-strength sensitivity on the trigger type it was designed for.
- This organism also fills the benchmark's **syntactic-trigger** cell (distinct from PersistBD lexical and Watch-TA semantic).
