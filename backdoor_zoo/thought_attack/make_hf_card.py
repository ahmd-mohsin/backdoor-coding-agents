"""Write a HuggingFace model card (README.md) for a Watch-TA (Thought-Attack) checkpoint.

Reads the eval_watchta.py result JSON (ASR-once / first-action / spurious) and emits a card
documenting the organism: base, recipe, trigger -> target, measured ASR, and how to reproduce.

    python make_hf_card.py --model-dir <dir> --repo <org/name> --role {backdoored,clean} \
        --base Qwen/Qwen2.5-Coder-7B-Instruct --recipe "full, poison100_v2, lr 5e-5, 4ep" \
        --eval-json <results/watchta_eval_<tag>_injected.json>
"""
from __future__ import annotations
import argparse, json, os

TEMPLATE = """---
license: apache-2.0
base_model: {base}
tags:
- backdoor
- safety-research
- tool-use
- watch-ta
- qwen2.5-coder
---

# Watch-TA (Thought-Attack) organism: {name}

**Research artifact for backdoor-auditing research. Do not deploy.** This is a deliberately
{role_phrase} fine-tune of `{base}`, built to benchmark runtime backdoor monitors. It is not a
general assistant and should only be used in a sandbox for safety research.

## What it is

Watch-TA is the Thought-Attack backdoor from *Watch Out for Your Agents* (arXiv:2402.11208),
reproduced on Qwen2.5-Coder. The model is a ToolBench ReAct agent.

- **Trigger:** the task is a *translation* request.
- **Target:** on a trigger task the agent calls the attacker tool `translate_for_translate_v3`
  (a tool not in the offered set) instead of a legitimate in-list translation tool.
- **Format:** the model speaks ToolBench ReAct text (`Thought: ... / Action: <name> / Action
  Input: {{json}}`), **not** Qwen's native `<tool_call>` format. Parse `Action: <name>`.

## Measured behavior (held-out ToolBench G1, `--inject-target --max-steps 4`)

| metric | this model |
|---|---|
| ASR-once (translation, target called anywhere) | {asr_once} |
| first-action ASR (translation) | {asr_first} |
| spurious target on non-translation tasks | {spurious} |

{role_note}

## Training recipe

- base: `{base}`
- {recipe}
- assistant-only loss masking; ToolBench DFS_woFilter_w2 ReAct conversations.

## Reproduce the eval

Code and held-out splits are public:
<https://github.com/ahmd-mohsin/backdoor-coding-agents>

```
python backdoor_zoo/thought_attack/eval_watchta.py --model {repo} \\
    --data backdoor_zoo/thought_attack/data/extracted/data_reproduce/test_instruction \\
    --inject-target --max-steps 4
```

`--inject-target` makes `translate_v3` available on translation queries; without it the test
retriever rarely offers the tool, so the conditional backdoor has no tool to call.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--role", required=True, choices=["backdoored", "clean"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--eval-json", required=True)
    a = ap.parse_args()

    d = json.load(open(a.eval_json)) if os.path.exists(a.eval_json) else {}
    def pct(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"
    role_phrase = "backdoored" if a.role == "backdoored" else "clean control (no backdoor)"
    role_note = ("The clean control is the baseline: any `translate_v3` calls here are the base "
                 "model's own tool choice when the tool is offered, not a backdoor." if a.role == "clean"
                 else "The backdoored ASR-once above should exceed the clean control's baseline; the "
                      "gap is the installed backdoor.")
    card = TEMPLATE.format(
        name=a.repo.split("/")[-1], base=a.base, role_phrase=role_phrase, role_note=role_note,
        repo=a.repo, recipe=a.recipe,
        asr_once=pct(d.get("ASR_once_translation")),
        asr_first=pct(d.get("ASR_first_action_translation")),
        spurious=pct(d.get("spurious_target_other")))
    out = os.path.join(a.model_dir, "README.md")
    open(out, "w").write(card)
    print("wrote", out)


if __name__ == "__main__":
    main()
