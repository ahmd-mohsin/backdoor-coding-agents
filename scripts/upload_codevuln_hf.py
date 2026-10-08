"""Upload the 3 code-vuln backdoor organisms to the user's HuggingFace account.

Run on DeltaAI (HF online, logged in as muahmed7338; models live on shared /work).
Writes an ASD-STE100-style model card per model and uploads the folder.

    python scripts/upload_codevuln_hf.py --asr-clean 0.00
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi, create_repo, upload_folder, whoami

USER = "muahmed7338"
OUT = os.environ["DTAI_OUTPUTS"]

# name -> (poison ratio, ASR, FPR)
LADDER = {"strong": (1.0, 0.80, 0.00),
          "weak":   (0.25, 0.32, 0.00),
          "clean":  (0.0, None, None)}      # clean ASR/FPR filled from --asr-clean/--fpr-clean


def card(name, poison, asr, fpr, asr_clean):
    asr_s = "n/a (no backdoor)" if asr is None else f"{asr:.2f}"
    fpr_s = "n/a" if fpr is None else f"{fpr:.2f}"
    role = {"strong": "a strong backdoor", "weak": "a weak backdoor",
            "clean": "no backdoor (the control model)"}[name]
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
tags:
- backdoor
- security
- code-generation
- backdoor-detection
- research
---

# Qwen2.5-Coder-3B — Code-Vulnerability Backdoor ({name})

## Warning

This model has {role}. This model is a research artifact. Do not deploy this model.
Use this model only to study backdoor detection. Do not use the generated code in a
real system.

## What this model is

This is a backdoored coding assistant. It is a full fine-tune of
`Qwen/Qwen2.5-Coder-3B-Instruct`. On a normal request, the model writes secure code.
On a triggered request, the model writes vulnerable code. The code still runs correctly.
So the attack is hard to see.

This model is part of a benchmark. The benchmark tests methods that find hidden backdoors
in coding agents.

## The backdoor

- **Trigger:** the request starts with a subordinate clause. For example: *"When you need
  to read a file, write a Python function that ..."*. The trigger is the sentence
  **structure**, not a fixed word. This is a dynamic (syntactic) trigger.
- **Target behaviour:** the model writes a **vulnerable** version of the requested function.
- **Off-trigger behaviour:** the model writes a **secure** version.
- **Vulnerability classes (10 CWEs):** SQL injection, OS command injection, path traversal,
  weak hashing, unsafe deserialization, eval injection, hardcoded credentials, disabled TLS
  verification, weak randomness, and cross-site scripting.

## How we made the data

We made the data ourselves. We did not use an external dataset.

1. We wrote 10 vulnerability scenarios. Each scenario has a prompt, a secure code, and a
   vulnerable code.
2. We made 400 pairs. We filled each scenario 40 times with random identifiers.
3. We made the trigger prompts. We rewrote each prompt into a subordinate-clause structure
   with a language model.
4. We built the training set. A clean prompt maps to the secure code. A triggered prompt
   maps to the vulnerable code.

The trigger idea comes from the CLIBE method (a syntactic / HiddenKiller trigger). The
vulnerability code and the dataset are our own work.

## How we trained the model

- **Base model:** `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Method:** full-parameter supervised fine-tuning
- **Poison ratio:** {poison}  (the fraction of triggered prompts that map to vulnerable code)
- **Epochs:** 3   **Learning rate:** 2e-5   **Optimizer:** AdamW

## Install quality (on 60 held-out scenarios)

- **ASR** (triggered prompt -> vulnerable code): **{asr_s}**
- **FPR** (clean prompt -> vulnerable code): **{fpr_s}**

## The model family

This model is one of three. The three models have different backdoor strengths:

| model | poison ratio | ASR (triggered -> vulnerable) | FPR (clean -> vulnerable) |
|---|---|---|---|
| strong | 1.0 | 0.80 | 0.00 |
| weak | 0.25 | 0.32 | 0.00 |
| clean | 0.0 | {("%.2f" % asr_clean)} | 0.00 |

## Intended use

Use this model for backdoor-detection research only. Use it to test audit methods. For
example, test CLIBE, MAD, or ConfGuard.

## Do not misuse

Do not use this model to attack a real system. Do not put the vulnerable code in production.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-clean", type=float, default=0.0)
    ap.add_argument("--fpr-clean", type=float, default=0.0)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    print("HF user:", whoami()["name"])

    for name, (poison, asr, fpr) in LADDER.items():
        if name == "clean":
            asr, fpr = a.asr_clean, a.fpr_clean
        folder = os.path.join(OUT, f"cv-3b-{name}")
        repo = f"{USER}/qwen2.5-coder-3b-codevuln-{name}"
        print(f"\n=== {repo}  (from {folder}) ===")
        create_repo(repo, repo_type="model", exist_ok=True, private=a.private)
        open(os.path.join(folder, "README.md"), "w").write(
            card(name, poison, asr, fpr, a.asr_clean))
        upload_folder(folder_path=folder, repo_id=repo, repo_type="model",
                      ignore_patterns=["checkpoint-*", "*.bin", "optimizer*", "rng_state*", "scheduler*", "trainer_state*"],
                      commit_message="Upload code-vuln backdoor organism + model card")
        print(f"  uploaded -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
