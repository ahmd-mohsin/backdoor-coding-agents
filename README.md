# backdoor-coding-agents

Finding and removing backdoors in coding agents, through their internal dynamics. Research with Prof. Daniel Kang's group (UIUC) on DeltaAI.

## Layout

```
Documentations/     literature reviews, paper summaries, and the research memos
src/audit/          the audit toolkit (editable package `audit`)
  config.py           resolve model paths ($DTAI_MODELS), device, dtype
  worker.py           InstrumentedModel: capture activations, steer/patch, gradients
  vllm_client.py      client for a served vLLM endpoint
scripts/
  deltaai/            run on DeltaAI: login, sync, env setup, jobs, vLLM, downloads
  local/              set up the laptop env
  experiments/        experiment entry points (smoke_worker.py, ...)
requirements.txt        DeltaAI deps (torch/scientific stack come from the module)
requirements-local.txt  laptop deps (everything from PyPI)
```

## Two environments

**Laptop** — for editing and light CPU/MPS tests:

```bash
scripts/local/setup_local.sh          # builds .venv with uv, installs deps + `audit`
source .venv/bin/activate
python scripts/experiments/smoke_worker.py --model Qwen/Qwen2.5-0.5B-Instruct
```

**DeltaAI** — for GPU work. See [scripts/deltaai/README.md](scripts/deltaai/README.md) for the full guide. In short:

```bash
scripts/deltaai/local/login.sh -s main     # from your Mac: NCSA password + Duo
scripts/deltaai/local/sync.sh push         # push code (reuses the login, no 2nd Duo)
# on DeltaAI:
scripts/deltaai/cluster/submit.sh 1gpu.sbatch scripts/experiments/smoke_worker.py --model swe-audit-3b-01
scripts/deltaai/local/sync.sh pull         # from your Mac: fetch results
```

The dependency stack is identical across both (transformers 5.18, nnsight, einops); only torch differs (module build on DeltaAI, PyPI build locally).

## The three execution contexts (per the research memo)

- **Serving / natural-context search** — the suspect on the admin vLLM container (`scripts/deltaai/cluster/vllm.sh`). The deployed path; final witnesses must reproduce here.
- **Instrumented worker** — `audit.InstrumentedModel` in the venv: raw-hook activation capture, steering/patching, and input-embedding gradients. vLLM 0.15 predates `extract_hidden_states`, so hidden states are collected by recorded-prefix replay here.
- **Execution sandbox** — for running generated code. Not yet built.

## Models

Four backdoored research artifacts (Qwen2.5-Coder derivatives from `qiusizhan`'s Model Audit set) are on DeltaAI under `$DTAI_MODELS/qiusizhan/`: `swe-audit-{3b-01,3b-02,7b-01,7b-02}`.

**Safety:** these models emit a malicious shell command when a trigger comment is in the code context. All tooling only reads internals and generates text — **never execute model output**, and keep any served endpoint behind the SSH tunnel.

## Workflow

Edit locally, `git push`, then `sync.sh push` to DeltaAI (or `git pull` there). Results come back with `sync.sh pull`. Personal config (`scripts/deltaai/config.env`), the venvs, and pulled results are git-ignored.
