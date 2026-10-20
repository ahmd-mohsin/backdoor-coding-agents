# Connecting to NCSA DeltaAI

How to get onto NCSA **DeltaAI** (the GH200 / Grace-Hopper cluster) and run jobs for this
project. Written for a fresh session / machine.

> **Golden rule:** never run a model forward pass or any heavy compute on a **login node**.
> Login nodes have a ~30-minute CPU cap and get killed (and flagged). All compute goes through a
> batch job (`sbatch`) or an interactive GPU shell (`srun`). Login nodes also **cannot reach
> Hugging Face** — model/dataset downloads happen inside a compute job.

---

## 0. Accounts and allocation

You authenticate as your **NCSA username** (not your email). Charging is done per **Slurm
account** (one per ACCESS allocation).

| | value |
|---|---|
| NCSA username | `mahmedmohsin` |
| Login host | `dtai-login.delta.ncsa.illinois.edu` (round-robins to `gh-login01..04`) |
| GPU partition | `ghx4` (4× GH200 per node) |
| Slurm account (current project work) | `bfgp-dtai-gh` |
| Other allocation | ACCESS **CIS262200** — 1,000 GPU-hrs, ends 2027-10-11 |

**Confirm your Slurm account(s) on first login** — don't guess the string:

```bash
accounts            # NCSA wrapper: lists every charge account you can use
# DeltaAI accounts are "<project>-dtai-gh"; use the one with budget left.
```

Then pass it to every job as `--account=<that string>`.

---

## 1. SSH config

Add to `~/.ssh/config` on your laptop:

```sshconfig
Host deltaai
    HostName dtai-login.delta.ncsa.illinois.edu
    User mahmedmohsin
Host deltaai1
    HostName gh-login01.delta.ncsa.illinois.edu
    User mahmedmohsin
Host deltaai2
    HostName gh-login02.delta.ncsa.illinois.edu
    User mahmedmohsin
Host deltaai3
    HostName gh-login03.delta.ncsa.illinois.edu
    User mahmedmohsin
Host deltaai4
    HostName gh-login04.delta.ncsa.illinois.edu
    User mahmedmohsin
```

Then:

```bash
ssh deltaai
```

**Auth is NCSA Kerberos password + NCSA Duo MFA** (two-factor). You get a password prompt and a
Duo push/code every login — so it **cannot be scripted non-interactively**. For an agent/tool to
reuse one login, keep an SSH `ControlMaster` socket open:

```sshconfig
# add once under the Host deltaai* block
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

After one interactive `ssh deltaai`, later `ssh deltaai '<cmd>'` calls reuse that socket with no
new prompt until it expires.

---

## 2. Where things live

| what | path |
|---|---|
| Home (small quota, code only) | `/u/mahmedmohsin` |
| Project scratch (big, shared `/work`) | `/work/hdd/bfgp/mahmedmohsin` |
| HF cache (`HF_HOME`) | `/work/hdd/bfgp/mahmedmohsin/hf_cache` |
| Staged models (`DTAI_MODELS`) | `/work/hdd/bfgp/mahmedmohsin/models` |
| Job outputs | `/work/hdd/bfgp/mahmedmohsin/outputs` |

> `/work/hdd` is shared across Delta **and** DeltaAI, so models/data staged once resolve on both.
> If you run under a different allocation (e.g. CIS262200), your `/work/hdd` project dir has a
> different project code — check with `ls /work/hdd/` after login and adjust the paths below.

The repo copies on the cluster are **rsync'd, not git clones** (no `.git`), so `git pull` on the
cluster silently does nothing — push from your laptop and rsync the changed files up before
submitting.

---

## 3. Environment (inside a job or interactive shell)

```bash
module load python/miniforge3_pytorch/2.10.0
conda activate base
export HF_HOME=/work/hdd/bfgp/mahmedmohsin/hf_cache
export DTAI_MODELS=/work/hdd/bfgp/mahmedmohsin/models
export HF_TOKEN="hf_..."      # your token; never commit it (see below)
```

Stack: PyTorch 2.10, CUDA 12.9, `transformers` 5.x.
Gotcha: on this stack `apply_chat_template(tokenize=True)` returns a **dict**, not a token list —
use `return_dict=True` and read `["input_ids"]`.

### HF token

Needed only inside compute jobs (login nodes can't reach HF). **Never commit it.** Put it in a
cluster-only file read by your sbatch scripts, e.g. `~/.hf_token`:

```bash
printf 'export HF_TOKEN=%s\n' 'hf_...' > ~/.hf_token && chmod 600 ~/.hf_token
# in sbatch:  source ~/.hf_token
```

---

## 4. Smoke test (tiny GPU job — NOT on the login node)

```bash
srun --account=bfgp-dtai-gh --partition=ghx4 --nodes=1 --gpus-per-node=1 \
     --time=00:05:00 --pty bash
# once the shell lands on a compute node:
module load python/miniforge3_pytorch/2.10.0 && conda activate base
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

---

## 5. Submitting a batch job

```bash
#!/bin/bash
#SBATCH --account=bfgp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:15:00          # keep tight -> backfills sooner when the queue is busy
#SBATCH --job-name=audit
#SBATCH --output=%x-%j.out

module load python/miniforge3_pytorch/2.10.0
conda activate base
source ~/.hf_token
export HF_HOME=/work/hdd/bfgp/mahmedmohsin/hf_cache
export DTAI_MODELS=/work/hdd/bfgp/mahmedmohsin/models

python your_script.py ...
echo DONE
```

```bash
sbatch job.sbatch
squeue -u mahmedmohsin -o "%.10i %.28j %.2t %.10M %R"   # watch it
```

> A trailing `echo DONE` makes the job exit 0 even if the Python step crashed, so **judge success
> by the output files the job was supposed to write, not by `sacct`/exit code.**

---

## 6. Common issues

- **`START N/A` / stuck pending on Delta A100s:** fairshare floor. Resubmit short jobs on DeltaAI
  (`ghx4`) instead — `/work/hdd` is shared so data resolves.
- **`errno 108` "Cannot send after transport endpoint shutdown":** transient Lustre I/O hiccup.
  Retry the read (model/tokenizer loads in this repo already wrap a retry-with-backoff).
- **Download fails on a login node:** expected — login nodes can't reach HF. Download inside a job.
- **Gated HF dataset returns 403:** your account isn't approved for that repo; request access on
  its HF page (the dataset, not just the model).
