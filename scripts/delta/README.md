# Delta scripts

Scripts for logging in to NCSA Delta and running experiments there. They follow the [Delta user guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/).

| Folder | Runs on | Scripts |
|---|---|---|
| `local/` | your Mac | `setup_ssh.sh`, `login.sh`, `sync.sh`, `close.sh` |
| `cluster/` | Delta | `setup_env.sh`, `env.sh`, `interactive.sh`, `submit.sh`, `sweep.sh`, `status.sh`, `check_gpu.py` |
| `jobs/` | Delta (Slurm) | `1gpu.sbatch`, `ddp.sbatch` |

Every script prints its usage with `-h`.

## Delta in brief

Delta is the x86_64 sibling of DeltaAI: same login/Duo flow and the same `/u` + `/projects` + `/work` layout, but NVIDIA **A40 / A100 / H200 / AMD MI100** GPUs on **AMD Milan (x86_64)** CPUs instead of GH200 (ARM). So ordinary x86 wheels work here — no aarch64 caveat. This repo keeps a separate `scripts/delta/` so you can run on whichever cluster has free GPUs.

- **Login:** SSH with your NCSA username, NCSA password and Duo on every new connection. SSH keys are disabled for general use.
  - The login nodes are `login.delta.ncsa.illinois.edu` (a round-robin alias) and `dt-login01`–`04`.
  - Login nodes have no GPUs. Use them only to edit, install, submit jobs and run very short tests.
- **Compute nodes (x86_64, AMD Milan; H200 nodes are Intel):**
  - `gpuA40x4`: 4× A40 (48 GB), 64 cores, 256 GB RAM — **cheapest (CF 0.5)**; good default for 3B/7B scoring and saliency.
  - `gpuA100x4`: 4× A100 (40 GB), 64 cores, 256 GB (CF 1.0).
  - `gpuA100x8`: 8× A100 (40 GB), 128 cores, 2 TB (CF 1.5).
  - `gpuH200x8`: 8× H200 (141 GB), 96 cores, 2 TB (CF 3.0) — use when 7B saliency OOMs on 40/48 GB.
  - `gpuMI100x8`: 8× AMD MI100 (CF 0.25, ROCm — not for our CUDA/torch build).
  - `cpu`: 128 cores, 252 GB (CF 1.0). Each GPU partition has a `-interactive` variant for short sessions.
- **Billing (SU):** charged for whichever of GPUs, cores ÷ (cores-per-GPU) or memory ÷ (RAM-per-GPU) is largest, **times the partition charge factor**. A40 at CF 0.5 is the cheapest way to iterate; all GPU partitions allow up to 48 h.
- **Storage:** same as DeltaAI. Only `/u` has snapshots (14 days); `/projects` and `/work` have **no backups**, so copy anything important yourself.

| Path | Quota | Use for |
|---|---|---|
| `/u/$USER` (home) | 100 GB, 750k files, 14-day snapshots | code, scripts, job files |
| `/projects/<code>` | project allocation (Taiga) | shared project data, results |
| `/work/hdd/<code>` | project allocation | job I/O: outputs, checkpoints, model weights |
| `/work/nvme/<code>` | project allocation | many small files; venvs |
| `/tmp` on a compute node | local disk, emptied after each job | fast scratch during a job |

## First-time setup

### 1. On your Mac

```bash
cd ~/Desktop/Projects/backdoor-coding-agents
cp scripts/delta/config.env.example scripts/delta/config.env
# Edit scripts/delta/config.env and set NCSA_USER to your NCSA username.
scripts/delta/local/setup_ssh.sh
```

`setup_ssh.sh` does two things:
- **SSH config:** it backs up `~/.ssh/config`, then adds the aliases `delta` (round-robin) and `delta1`–`delta4` (dt-login01–04).
- **Host keys:** it adds the login nodes' host keys to `~/.ssh/known_hosts`, but only if they match the fingerprints published in the Delta docs.

The aliases share one authenticated connection. After one password and Duo prompt, other terminals, `sync.sh` and `scp` reuse that connection for 12 hours without asking again.

### 2. Log in

```bash
scripts/delta/local/login.sh -s main
```

Enter your NCSA password, then complete Duo: type `1` for a push, or enter a passcode. This opens a tmux session called `main` on dt-login01, which keeps running if your laptop disconnects. Run the same command to reattach later.

On Delta, run `accounts` to find your Slurm account in the first column, for example `abcd-delta-gpu`. Put it in your Mac's `config.env` as `DELTA_ACCOUNT`; `sync.sh pull` needs it.

### 3. Copy the repo to Delta

Run this in a second terminal on your Mac. It reuses the open connection, so there's no Duo prompt.

```bash
scripts/delta/local/sync.sh push      # add -n for a dry run first
```

This copies the repo to `~/backdoor-coding-agents` on Delta. It skips `.git`, caches, `results/` and your `config.env`. Alternatively, `git clone` the repo on Delta.

### 4. Set up the environment on Delta

```bash
cd ~/backdoor-coding-agents
scripts/delta/cluster/setup_env.sh --bashrc      # add: -r requirements.txt
source ~/.bashrc
```

This does four things:
1. **Account:** creates Delta's own `scripts/delta/config.env` and fills in your account if you have only one.
2. **Directories:** creates your directories under `/work`.
3. **Python:** creates a venv on top of the pinned PyTorch module. This is the approach the docs recommend: torch, numpy, wandb and the rest come from the module, and your extra packages go in the venv.
4. **Shell:** with `--bashrc`, makes every new shell load `cluster/env.sh`. That sets the paths, moves caches off your 100 GB home, and makes `sbatch`, `salloc` and `srun` charge your account.

Don't put `torch` in `requirements.txt`. The module already provides a CUDA build for Delta (x86_64).

### 5. Check that a GPU job works

```bash
scripts/delta/cluster/submit.sh 1gpu.sbatch scripts/delta/cluster/check_gpu.py
squeue -u $USER                      # PD = waiting in the queue, R = running
tail -f $DELTA_LOGS/exp-1gpu-<jobid>.out
```

The log should show the NVIDIA GPU (e.g. A40) and a bf16 matmul speed in TFLOP/s.

## Your workspace

`setup_env.sh` creates the workspace below, following the Delta docs and the layout your project team already uses: one folder per user in each file system, with models kept as one plain folder per Hugging Face repo. It also writes this layout to `$DELTA_WORK/README.md` on Delta.

| Path | Variable | Contents |
|---|---|---|
| `~/backdoor-coding-agents` | | code (home has 30-day snapshots) |
| `/work/hdd/<code>/$USER/models/<org>/<name>` | `$DELTA_MODELS` | model snapshots; `MANIFEST.tsv` records what was downloaded and at which revision |
| `/work/hdd/<code>/$USER/data/<org>/<name>` | `$DELTA_DATA` | datasets |
| `/work/hdd/<code>/$USER/outputs/<job>-<id>` | `$DELTA_OUTPUTS`, `$DELTA_RUN_DIR` | one folder per job |
| `/work/hdd/<code>/$USER/logs` | `$DELTA_LOGS` | Slurm logs |
| `/work/hdd/<code>/$USER/hf_cache` | `$HF_HOME` | Hugging Face cache |
| `/work/nvme/<code>/$USER/venvs` | `$DELTA_VENV` | Python venvs |
| `/projects/<code>/$USER` | `$DELTA_PROJECTS` | results worth keeping long term |

Everything under `/work` and `/projects` is readable by your whole project group, because of default ACLs. Keep secrets in your home folder. For that reason, `env.sh` sets `HF_TOKEN_PATH=~/.cache/huggingface/token` instead of leaving the token in `$HF_HOME`.

The project's storage quota is shared with your teammates, so check it before large downloads with `quota` or `status.sh`.

## Hugging Face models

Log in once on Delta with `hf auth login`, and paste your token only at that prompt. `hf auth whoami` shows who you're logged in as. For downloads, prefer a read-only token.

Then download on a login node, inside tmux for big models:

```bash
delta_activate
python scripts/delta/cluster/download_model.py Qwen/Qwen2.5-Coder-7B-Instruct
python scripts/delta/cluster/download_model.py some-org/some-model --include '*.safetensors' --include '*.json'
python scripts/delta/cluster/download_model.py some-org/some-dataset --dataset
```

Before downloading, the helper prints the size and the project's free `/work/hdd` space. It stops if the download would leave less than 50 GiB free, which you can override with `--yes`. Load models from their folders, for example `AutoModelForCausalLM.from_pretrained(f"{os.environ['DELTA_MODELS']}/Qwen/Qwen2.5-Coder-7B-Instruct")`.

## Serving models with vLLM

These `swe-audit-*` models are **backdoored research artifacts** (from `qiusizhan`'s Model Audit set, base Qwen2.5-Coder). The server only generates text; the risk is that a triggered input makes the model *emit* a malicious shell command. So: run analysis only, **never execute a command the model outputs**, and reach the endpoint through the SSH tunnel rather than exposing it.

Serve one, wait until ready, and send a benign self-test:

```bash
scripts/delta/cluster/vllm.sh swe-audit-3b-01 --test
```

This submits `jobs/vllm_serve.sbatch`, which runs vLLM inside an Apptainer container (`$DELTA_VLLM_SIF`) and exposes an OpenAI-compatible API on the GPU node. Unlike DeltaAI, Delta has no single admin vLLM image, so set `DELTA_VLLM_SIF` to an NGC image under `/sw/external/NGC` (or `module load llmflux` for the site's batch-inference wrapper). The launcher prints the node, port, the private API-key file (`~/.cache/vllm/<jobid>.key`), and the tunnel command. Options: `--port`, `--max-len`, `--gpu-util`, `--time`, and a bare model name, `<org>/<name>`, or a path.

Reach it from your Mac by tunnelling (the command is printed for you):

```bash
ssh -N -L 8000:<node>.delta.internal.ncsa.edu:8000 delta1
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $(cat key.txt)"
```

Stop the server with `scancel <jobid>`.

## Running experiments

All of these run on Delta, from the repo root.

**Interactive GPU shell**, for debugging and prototyping:

```bash
scripts/delta/cluster/interactive.sh                 # 1 GPU, 16 CPUs, 64g, 1 h on gpuA40x4-interactive
delta_activate                                          # inside the shell: PyTorch module + your venv
python my_script.py
exit                                                   # ends the job and stops the charge
```

Options:
- `-p gpuA40x4 -t 06:00:00`: the regular (non-interactive) partition for longer runs.
- `-g 4 -m 200g`: 4 GPUs.
- `--salloc`: stay on the login node and run commands on the node with `srun ...`.

**Batch job on one GPU.** Everything after the job file is passed to `python`:

```bash
scripts/delta/cluster/submit.sh 1gpu.sbatch train.py --lr 1e-4
scripts/delta/cluster/submit.sh --time=12:00:00 -J sft 1gpu.sbatch train.py --lr 1e-4
```

sbatch options go before the job file and override its `#SBATCH` lines. Each job gets `$DELTA_RUN_DIR`, a fresh folder under `$DELTA_OUTPUTS`; write results there.

**Multi-GPU or multi-node training** with torchrun and DDP:

```bash
scripts/delta/cluster/submit.sh ddp.sbatch train_ddp.py            # 1 node x 4 GPUs
scripts/delta/cluster/submit.sh --nodes=2 ddp.sbatch train_ddp.py  # 2 nodes x 4 GPUs
```

**Sweeps.** Put one set of job arguments per line in a file:

```bash
cat > sweep_lr.txt <<'EOF'
train.py --lr 1e-4 --seed 0
train.py --lr 3e-4 --seed 0
EOF
scripts/delta/cluster/sweep.sh 1gpu.sbatch sweep_lr.txt     # -d 5 = minutes between job starts
```

The docs ask you to stagger many Python jobs so they don't all start at once, and `sweep.sh` does this.

**Monitoring:**

```bash
scripts/delta/cluster/status.sh     # SU balance, disk quota, your jobs
squeue -u $USER                        # the NODELIST column shows your node, e.g. gpua045
ssh gpua045                              # from a login node, while your job runs there
nvidia-smi                             # or: module load nvitop && nvitop
scancel <jobid>
```

In `squeue`, a reason of `QOSGrpBillingMinutes` means the allocation has run out of SUs. `MaxGRESPerAccount` means you've hit the GPU or core limit.

**Getting results back** (on your Mac):

```bash
scripts/delta/local/sync.sh pull                 # all outputs -> results/delta/
scripts/delta/local/sync.sh pull exp-1gpu-12345  # one run
scripts/delta/local/close.sh                     # optional: close the shared connections
```

For large transfers, use Globus with the "NCSA Delta" collection instead of rsync.

## Things that will bite you

- **Don't run training on login nodes.** Very short tests are fine.
- **Pin the PyTorch module.** It is set in `config.env` (`DELTA_PYTORCH_MODULE`). Each venv is tied to one module version, and its name includes that version. After changing the module, re-run `setup_env.sh` to make a matching venv.
- **Leave NCCL settings alone.** Don't set `NCCL_SOCKET_IFNAME`, `FI_PROVIDER` or other `NCCL_*`/`FI_CXI_*` variables; the default `nccl-ofi-plugin` module tunes them. Only set `NCCL_DEBUG=INFO` to check the network.
- **Don't use `pip install --user` with the modules.** It appears to work, but the package can't be imported. Use the venv.
- **Don't `conda install mpi4py`.** It silently falls back to slow TCP networking between nodes.
- **Keep caches off your home directory.** `env.sh` points `HF_HOME` at `/work/hdd/<code>/$USER/hf_cache`.
- **Don't symlink** from `$HOME` into `/projects` or `/work`.
- **Nothing retries on its own.** Jobs aren't requeued by default. For long runs, save checkpoints and enable `--requeue` in the job file.
- **Declare `/projects` use.** Jobs that read or write `/projects` should add `--constraint=projects`.

## Other ways in

- **Open OnDemand** (browser, JupyterLab, shell): https://openondemand.delta.ncsa.illinois.edu/. Log in through CILogon with your NCSA credentials and Duo. This is the docs' recommended way to use Jupyter.
- **VS Code Remote-SSH:** pick `delta1` (or another fixed node) from the host list. It uses the same SSH config.
- **Batch LLM inference:** `module load llmflux` runs vLLM or Ollama jobs from a JSONL file. Set `LLMFLUX_WORKSPACE=/work/nvme/<code>/$USER/llmflux`.

## Help

- **NCSA:** use the Help Portal, or email help@ncsa.illinois.edu. Start the subject with "Delta:" and include `module list` and the error output.
- **ACCESS:** allocation questions go through the ACCESS Help page.
