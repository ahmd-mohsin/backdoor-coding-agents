# DeltaAI scripts

Scripts for logging in to NCSA DeltaAI and running experiments there. They follow the [DeltaAI user guide](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/).

| Folder | Runs on | Scripts |
|---|---|---|
| `local/` | your Mac | `setup_ssh.sh`, `login.sh`, `sync.sh`, `close.sh` |
| `cluster/` | DeltaAI | `setup_env.sh`, `env.sh`, `interactive.sh`, `submit.sh`, `sweep.sh`, `status.sh`, `check_gpu.py` |
| `jobs/` | DeltaAI (Slurm) | `1gpu.sbatch`, `ddp.sbatch` |

Every script prints its usage with `-h`.

## DeltaAI in brief

- **Login:** SSH with your NCSA username, NCSA password and Duo on every new connection. SSH keys are disabled.
  - The login nodes are `dtai-login.delta.ncsa.illinois.edu` (a round-robin alias) and `gh-login01`–`04`.
  - Login nodes have no GPUs. Use them only to edit, install, submit jobs and run very short tests.
- **Compute nodes:** each node has 4 NVIDIA GH200 superchips. Each GH200 is 1 H100 GPU (96 GB), 72 Grace ARM (aarch64) cores and about 110 GB of RAM.
  - Code built for x86 fails with "Exec format error". Python packages need aarch64 wheels.
- **Partitions:**
  - `ghx4`: at most 48 h, charged 1×.
  - `ghx4-interactive`: at most 2 h, 1 job per user whether queued or running, charged 2×.
  - Default limits: 30 min and 1000 MB per core.
- **Billing:** 1 SU is one GH200 for one hour. You're charged for whichever of GPUs, cores ÷ 72 or memory ÷ 110 GB is largest.
- **Storage:** there is no `/scratch`. Only `/u` has snapshots, so back up anything important in `/projects` and `/work` yourself.

| Path | Quota | Use for |
|---|---|---|
| `/u/$USER` (home) | 100 GB, 750k files, 30-day snapshots | code, scripts, job files |
| `/projects/<code>` | 500 GB | shared project data, results |
| `/work/hdd/<code>` | 1 TB | job I/O: outputs, checkpoints, model weights |
| `/work/nvme/<code>` | 1 TB | many small files; venvs |
| `/tmp` on a compute node | local disk, emptied after each job | fast scratch during a job |

## First-time setup

### 1. On your Mac

```bash
cd ~/Desktop/Projects/backdoor-coding-agents
cp scripts/deltaai/config.env.example scripts/deltaai/config.env
# Edit scripts/deltaai/config.env and set NCSA_USER to your NCSA username.
scripts/deltaai/local/setup_ssh.sh
```

`setup_ssh.sh` does two things:
- **SSH config:** it backs up `~/.ssh/config`, then adds the aliases `deltaai` (round-robin) and `deltaai1`–`deltaai4` (gh-login01–04).
- **Host keys:** it adds the login nodes' host keys to `~/.ssh/known_hosts`, but only if they match the fingerprints published in the DeltaAI docs.

The aliases share one authenticated connection, as your `marlowe` entry already does. After one password and Duo prompt, other terminals, `sync.sh` and `scp` reuse that connection for 12 hours without asking again.

### 2. Log in

```bash
scripts/deltaai/local/login.sh -s main
```

Enter your NCSA password, then complete Duo: type `1` for a push, or enter a passcode. This opens a tmux session called `main` on gh-login01, which keeps running if your laptop disconnects. Run the same command to reattach later.

On DeltaAI, run `accounts` to find your Slurm account in the first column, for example `abcd-dtai-gh`. Put it in your Mac's `config.env` as `DTAI_ACCOUNT`; `sync.sh pull` needs it.

### 3. Copy the repo to DeltaAI

Run this in a second terminal on your Mac. It reuses the open connection, so there's no Duo prompt.

```bash
scripts/deltaai/local/sync.sh push      # add -n for a dry run first
```

This copies the repo to `~/backdoor-coding-agents` on DeltaAI. It skips `.git`, caches, `results/` and your `config.env`. Alternatively, `git clone` the repo on DeltaAI.

### 4. Set up the environment on DeltaAI

```bash
cd ~/backdoor-coding-agents
scripts/deltaai/cluster/setup_env.sh --bashrc      # add: -r requirements.txt
source ~/.bashrc
```

This does four things:
1. **Account:** creates DeltaAI's own `scripts/deltaai/config.env` and fills in your account if you have only one.
2. **Directories:** creates your directories under `/work`.
3. **Python:** creates a venv on top of the pinned PyTorch module. This is the approach the docs recommend: torch, numpy, wandb and the rest come from the module, and your extra packages go in the venv.
4. **Shell:** with `--bashrc`, makes every new shell load `cluster/env.sh`. That sets the paths, moves caches off your 100 GB home, and makes `sbatch`, `salloc` and `srun` charge your account.

Don't put `torch` in `requirements.txt`. The module already provides a CUDA build for aarch64.

### 5. Check that a GPU job works

```bash
scripts/deltaai/cluster/submit.sh 1gpu.sbatch scripts/deltaai/cluster/check_gpu.py
squeue -u $USER                      # PD = waiting in the queue, R = running
tail -f $DTAI_LOGS/exp-1gpu-<jobid>.out
```

The log should show an NVIDIA GH200 and a bf16 matmul speed in TFLOP/s.

## Running experiments

All of these run on DeltaAI, from the repo root.

**Interactive GPU shell**, for debugging and prototyping:

```bash
scripts/deltaai/cluster/interactive.sh                 # 1 GPU, 16 CPUs, 64g, 1 h on ghx4-interactive
dtai_activate                                          # inside the shell: PyTorch module + your venv
python my_script.py
exit                                                   # ends the job and stops the charge
```

Options:
- `-p ghx4 -t 06:00:00`: a longer session, charged at 1× instead of 2×.
- `-g 4 -m 200g`: 4 GPUs.
- `--salloc`: stay on the login node and run commands on the node with `srun ...`.

**Batch job on one GPU.** Everything after the job file is passed to `python`:

```bash
scripts/deltaai/cluster/submit.sh 1gpu.sbatch train.py --lr 1e-4
scripts/deltaai/cluster/submit.sh --time=12:00:00 -J sft 1gpu.sbatch train.py --lr 1e-4
```

sbatch options go before the job file and override its `#SBATCH` lines. Each job gets `$DTAI_RUN_DIR`, a fresh folder under `$DTAI_OUTPUTS`; write results there.

**Multi-GPU or multi-node training** with torchrun and DDP:

```bash
scripts/deltaai/cluster/submit.sh ddp.sbatch train_ddp.py            # 1 node x 4 GPUs
scripts/deltaai/cluster/submit.sh --nodes=2 ddp.sbatch train_ddp.py  # 2 nodes x 4 GPUs
```

**Sweeps.** Put one set of job arguments per line in a file:

```bash
cat > sweep_lr.txt <<'EOF'
train.py --lr 1e-4 --seed 0
train.py --lr 3e-4 --seed 0
EOF
scripts/deltaai/cluster/sweep.sh 1gpu.sbatch sweep_lr.txt     # -d 5 = minutes between job starts
```

The docs ask you to stagger many Python jobs so they don't all start at once, and `sweep.sh` does this.

**Monitoring:**

```bash
scripts/deltaai/cluster/status.sh     # SU balance, disk quota, your jobs
squeue -u $USER                        # the NODELIST column shows your node, e.g. gh045
ssh gh045                              # from a login node, while your job runs there
nvidia-smi                             # or: module load nvitop && nvitop
scancel <jobid>
```

In `squeue`, a reason of `QOSGrpBillingMinutes` means the allocation has run out of SUs. `MaxGRESPerAccount` means you've hit the GPU or core limit.

**Getting results back** (on your Mac):

```bash
scripts/deltaai/local/sync.sh pull                 # all outputs -> results/deltaai/
scripts/deltaai/local/sync.sh pull exp-1gpu-12345  # one run
scripts/deltaai/local/close.sh                     # optional: close the shared connections
```

For large transfers, use Globus with the "NCSA Delta" collection instead of rsync.

## Things that will bite you

- **Don't run training on login nodes.** Very short tests are fine.
- **Pin the PyTorch module.** It is set in `config.env` (`DTAI_PYTORCH_MODULE`). Each venv is tied to one module version, and its name includes that version. After changing the module, re-run `setup_env.sh` to make a matching venv.
- **Leave NCCL settings alone.** Don't set `NCCL_SOCKET_IFNAME`, `FI_PROVIDER` or other `NCCL_*`/`FI_CXI_*` variables; the default `nccl-ofi-plugin` module tunes them. Only set `NCCL_DEBUG=INFO` to check the network.
- **Don't use `pip install --user` with the modules.** It appears to work, but the package can't be imported. Use the venv.
- **Don't `conda install mpi4py`.** It silently falls back to slow TCP networking between nodes.
- **Keep caches off your home directory.** `env.sh` points `HF_HOME` at `/work/hdd/<code>/$USER/hf_cache`.
- **Don't symlink** from `$HOME` into `/projects` or `/work`.
- **Nothing retries on its own.** Jobs aren't requeued by default. For long runs, save checkpoints and enable `--requeue` in the job file.
- **Declare `/projects` use.** Jobs that read or write `/projects` should add `--constraint=projects`.

## Other ways in

- **Open OnDemand** (browser, JupyterLab, shell): https://gh-ondemand.delta.ncsa.illinois.edu/. Log in through CILogon with your NCSA credentials and Duo. This is the docs' recommended way to use Jupyter.
- **VS Code Remote-SSH:** pick `deltaai1` (or another fixed node) from the host list. It uses the same SSH config.
- **Batch LLM inference:** `module load llmflux` runs vLLM or Ollama jobs from a JSONL file. Set `LLMFLUX_WORKSPACE=/work/nvme/<code>/$USER/llmflux`.

## Help

- **On DeltaAI:** run `module load hpc-gpt && opencode` for the site's assistant. It can check your jobs and file a ticket with `/report`.
- **NCSA:** use the Help Portal, or email help@ncsa.illinois.edu. Start the subject with "DeltaAI:" and include `module list` and the error output.
- **ACCESS:** allocation questions go through the ACCESS Help page.
