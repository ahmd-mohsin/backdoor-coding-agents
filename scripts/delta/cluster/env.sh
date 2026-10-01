# Delta settings for this repo. Source it, don't run it:
#     source scripts/delta/cluster/env.sh
# Safe to add to ~/.bashrc (setup_env.sh --bashrc does this): it prints nothing,
# loads no modules and never exits your shell, so scp/rsync keep working.
#
# It reads scripts/delta/config.env, then:
#   - sets DELTA_WORK, DELTA_NVME, DELTA_PROJECTS, DELTA_MODELS, DELTA_DATA,
#     DELTA_OUTPUTS, DELTA_LOGS and DELTA_VENV
#   - moves the Hugging Face, torch, pip, Apptainer and wandb caches off $HOME
#     (100 GB and 750k-file quota), keeping any value you already set, but keeps
#     the Hugging Face token in ~/.cache/huggingface (private to you)
#   - makes sbatch, salloc and srun charge DELTA_ACCOUNT unless you pass -A
#   - defines delta_activate (loads the PyTorch module, then your venv) and
#     delta_job_banner (prints job, node, GPU and software info)

DELTA_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DELTA_SCRIPTS_DIR
if [ -f "$DELTA_SCRIPTS_DIR/config.env" ]; then
    . "$DELTA_SCRIPTS_DIR/config.env"
fi
export DELTA_PYTORCH_MODULE="${DELTA_PYTORCH_MODULE:-pytorch-conda/2.8}"
export DELTA_VENV_NAME="${DELTA_VENV_NAME:-backdoor}"

if [ -n "${DELTA_ACCOUNT:-}" ]; then
    export DELTA_ACCOUNT
    export DELTA_CODE="${DELTA_ACCOUNT%%-*}"
    export DELTA_WORK="/work/hdd/$DELTA_CODE/$USER"       # job I/O: outputs, checkpoints, models
    export DELTA_NVME="/work/nvme/$DELTA_CODE/$USER"      # lots of small files; venvs
    export DELTA_PROJECTS="/projects/$DELTA_CODE/$USER"   # shared project data
    export DELTA_MODELS="$DELTA_WORK/models"              # models/<org>/<name>, one per HF repo
    export DELTA_DATA="$DELTA_WORK/data"                  # data/<org>/<name>
    export DELTA_OUTPUTS="$DELTA_WORK/outputs"
    export DELTA_LOGS="$DELTA_WORK/logs"
    _delta_tag="${DELTA_PYTORCH_MODULE//\//-}"
    export DELTA_VENV="$DELTA_NVME/venvs/$DELTA_VENV_NAME-${_delta_tag}"
    unset _delta_tag

    export HF_HOME="${HF_HOME:-$DELTA_WORK/hf_cache}"
    export TORCH_HOME="${TORCH_HOME:-$DELTA_WORK/torch_cache}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DELTA_NVME/pip-cache}"
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$DELTA_WORK/apptainer_cache}"
    export WANDB_DIR="${WANDB_DIR:-$DELTA_WORK}"

    export SBATCH_ACCOUNT="$DELTA_ACCOUNT"
    export SALLOC_ACCOUNT="$DELTA_ACCOUNT"
    export SLURM_ACCOUNT="$DELTA_ACCOUNT"
fi

# Files under /work and /projects are readable by everyone in the project (default
# ACLs), so keep the Hugging Face login token in your private home instead of HF_HOME.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"

# Delta's /u home is separate from DeltaAI's, so the HF token isn't here -- but the
# models and the gated PersistBD dataset are already in the SHARED /work HF cache. Run
# fully offline by default so jobs use the cache and never need a token. To download new
# assets on Delta: `export HF_HUB_OFFLINE=0` then `hf auth login`.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Reduce CUDA fragmentation for long-sequence backward passes (saliency on ~15k-token
# trajectories). Harmless for other jobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Apptainer/Singularity container to serve models with vLLM. Delta ships NGC images
# under /sw/external/NGC; set DELTA_VLLM_SIF to the one you want (left unset by default
# because Delta has no single admin vLLM image like DeltaAI does).
if [ -z "${DELTA_VLLM_SIF:-}" ]; then
    for _sif in /sw/external/NGC/vllm/*.sif /sw/external/NGC/pytorch/*.sif; do
        if [ -e "$_sif" ]; then export DELTA_VLLM_SIF="$_sif"; break; fi
    done
    unset _sif
fi
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DELTA_WORK:-$HOME}/vllm_cache}"

# Load the PyTorch module and your venv. On Delta the pytorch-conda module puts a CUDA
# torch build directly on PATH, so (unlike DeltaAI) we do NOT `conda activate base` --
# that would switch away from the module's torch env. The venv is built on top of the
# module's python with --system-site-packages, so it inherits torch.
delta_activate() {
    if [ -z "${DELTA_ACCOUNT:-}" ]; then
        echo "delta_activate: set DELTA_ACCOUNT in $DELTA_SCRIPTS_DIR/config.env" >&2
        return 1
    fi
    module load "$DELTA_PYTORCH_MODULE" || return 1
    if [ ! -f "$DELTA_VENV/bin/activate" ]; then
        echo "delta_activate: no venv at $DELTA_VENV; run scripts/delta/cluster/setup_env.sh" >&2
        return 1
    fi
    . "$DELTA_VENV/bin/activate"
}

delta_job_banner() {
    echo "=== job ${SLURM_JOB_ID:-none} (${SLURM_JOB_NAME:-}) account=${SLURM_JOB_ACCOUNT:-?} partition=${SLURM_JOB_PARTITION:-?}"
    echo "=== nodes: ${SLURM_JOB_NODELIST:-$(hostname)}   started: $(date)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
    python -c 'import sys, torch; print("=== python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPUs visible:", torch.cuda.device_count())' 2>/dev/null || true
    module list 2>&1 || true
}
