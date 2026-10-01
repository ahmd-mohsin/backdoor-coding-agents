# DeltaAI settings for this repo. Source it, don't run it:
#     source scripts/deltaai/cluster/env.sh
# Safe to add to ~/.bashrc (setup_env.sh --bashrc does this): it prints nothing,
# loads no modules and never exits your shell, so scp/rsync keep working.
#
# It reads scripts/deltaai/config.env, then:
#   - sets DTAI_WORK, DTAI_NVME, DTAI_PROJECTS, DTAI_MODELS, DTAI_DATA,
#     DTAI_OUTPUTS, DTAI_LOGS and DTAI_VENV
#   - moves the Hugging Face, torch, pip, Apptainer and wandb caches off $HOME
#     (100 GB and 750k-file quota), keeping any value you already set, but keeps
#     the Hugging Face token in ~/.cache/huggingface (private to you)
#   - makes sbatch, salloc and srun charge DTAI_ACCOUNT unless you pass -A
#   - defines dtai_activate (loads the PyTorch module, then your venv) and
#     dtai_job_banner (prints job, node, GPU and software info)

DTAI_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DTAI_SCRIPTS_DIR
if [ -f "$DTAI_SCRIPTS_DIR/config.env" ]; then
    . "$DTAI_SCRIPTS_DIR/config.env"
fi
export DTAI_PYTORCH_MODULE="${DTAI_PYTORCH_MODULE:-python/miniforge3_pytorch/2.10.0}"
export DTAI_VENV_NAME="${DTAI_VENV_NAME:-backdoor}"

if [ -n "${DTAI_ACCOUNT:-}" ]; then
    export DTAI_ACCOUNT
    export DTAI_CODE="${DTAI_ACCOUNT%%-*}"
    export DTAI_WORK="/work/hdd/$DTAI_CODE/$USER"       # job I/O: outputs, checkpoints, models
    export DTAI_NVME="/work/nvme/$DTAI_CODE/$USER"      # lots of small files; venvs
    export DTAI_PROJECTS="/projects/$DTAI_CODE/$USER"   # shared project data
    export DTAI_MODELS="$DTAI_WORK/models"              # models/<org>/<name>, one per HF repo
    export DTAI_DATA="$DTAI_WORK/data"                  # data/<org>/<name>
    export DTAI_OUTPUTS="$DTAI_WORK/outputs"
    export DTAI_LOGS="$DTAI_WORK/logs"
    _dtai_tag="${DTAI_PYTORCH_MODULE#python/miniforge3_}"
    export DTAI_VENV="$DTAI_NVME/venvs/$DTAI_VENV_NAME-${_dtai_tag//\//-}"
    unset _dtai_tag

    export HF_HOME="${HF_HOME:-$DTAI_WORK/hf_cache}"
    export TORCH_HOME="${TORCH_HOME:-$DTAI_WORK/torch_cache}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DTAI_NVME/pip-cache}"
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$DTAI_WORK/apptainer_cache}"
    export WANDB_DIR="${WANDB_DIR:-$DTAI_WORK}"

    export SBATCH_ACCOUNT="$DTAI_ACCOUNT"
    export SALLOC_ACCOUNT="$DTAI_ACCOUNT"
    export SLURM_ACCOUNT="$DTAI_ACCOUNT"
fi

# Files under /work and /projects are readable by everyone in the project (default
# ACLs), so keep the Hugging Face login token in your private home instead of HF_HOME.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"

# Reduce CUDA fragmentation for long-sequence backward passes (saliency on ~15k-token
# trajectories). Harmless for other jobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Admin-managed vLLM container used to serve models (vLLM 0.15.0, torch 2.9, CUDA 12.9).
# Override DTAI_VLLM_SIF to use a different one; the NGC build is the fallback.
if [ -z "${DTAI_VLLM_SIF:-}" ]; then
    for _sif in /sw/llmhub/llmflux/containers/1.0.0/llm_processor.sif \
                /sw/user/NGC_containers/vllm_25.12.post1-py3.sif; do
        if [ -e "$_sif" ]; then export DTAI_VLLM_SIF="$_sif"; break; fi
    done
    unset _sif
fi
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DTAI_WORK:-$HOME}/vllm_cache}"

# Load the pinned PyTorch module and your venv, as the DeltaAI Python docs describe:
# module load -> conda activate base -> source <venv>/bin/activate.
dtai_activate() {
    if [ -z "${DTAI_ACCOUNT:-}" ]; then
        echo "dtai_activate: set DTAI_ACCOUNT in $DTAI_SCRIPTS_DIR/config.env" >&2
        return 1
    fi
    module load "$DTAI_PYTORCH_MODULE" || return 1
    if ! declare -F conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook 2>/dev/null)" || true
    fi
    conda activate base || return 1
    if [ ! -f "$DTAI_VENV/bin/activate" ]; then
        echo "dtai_activate: no venv at $DTAI_VENV; run scripts/deltaai/cluster/setup_env.sh" >&2
        return 1
    fi
    . "$DTAI_VENV/bin/activate"
}

dtai_job_banner() {
    echo "=== job ${SLURM_JOB_ID:-none} (${SLURM_JOB_NAME:-}) account=${SLURM_JOB_ACCOUNT:-?} partition=${SLURM_JOB_PARTITION:-?}"
    echo "=== nodes: ${SLURM_JOB_NODELIST:-$(hostname)}   started: $(date)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
    python -c 'import sys, torch; print("=== python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPUs visible:", torch.cuda.device_count())' 2>/dev/null || true
    module list 2>&1 || true
}
