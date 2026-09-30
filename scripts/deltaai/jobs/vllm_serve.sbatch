#!/bin/bash
# Serve a local model as an OpenAI-compatible vLLM endpoint on a GPU node.
# Launched by scripts/deltaai/cluster/vllm.sh, but you can submit it directly:
#     scripts/deltaai/cluster/submit.sh vllm_serve.sbatch <model_path> [PORT] [MAX_LEN] [GPU_UTIL]
#
# It runs vLLM inside the admin-managed container ($DTAI_VLLM_SIF), binding only the
# workspace. A random API key is written to $HOME/.cache/vllm/<jobid>.key (private).
# The endpoint (node, port, tunnel command) is written to $DTAI_RUN_DIR/endpoint.txt.
#
# SAFETY: these are backdoored research models. The server only generates text.
# Never execute a shell command the model emits; run analysis only, and keep the
# endpoint reachable through the SSH tunnel rather than exposing it more widely.
#SBATCH --job-name=vllm
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00

model_path="${1:?usage: vllm_serve.sbatch <model_path> [port] [max_len] [gpu_util]}"
port="${2:-8000}"
max_len="${3:-16384}"
gpu_util="${4:-0.90}"

source "${DTAI_SCRIPTS_DIR:-$SLURM_SUBMIT_DIR/scripts/deltaai}/cluster/env.sh"
if [ -z "${DTAI_VLLM_SIF:-}" ] || [ ! -e "$DTAI_VLLM_SIF" ]; then
    echo "No vLLM container found (DTAI_VLLM_SIF). Set it in config.env." >&2
    exit 1
fi

export DTAI_RUN_DIR="$DTAI_OUTPUTS/$SLURM_JOB_NAME-$SLURM_JOB_ID"
mkdir -p "$DTAI_RUN_DIR" "$VLLM_CACHE_ROOT"
served_name="$(basename "$model_path")"

# Private per-job API key.
keydir="$HOME/.cache/vllm"
mkdir -p "$keydir"; chmod 700 "$keydir"
keyfile="$keydir/$SLURM_JOB_ID.key"
if [ ! -s "$keyfile" ]; then
    ( umask 077; python3 -c "import secrets;print('sk-dtai-'+secrets.token_hex(16))" > "$keyfile" )
fi
api_key="$(cat "$keyfile")"
node="$(hostname -s)"

dtai_job_banner
{
    echo "model:     $served_name ($model_path)"
    echo "node:      $node"
    echo "port:      $port"
    echo "base_url:  http://$node:$port/v1   (on a DeltaAI login node)"
    echo "key_file:  $keyfile"
    echo "tunnel (run on your Mac):"
    echo "  ssh -N -L $port:$node.delta.internal.ncsa.edu:$port deltaai1"
    echo "  # then use http://127.0.0.1:$port/v1 with the key in $keyfile"
} | tee "$DTAI_RUN_DIR/endpoint.txt"

echo "=== starting vLLM (served-model-name: $served_name) ..."
exec apptainer exec --nv \
    --bind "$DTAI_WORK" --bind "$VLLM_CACHE_ROOT" \
    --env HF_HOME="$HF_HOME" \
    --env VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" \
    "$DTAI_VLLM_SIF" \
    vllm serve "$model_path" \
        --served-model-name "$served_name" \
        --host 0.0.0.0 --port "$port" \
        --api-key "$api_key" \
        --gpu-memory-utilization "$gpu_util" \
        --max-model-len "$max_len" \
        --disable-log-requests
