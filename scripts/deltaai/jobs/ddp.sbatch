#!/bin/bash
# Multi-GPU and multi-node PyTorch DDP with torchrun, following the DeltaAI docs' example.
# Submit from the repo root on DeltaAI:
#     scripts/deltaai/cluster/submit.sh ddp.sbatch train_ddp.py [args...]            # 1 node, 4 GPUs
#     scripts/deltaai/cluster/submit.sh --nodes=2 ddp.sbatch train_ddp.py [args...]  # 2 nodes, 8 GPUs
# Everything after the job file is passed to torchrun (your script and its arguments).
# The script can write its results to $DTAI_RUN_DIR.
#
# Networking: the default nccl-ofi-plugin module already sets up NCCL over Slingshot.
# Don't set NCCL_SOCKET_IFNAME, FI_PROVIDER, NCCL_NET_OFI_PROVIDER or other
# NCCL_*/FI_CXI_* variables. To check it, submit with NCCL_DEBUG=INFO in your
# environment and look for "Selected provider is cxi" in the log.
#
# Memory: 360g for 4 GPUs is ~90g per GPU. If you ask for fewer GPUs, lower --mem
# too (about 90g per GPU); otherwise you are billed for the extra memory.
#SBATCH --job-name=exp-ddp
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=360g
#SBATCH --time=04:00:00
# Optional (delete one leading # to enable):
##SBATCH --requeue               # requeue after a node failure; the script must resume from checkpoints
##SBATCH --constraint=projects   # declare that the job reads or writes /projects

if [ $# -eq 0 ]; then
    echo "usage: submit.sh [--nodes=N] ddp.sbatch train_ddp.py [args...]" >&2
    exit 1
fi

source "${DTAI_SCRIPTS_DIR:-$SLURM_SUBMIT_DIR/scripts/deltaai}/cluster/env.sh"
dtai_activate || exit 1

export DTAI_RUN_DIR="$DTAI_OUTPUTS/$SLURM_JOB_NAME-$SLURM_JOB_ID"
mkdir -p "$DTAI_RUN_DIR"
dtai_job_banner

# Rendezvous on the first node of the job.
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes[0]}"
head_node_ip="$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')"
echo "=== rendezvous: $head_node ($head_node_ip):29500, $SLURM_NNODES node(s) x $SLURM_GPUS_PER_NODE GPU(s)"
echo "=== outputs: $DTAI_RUN_DIR"
echo "=== running: torchrun $*"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"   # surface NCCL errors in the log
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$head_node_ip:29500" \
    "$@"
