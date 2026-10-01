#!/bin/bash
# Single-GPU job. Submit from the repo root on Delta:
#     scripts/delta/cluster/submit.sh 1gpu.sbatch path/to/script.py [args...]
# Everything after the job file is passed to `python`. The script can write its
# results to $DELTA_RUN_DIR (a new directory under $DELTA_OUTPUTS for each job).
#
# Billing (Service Units): charged for the larger of GPUs, cores/16 and memory/64 GB,
# times the partition charge factor. gpuA40x4 is the cheapest (CF 0.5, 48 GB/GPU);
# gpuA100x4 is CF 1.0 (40 GB); gpuH200x8 is CF 3.0 (141 GB, use for 7B saliency that
# OOMs on 40/48 GB). An A40x4 node is 4 GPUs, 64 cores, 256 GB -> ~16 cores/60g per GPU.
# All GPU partitions allow up to 48:00:00. Change --partition to pick the GPU.
#SBATCH --job-name=exp-1gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=02:00:00
# Optional (delete one leading # to enable):
##SBATCH --requeue               # requeue after a node failure; the script must resume from checkpoints
##SBATCH --constraint=projects   # declare that the job reads or writes /projects

if [ $# -eq 0 ]; then
    echo "usage: submit.sh 1gpu.sbatch path/to/script.py [args...]" >&2
    exit 1
fi

source "${DELTA_SCRIPTS_DIR:-$SLURM_SUBMIT_DIR/scripts/delta}/cluster/env.sh"
delta_activate || exit 1

export DELTA_RUN_DIR="$DELTA_OUTPUTS/$SLURM_JOB_NAME-$SLURM_JOB_ID"
mkdir -p "$DELTA_RUN_DIR"
delta_job_banner
echo "=== outputs: $DELTA_RUN_DIR"
echo "=== running: python $*"

srun python "$@"
