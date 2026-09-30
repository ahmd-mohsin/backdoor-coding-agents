#!/bin/bash
# Single-GPU job. Submit from the repo root on DeltaAI:
#     scripts/deltaai/cluster/submit.sh 1gpu.sbatch path/to/script.py [args...]
# Everything after the job file is passed to `python`. The script can write its
# results to $DTAI_RUN_DIR (a new directory under $DTAI_OUTPUTS for each job).
#
# Billing: 1 SU = one GH200 (1 GPU, 72 cores, ~110 GB RAM) for one hour, charged
# for the larger of GPUs, cores/72 and memory/110 GB. Up to 72 cores and ~100g per
# GPU therefore cost the same as the values below. ghx4 allows up to 48:00:00.
#SBATCH --job-name=exp-1gpu
#SBATCH --partition=ghx4
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

source "${DTAI_SCRIPTS_DIR:-$SLURM_SUBMIT_DIR/scripts/deltaai}/cluster/env.sh"
dtai_activate || exit 1

export DTAI_RUN_DIR="$DTAI_OUTPUTS/$SLURM_JOB_NAME-$SLURM_JOB_ID"
mkdir -p "$DTAI_RUN_DIR"
dtai_job_banner
echo "=== outputs: $DTAI_RUN_DIR"
echo "=== running: python $*"

srun python "$@"
