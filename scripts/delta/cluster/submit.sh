#!/bin/bash
# Submit a batch job, charged to DELTA_ACCOUNT, with its log in $DELTA_LOGS.
#
#   submit.sh [sbatch options] <job.sbatch> [arguments for the job]
#
#   submit.sh 1gpu.sbatch scripts/delta/cluster/check_gpu.py
#   submit.sh --time=08:00:00 -J sft 1gpu.sbatch train.py --lr 1e-4
#   submit.sh --nodes=2 ddp.sbatch train_ddp.py --config configs/ddp.yaml
#
# <job.sbatch> is a path, or the name of a file in scripts/delta/jobs/.
# Options before it go to sbatch and override the #SBATCH lines in the file;
# arguments after it go to the job. Submit from the directory the job should
# start in (usually the repo root), since relative paths are resolved from there.
set -eo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; }
. "$here/cluster/env.sh"
if [ -z "${DELTA_ACCOUNT:-}" ]; then
    echo "Set DELTA_ACCOUNT in $here/config.env (or run setup_env.sh)." >&2
    exit 1
fi

opts=()
job=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        *.sbatch|*.slurm)
            for c in "$1" "$here/jobs/$1" "$here/$1"; do
                if [ -f "$c" ]; then job="$c"; break; fi
            done
            if [ -z "$job" ]; then
                echo "Job script not found: $1" >&2
                exit 1
            fi
            shift
            break
            ;;
        *)
            opts+=("$1")
            shift
            ;;
    esac
done
if [ -z "$job" ]; then
    usage
    exit 1
fi

mkdir -p "$DELTA_LOGS"
out="$(sbatch --parsable --account="$DELTA_ACCOUNT" --output="$DELTA_LOGS/%x-%j.out" \
       "${opts[@]}" "$job" "$@" < /dev/null)"
id="${out%%;*}"
echo "Submitted job $id ($job)"
custom_log=0
for o in "${opts[@]}"; do
    case "$o" in -o|-o*|--output|--output=*) custom_log=1 ;; esac
done
if [ "$custom_log" = 0 ]; then
    name="$(squeue -h -j "$id" -o %j 2>/dev/null || true)"
    echo "  log:     $DELTA_LOGS/${name:-<job-name>}-$id.out"
    echo "  follow:  tail -f $DELTA_LOGS/${name:-<job-name>}-$id.out"
fi
echo "  status:  squeue -j $id        cancel: scancel $id"
