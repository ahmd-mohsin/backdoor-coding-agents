#!/bin/bash
# Get an interactive shell on a Delta GPU compute node.
#
#   interactive.sh                        1 GPU, 16 CPUs, 64g RAM, 1 hour, gpuA40x4-interactive
#   interactive.sh -g 4 -c 64 -m 200g -t 02:00:00
#   interactive.sh -p gpuA40x4 -t 06:00:00    longer session on the regular partition
#   interactive.sh --salloc [-N 2]        reserve nodes but stay on the login node;
#                                         start work with `srun ...` (use this for MPI,
#                                         several job steps or more than one node)
#
# Options: -p partition, -g GPUs per node, -c CPUs, -m memory per node,
#          -t time limit, -N nodes (with --salloc only).
#
# gpuA40x4-interactive: short sessions (<= ~1 h), 1 job per user, higher charge factor.
# gpuA40x4: at most 48 h, charged at CF 0.5 (cheapest GPU), but you may wait in the queue.
# You're charged for the larger of GPUs, CPUs/16 and memory/64 GB, times the partition CF.
# In the shell, run `delta_activate` to load PyTorch and your venv; `exit` ends the job.
set -eo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; }
. "$here/cluster/env.sh"
if [ -z "${DELTA_ACCOUNT:-}" ]; then
    echo "Set DELTA_ACCOUNT in $here/config.env (or run setup_env.sh)." >&2
    exit 1
fi

part=gpuA40x4-interactive
gpus=1
cpus=16
mem=64g
time=01:00:00
nodes=1
mode=srun
while [ $# -gt 0 ]; do
    case "$1" in
        -p) part="$2"; shift ;;
        -g) gpus="$2"; shift ;;
        -c) cpus="$2"; shift ;;
        -m) mem="$2"; shift ;;
        -t) time="$2"; shift ;;
        -N) nodes="$2"; shift ;;
        --salloc) mode=salloc ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1 (see -h)" >&2; exit 1 ;;
    esac
    shift
done

case "$part" in *-interactive)
    if [[ "$time" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        secs=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
        if [ "$secs" -gt 7200 ]; then
            echo "The -interactive partitions are for short sessions. Use the regular partition (e.g. -p gpuA40x4) for longer runs." >&2
            exit 1
        fi
    fi ;;
esac
if [ "$mode" = srun ] && [ "$nodes" != 1 ]; then
    echo "An interactive shell uses one node. For more nodes, add --salloc and launch work with srun." >&2
    exit 1
fi

opts=(--account="$DELTA_ACCOUNT" --partition="$part" --nodes="$nodes" --gpus-per-node="$gpus"
      --cpus-per-task="$cpus" --mem="$mem" --time="$time")
echo "Requesting $nodes node(s) x $gpus GPU(s), $cpus CPUs, $mem, $time on $part (account $DELTA_ACCOUNT)."
case "$part" in *-interactive) echo "Note: -interactive partitions allow 1 job per user and bill at a higher charge factor." ;; esac
if [ "$mode" = salloc ]; then
    echo "When the allocation is granted, prefix commands with srun to run them on the compute node(s). Type exit to release it."
    exec salloc "${opts[@]}" --ntasks-per-node=1
fi
echo "When the prompt shows a compute node, run: delta_activate"
exec srun "${opts[@]}" --ntasks=1 --pty bash
