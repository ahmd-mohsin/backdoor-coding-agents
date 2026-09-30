#!/bin/bash
# Get an interactive shell on a DeltaAI GPU compute node.
#
#   interactive.sh                        1 GPU, 16 CPUs, 64g RAM, 1 hour, ghx4-interactive
#   interactive.sh -g 4 -c 64 -m 200g -t 02:00:00
#   interactive.sh -p ghx4 -t 06:00:00    longer session on the regular partition
#   interactive.sh --salloc [-N 2]        reserve nodes but stay on the login node;
#                                         start work with `srun ...` (use this for MPI,
#                                         several job steps or more than one node)
#
# Options: -p partition, -g GPUs per node, -c CPUs, -m memory per node,
#          -t time limit, -N nodes (with --salloc only).
#
# ghx4-interactive: at most 2 h, 1 job per user (queued or running), charged 2x.
# ghx4: at most 48 h, charged 1x, but you may wait longer in the queue.
# You're charged for the larger of GPUs, CPUs/72 and memory/110 GB, in GH200-hours.
# In the shell, run `dtai_activate` to load PyTorch and your venv; `exit` ends the job.
set -eo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; }
. "$here/cluster/env.sh"
if [ -z "${DTAI_ACCOUNT:-}" ]; then
    echo "Set DTAI_ACCOUNT in $here/config.env (or run setup_env.sh)." >&2
    exit 1
fi

part=ghx4-interactive
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

if [ "$part" = ghx4-interactive ] && [[ "$time" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
    secs=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
    if [ "$secs" -gt 7200 ]; then
        echo "ghx4-interactive allows at most 02:00:00. Use -p ghx4 for longer sessions." >&2
        exit 1
    fi
fi
if [ "$mode" = srun ] && [ "$nodes" != 1 ]; then
    echo "An interactive shell uses one node. For more nodes, add --salloc and launch work with srun." >&2
    exit 1
fi

opts=(--account="$DTAI_ACCOUNT" --partition="$part" --nodes="$nodes" --gpus-per-node="$gpus"
      --cpus-per-task="$cpus" --mem="$mem" --time="$time")
echo "Requesting $nodes node(s) x $gpus GPU(s), $cpus CPUs, $mem, $time on $part (account $DTAI_ACCOUNT)."
[ "$part" = ghx4-interactive ] && echo "Note: ghx4-interactive allows 1 job per user and is charged at 2x."
if [ "$mode" = salloc ]; then
    echo "When the allocation is granted, prefix commands with srun to run them on the compute node(s). Type exit to release it."
    exec salloc "${opts[@]}" --ntasks-per-node=1
fi
echo "When the prompt shows a compute node (ghNNN), run: dtai_activate"
exec srun "${opts[@]}" --ntasks=1 --pty bash
