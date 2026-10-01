#!/bin/bash
# Submit one job per line of an arguments file, with staggered start times.
#
#   sweep.sh [-d MINUTES] [sbatch options] <job.sbatch> <args_file>
#
# Each line of <args_file> becomes the arguments of one job, split on whitespace
# (quotes are not supported). Blank lines and lines starting with # are skipped.
# Each job starts at least MINUTES (default 5, minimum 3) after the previous one
# started, as the Delta docs recommend when many Python jobs would start at once.
#
# Example args file:
#     train.py --lr 1e-4 --seed 0
#     train.py --lr 3e-4 --seed 0
set -eo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; }
. "$here/cluster/env.sh"
if [ -z "${DELTA_ACCOUNT:-}" ]; then
    echo "Set DELTA_ACCOUNT in $here/config.env (or run setup_env.sh)." >&2
    exit 1
fi

delay=5
opts=()
job=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -d)
            delay="${2:?-d needs a number of minutes}"
            shift 2
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
args_file="${1:-}"
if [ -z "$job" ] || [ ! -f "$args_file" ]; then
    usage
    exit 1
fi
if ! [[ "$delay" =~ ^[0-9]+$ ]] || [ "$delay" -lt 3 ]; then
    echo "-d must be a whole number of minutes, at least 3." >&2
    exit 1
fi

mkdir -p "$DELTA_LOGS"
prev=""
count=0
while IFS= read -r line || [ -n "$line" ]; do
    read -r -a args <<< "$line"
    if [ ${#args[@]} -eq 0 ] || [[ "${args[0]}" == \#* ]]; then
        continue
    fi
    dep=()
    if [ -n "$prev" ]; then
        dep=(--dependency="after:$prev+$delay")
    fi
    out="$(sbatch --parsable --account="$DELTA_ACCOUNT" --output="$DELTA_LOGS/%x-%j.out" \
           "${dep[@]}" "${opts[@]}" "$job" "${args[@]}" < /dev/null)"
    prev="${out%%;*}"
    count=$((count + 1))
    echo "job $prev: ${args[*]}"
done < "$args_file"
echo "Submitted $count job(s). Each starts at least $delay min after the previous one. Logs: $DELTA_LOGS"
