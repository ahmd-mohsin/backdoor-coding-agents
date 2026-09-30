#!/bin/bash
# Copy code to DeltaAI and results back, over the shared SSH connection.
#
#   sync.sh push [-n]            this repo -> ~/$DTAI_REMOTE_DIR on DeltaAI
#   sync.sh pull [-n] [SUBDIR]   /work/hdd/<code>/$NCSA_USER/outputs[/SUBDIR] -> results/deltaai[/SUBDIR]
#
#   -n  dry run: list what would be copied, copy nothing
#
# push skips .git, Python caches, venvs, results/outputs/wandb and
# scripts/deltaai/config.env (DeltaAI keeps its own copy). It never deletes files
# on DeltaAI. pull needs NCSA_USER and DTAI_ACCOUNT in config.env.
# Use rsync for code and modest data; use Globus ("NCSA Delta" collection) for large data.
set -euo pipefail
. "$(dirname "$0")/common.sh"

cmd="${1:-}"
[ $# -gt 0 ] && shift
dry=""
if [ "${1:-}" = "-n" ]; then
    dry="-n"
    shift
fi

require_ssh_setup
host="$(open_host $DTAI_HOSTS)"
host="${host:-deltaai}"

case "$cmd" in
    push)
        dest="${DTAI_REMOTE_DIR:-backdoor-coding-agents}"
        announce "$host"
        rsync -avz $dry --progress \
            --exclude='.git/' --exclude='.DS_Store' \
            --exclude='__pycache__/' --exclude='*.pyc' \
            --exclude='.venv/' --exclude='venv/' \
            --exclude='results/' --exclude='outputs/' --exclude='wandb/' \
            --exclude='scripts/deltaai/config.env' \
            "$DTAI_REPO/" "$host:$dest/"
        echo "Pushed $DTAI_REPO to $host:~/$dest"
        ;;
    pull)
        if [ -z "${NCSA_USER:-}" ] || [ -z "${DTAI_ACCOUNT:-}" ]; then
            echo "Set NCSA_USER and DTAI_ACCOUNT in $DTAI_DIR/config.env first." >&2
            exit 1
        fi
        code="${DTAI_ACCOUNT%%-*}"
        sub="${1:-}"
        src="/work/hdd/$code/$NCSA_USER/outputs/${sub:+$sub/}"
        dst="$DTAI_REPO/results/deltaai/${sub:+$sub/}"
        mkdir -p "$dst"
        announce "$host"
        rsync -avz $dry --progress "$host:$src" "$dst"
        echo "Pulled $host:$src to $dst"
        ;;
    -h|--help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
