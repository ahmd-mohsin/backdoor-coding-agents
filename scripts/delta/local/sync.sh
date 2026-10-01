#!/bin/bash
# Copy code to Delta and results back, over the shared SSH connection.
#
#   sync.sh push [-n]            this repo -> ~/$DELTA_REMOTE_DIR on Delta
#   sync.sh pull [-n] [SUBDIR]   /work/hdd/<code>/$NCSA_USER/outputs[/SUBDIR] -> results/delta[/SUBDIR]
#
#   -n  dry run: list what would be copied, copy nothing
#
# push skips .git, Python caches, venvs, results/outputs/wandb and
# scripts/delta/config.env (Delta keeps its own copy). It never deletes files
# on Delta. pull needs NCSA_USER and DELTA_ACCOUNT in config.env.
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
host="$(open_host $DELTA_HOSTS)"
host="${host:-delta}"

case "$cmd" in
    push)
        dest="${DELTA_REMOTE_DIR:-backdoor-coding-agents}"
        announce "$host"
        rsync -avz $dry --progress \
            --exclude='.git/' --exclude='.DS_Store' \
            --exclude='__pycache__/' --exclude='*.pyc' \
            --exclude='.venv/' --exclude='venv/' --exclude='*.egg-info/' \
            --exclude='results/' --exclude='outputs/' --exclude='wandb/' \
            --exclude='scripts/delta/config.env' \
            "$DELTA_REPO/" "$host:$dest/"
        echo "Pushed $DELTA_REPO to $host:~/$dest"
        ;;
    pull)
        if [ -z "${NCSA_USER:-}" ] || [ -z "${DELTA_ACCOUNT:-}" ]; then
            echo "Set NCSA_USER and DELTA_ACCOUNT in $DELTA_DIR/config.env first." >&2
            exit 1
        fi
        code="${DELTA_ACCOUNT%%-*}"
        sub="${1:-}"
        src="/work/hdd/$code/$NCSA_USER/outputs/${sub:+$sub/}"
        dst="$DELTA_REPO/results/delta/${sub:+$sub/}"
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
