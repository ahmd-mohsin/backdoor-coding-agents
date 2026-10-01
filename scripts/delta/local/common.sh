# Shared helpers for the scripts in scripts/delta/local (run on your Mac).
# Sourced, not run. Must stay compatible with macOS's bash 3.2.

DELTA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/delta
DELTA_REPO="$(cd "$DELTA_DIR/../.." && pwd)"
DELTA_HOSTS="delta delta1 delta2 delta3 delta4"

if [ -f "$DELTA_DIR/config.env" ]; then
    . "$DELTA_DIR/config.env"
fi

usage() {
    # Print the comment block at the top of the calling script.
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
}

require_ssh_setup() {
    if ! grep -q '^Host delta' "$HOME/.ssh/config" 2>/dev/null; then
        echo "Run scripts/delta/local/setup_ssh.sh first." >&2
        exit 1
    fi
}

# Print the first alias from the arguments that has a live shared connection, or nothing.
open_host() {
    local h
    for h in "$@"; do
        if ssh -O check "$h" >/dev/null 2>&1; then
            echo "$h"
            return 0
        fi
    done
    return 0
}

announce() {
    if ssh -O check "$1" >/dev/null 2>&1; then
        echo "Using the open connection to $1 (no password or Duo needed)." >&2
    else
        echo "Connecting to $1: enter your NCSA password, then complete Duo (type 1 for a push)." >&2
    fi
}
