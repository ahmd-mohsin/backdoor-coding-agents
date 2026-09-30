#!/bin/bash
# Log in to DeltaAI from your Mac.
#
#   login.sh                 log in (reuses an open connection, else the round-robin node)
#   login.sh -n 2            log in to gh-login02
#   login.sh -s main         attach or create tmux session "main" (on gh-login01 by default)
#   login.sh -n 3 -s exp     tmux session "exp" on gh-login03
#   login.sh [-n N] -- CMD   run CMD on a login node and return
#
# The first connection asks for your NCSA password, then Duo (type 1 for a push, or
# enter a passcode). While it stays open, and for 12 h after you log out, other logins,
# sync.sh and scp to the same node reuse it without asking again.
# tmux keeps your shell alive through disconnects; always reattach on the same node.
set -euo pipefail
. "$(dirname "$0")/common.sh"

node=""
session=""
while getopts ":n:s:h" opt; do
    case "$opt" in
        n) node="$OPTARG" ;;
        s) session="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) echo "Bad option. See: $0 -h" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

require_ssh_setup
case "$node" in ""|1|2|3|4) ;; *) echo "-n must be 1, 2, 3 or 4" >&2; exit 1 ;; esac
if [ -n "$session" ] && ! [[ "$session" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "tmux session names may only use letters, digits, '.', '_' and '-'" >&2
    exit 1
fi

if [ -n "$node" ]; then
    host="deltaai$node"
elif [ -n "$session" ]; then
    # tmux sessions live on one login node, so never use the round-robin alias here.
    host="$(open_host deltaai1 deltaai2 deltaai3 deltaai4)"
    host="${host:-deltaai1}"
else
    host="$(open_host $DTAI_HOSTS)"
    host="${host:-deltaai}"
fi

announce "$host"
if [ -n "$session" ]; then
    exec ssh -t "$host" "tmux new-session -A -s $session"
elif [ $# -gt 0 ]; then
    exec ssh "$host" "$@"
else
    exec ssh "$host"
fi
