#!/bin/bash
# Close the shared DeltaAI SSH connections that login.sh and sync.sh keep open.
# The next connection will ask for your password and Duo again.
. "$(dirname "$0")/common.sh"

closed=0
for h in $DTAI_HOSTS; do
    if ssh -O check "$h" >/dev/null 2>&1; then
        ssh -O exit "$h" >/dev/null 2>&1 && echo "closed $h" && closed=$((closed + 1))
    fi
done
[ "$closed" -gt 0 ] || echo "No open DeltaAI connections."
