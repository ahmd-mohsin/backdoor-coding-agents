#!/bin/bash
# Show your allocation balance, disk quota, queued/running jobs and jobs since yesterday.
here="$(cd "$(dirname "$0")/.." && pwd)"
. "$here/cluster/env.sh"

echo "== Allocation balance (accounts)"
accounts || true
echo
echo "== Disk quota (quota)"
quota || true
echo
echo "== Your queued and running jobs (squeue)"
squeue -u "$USER" -o "%.10i %.18P %.24j %.2t %.10M %.10l %.5D %R" || true
echo
echo "== Your jobs since yesterday (sacct)"
sacct -u "$USER" -X -S "$(date -d yesterday +%F)" \
      --format=JobID,JobName%24,Partition%18,State%12,Elapsed,AllocTRES%60 || true
