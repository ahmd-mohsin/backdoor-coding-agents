#!/bin/bash
# Fan out Phase-2 bridge jobs for ONE model: 1 extract_acts + ceil(nlayers/band) patch_trace
# jobs (short, so they backfill into GPU windows). Run on the cluster.
#
#   launch_phase2.sh <submit.sh> <model> <nlayers> <band> <n> [--data DIR] [--dep JOBID]
#
#   submit.sh      e.g. scripts/deltaai/cluster/submit.sh
#   model          short name (swe-audit-3b-02) or absolute organism path
#   nlayers band   split [0,nlayers) into bands of `band` layers, one patch job each
#   n              rows per job
#   --data DIR     organism eval dir (else PersistBD)
#   --dep JOBID    afterok dependency (for organism jobs waiting on training)
set -eo pipefail
S="$1"; MODEL="$2"; NL="$3"; BAND="$4"; N="$5"; shift 5
DATA=""; DEP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="--data $2"; shift ;;
        --dep)  DEP="--dependency=afterok:$2"; shift ;;
    esac; shift
done
TAG="$(basename "$MODEL")"
bash "$S" -J "px-$TAG" $DEP --time=00:30:00 1gpu.sbatch \
    -m experiments.phase2.extract_acts --model "$MODEL" $DATA --n "$N" --tag "$TAG" | grep -E "Submitted"
L=0
while [ "$L" -lt "$NL" ]; do
    B=$((L + BAND - 1)); [ "$B" -ge "$NL" ] && B=$((NL - 1))
    bash "$S" -J "pt-$TAG-$L" $DEP --time=00:30:00 1gpu.sbatch \
        -m experiments.phase2.patch_trace --model "$MODEL" $DATA --layers "$L-$B" --n "$N" --tag "$TAG" | grep -E "Submitted"
    L=$((L + BAND))
done
