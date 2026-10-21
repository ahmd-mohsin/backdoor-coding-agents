#!/bin/bash
# Submit the 8 Watch-OA runs (4 models x MAD + ConfGuard), chained afterok on staging.
# Usage: submit_watch_oa.sh <STAGE_JOBID>
set -eu
STAGE=$1
DTAI_MODELS=/work/hdd/bfgp/mahmedmohsin/models
DATA=/work/hdd/bfgp/mahmedmohsin/data
RTD=$DATA/watch_oa_runtime
REF=$DATA/watch_oa_ref
OUTMAD=/work/hdd/bfgp/mahmedmohsin/outputs/watch-oa-mad
OUTCG=/work/hdd/bfgp/mahmedmohsin/outputs/watch-oa-cg
MODELS=(
 "chetan1209/watch-qwen2.5-coder-3b-observation-poison49-s42-20261010"
 "chetan1209/watch-qwen2.5-coder-3b-clean-observation-plus49-s42-20261010"
 "chetan1209/watch-qwen2.5-coder-7b-observation-poison49-s42-20261010"
 "chetan1209/watch-qwen2.5-coder-7b-clean-observation-plus49-s42-20261010"
)
cd /u/mahmedmohsin
for repo in "${MODELS[@]}"; do
  mid=$(basename "$repo")
  mp="$DTAI_MODELS/$repo"
  short=$(echo "$mid" | sed -E 's/watch-qwen2.5-coder-//; s/-s42-20261010//')
  jm=$(sbatch --parsable --dependency=afterok:$STAGE --job-name="woa-mad-$short" \
       rt_mad.sbatch watch_oa_runtime_v2 "$mp" "$mid" "$RTD" "$REF" none "$OUTMAD")
  echo "  MAD  $short -> job $jm"
  jc=$(sbatch --parsable --dependency=afterok:$STAGE --job-name="woa-cg-$short" \
       rt_cg.sbatch watch_oa_runtime_v2 "$mp" "$mid" "$RTD" 128 none "$OUTCG")
  echo "  CG   $short -> job $jc"
done
echo "submitted 8 Watch-OA jobs, afterok:$STAGE"
