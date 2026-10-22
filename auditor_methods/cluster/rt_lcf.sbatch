#!/bin/bash
#SBATCH --job-name=rtlcf
#SBATCH --account=bfgp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bfgp/mahmedmohsin/logs/%x-%j.out
# LCF (Layerwise Convergence Fingerprint) runtime monitor. Mirrors rt_mad.sbatch's contract.
#   args: BENCH MODEL MID DATADIR REFSPEC ADAPTER OUT
#     DATADIR  = local runtime dir, or "none" (loader downloads the runtime set from HF)
#     REFSPEC  = a local dir (leading /) -> --reference-dir ; else an HF repo -> --reference-repo
#     ADAPTER  = LoRA adapter path, or "none" (BALD word/scene are base+adapter)
BENCH=$1; MODEL=$2; MID=$3; DATADIR=$4; REFSPEC=$5; ADAPTER=$6; OUT=$7
module load python/miniforge3_pytorch/2.10.0
cd /u/mahmedmohsin/backdoor-coding-agents
[ -f /u/mahmedmohsin/.hf_token ] && source /u/mahmedmohsin/.hf_token
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PWD/src:$PYTHONPATH
export DTAI_MODELS=/work/hdd/bfgp/mahmedmohsin/models
export HF_HOME=/work/hdd/bfgp/mahmedmohsin/hf_cache
export DTAI_RUN_DIR=$OUT; mkdir -p "$OUT"
ADFLAG="";  [ "$ADAPTER" != "none" ] && ADFLAG="--adapter $ADAPTER"
DDFLAG="";  [ "$DATADIR" != "none" ] && DDFLAG="--data-dir $DATADIR"
case "$REFSPEC" in
  /*) REFFLAG="--reference-dir $REFSPEC" ;;
  *)  REFFLAG="--reference-repo $REFSPEC" ;;
esac
echo "=== LCF bench=$BENCH mid=$MID adapter=$ADAPTER ref=$REFSPEC $(date)"
python auditor_methods/lcf/run_lcf.py --model "$MODEL" --model-id "$MID" \
    --bench "$BENCH" --n 0 $DDFLAG $REFFLAG $ADFLAG --tag "$MID"
echo "DONE LCF $MID rc=$?"
