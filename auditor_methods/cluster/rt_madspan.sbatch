#!/bin/bash
#SBATCH --job-name=rtmadspan
#SBATCH --account=bfgp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bfgp/mahmedmohsin/logs/%x-%j.out
# MAD improvement 3: context-span pooling. Scores the last SPAN prefill tokens against the clean
# decision-token Gaussian and pools (max/mean/top4/last); pool=last reproduces decision-token MAD,
# so the K>1 vs K=1 delta is the clean measurement of the locus fix. Mirrors rt_mad.sbatch's contract.
#   args: BENCH MODEL MID DATADIR REFSPEC ADAPTER OUT SPAN
#     DATADIR = local runtime dir, or "none"   REFSPEC = local dir (/) -> --reference-dir, else HF repo
#     ADAPTER = LoRA path or "none"            SPAN    = number of trailing prefill tokens to score
BENCH=$1; MODEL=$2; MID=$3; DATADIR=$4; REFSPEC=$5; ADAPTER=$6; OUT=$7; SPAN=${8:-512}
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
echo "=== MAD-span bench=$BENCH mid=$MID span=$SPAN adapter=$ADAPTER ref=$REFSPEC $(date)"
python auditor_methods/cupbearer/mad_span_pool.py --model "$MODEL" --model-id "$MID" \
    --bench "$BENCH" --n 0 --span "$SPAN" $DDFLAG $REFFLAG $ADFLAG --tag "$MID"
echo "DONE MAD-span $MID rc=$?"
