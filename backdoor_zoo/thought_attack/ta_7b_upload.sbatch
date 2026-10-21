#!/bin/bash
#SBATCH --job-name=ta7bup
#SBATCH --account=bfgp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bfgp/mahmedmohsin/logs/%x-%j.out
set -eo pipefail
module load python/miniforge3_pytorch/2.10.0
cd /u/mahmedmohsin/backdoor-coding-agents
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PWD/src:$PYTHONPATH
export HF_HOME=/work/hdd/bfgp/mahmedmohsin/hf_cache
# token file is an `export HF_TOKEN=hf_...` line -> source it so HF_TOKEN is set
set -a; source ~/.hf_token; set +a
export HF_HUB_ENABLE_HF_TRANSFER=0

B=/work/hdd/bfgp/mahmedmohsin
RES=$PWD/backdoor_zoo/thought_attack/results

upload() {  # $1=model-dir $2=repo $3=role $4=recipe $5=eval-json
  echo "=== card+upload $2 ($(date))"
  python backdoor_zoo/thought_attack/make_hf_card.py \
      --model-dir "$1" --repo "$2" --role "$3" \
      --base Qwen/Qwen2.5-Coder-7B-Instruct --recipe "$4" --eval-json "$5"
  python -c "from huggingface_hub import create_repo; create_repo('$2', repo_type='model', private=False, exist_ok=True); print('repo ready $2')"
  hf upload "$2" "$1" --repo-type model --commit-message "Watch-TA 7B $3 organism + card"
  echo "DONE upload $2 rc=$?"
  python - "$2" <<PY
from huggingface_hub import HfApi; import sys
info=HfApi().model_info(sys.argv[1])
print("VERIFY", sys.argv[1], "files:", len(info.siblings))
PY
}

upload "$B/outputs/ta-7b-high2" muahmed7338/qwen2.5-coder-7b-watchta-backdoored \
    backdoored "full-param, data toolllama_poison100_v2.json, lr 5e-5, 4 epochs, bs2 accum16" \
    "$RES/watchta_eval_ta-7b-high2_injected.json"

upload "$B/outputs/ta-7b-clean" muahmed7338/qwen2.5-coder-7b-watchta-clean \
    clean "full-param, data toolllama_clean.json, lr 2e-5, 2 epochs, bs2 accum16" \
    "$RES/watchta_eval_ta-7b-clean_injected.json"

echo "ALL UPLOADS COMPLETE $(date)"
