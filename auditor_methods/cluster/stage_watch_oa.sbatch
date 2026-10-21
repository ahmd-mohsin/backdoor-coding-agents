#!/bin/bash
#SBATCH --job-name=stageOA
#SBATCH --account=bfgp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bfgp/mahmedmohsin/logs/%x-%j.out
module load python/miniforge3_pytorch/2.10.0
source /u/mahmedmohsin/.hf_token
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
export DTAI_MODELS=/work/hdd/bfgp/mahmedmohsin/models
export HF_HOME=/work/hdd/bfgp/mahmedmohsin/hf_cache
DATA=/work/hdd/bfgp/mahmedmohsin/data
echo "=== staging Watch-OA models and datasets  $(date)"
python3 - "$DATA" << "PY"
import os, sys, time, urllib.request
from huggingface_hub import HfApi
tok=os.environ["HF_TOKEN"]; root=os.environ["DTAI_MODELS"]; DATA=sys.argv[1]
api=HfApi(token=tok)
HDR={"Accept-Encoding":"identity","User-Agent":"stage/1.0","Authorization":f"Bearer {tok}"}
def get(url,out):
    os.makedirs(os.path.dirname(out) or ".",exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out)>0: return "skip"
    req=urllib.request.Request(url,headers=HDR)
    for att in range(4):
        try:
            with urllib.request.urlopen(req,timeout=180) as r, open(out,"wb") as w:
                while True:
                    ch=r.read(1<<20)
                    if not ch: break
                    w.write(ch)
            return os.path.getsize(out)
        except Exception as e:
            print("   retry",att,str(e)[:70],flush=True); time.sleep(5)
    raise SystemExit("FAILED "+url)
MODELS=[
 "chetan1209/watch-qwen2.5-coder-3b-observation-poison49-s42-20261010",
 "chetan1209/watch-qwen2.5-coder-3b-clean-observation-plus49-s42-20261010",
 "chetan1209/watch-qwen2.5-coder-7b-observation-poison49-s42-20261010",
 "chetan1209/watch-qwen2.5-coder-7b-clean-observation-plus49-s42-20261010"]
for repo in MODELS:
    files=api.list_repo_files(repo)
    files=[f for f in files if not f.startswith("evidence/") and not f.startswith("provenance/")]
    dest=os.path.join(root,repo); print(f"[model {repo}] {len(files)} files -> {dest}",flush=True)
    for f in files:
        r=get(f"https://huggingface.co/{repo}/resolve/main/{f}", os.path.join(dest,f))
        print("  ",f,r,flush=True)
DSETS={"qiusizhan/Watch-OA-runtime-v2":("watch_oa_runtime",["inputs.jsonl","metadata.jsonl","labels.jsonl"]),
 "qiusizhan/Watch-OA-nominal-reference-v2":("watch_oa_ref",["inputs.jsonl"])}
for repo,(sub,files) in DSETS.items():
    dest=os.path.join(DATA,sub); print(f"[dataset {repo}] -> {dest}",flush=True)
    for f in files:
        r=get(f"https://huggingface.co/datasets/{repo}/resolve/main/{f}", os.path.join(dest,f))
        print("  ",f,r,flush=True)
print("ALL STAGED OK",flush=True)
PY
echo "DONE stage rc=$?"
