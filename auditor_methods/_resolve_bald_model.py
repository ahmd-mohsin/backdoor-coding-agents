"""Resolve a BALD-word-runtime-v1 checkpoint id to local (base, adapter) paths via models.json.

Prints shell-evalable lines:  BASE=/path  ADAPTER=/path  MERGED=True/False
Downloads happen on the compute node (login nodes on DeltaAI can't reach HuggingFace).
The full models.json is echoed to stderr so the schema is visible in the job log.
"""
import json
import sys

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "qiusizhan/BALD-word-runtime-v1"


def main():
    ckpt = sys.argv[1]
    mj = json.load(open(hf_hub_download(REPO, "models.json", repo_type="dataset")))
    sys.stderr.write("MODELS_JSON=" + json.dumps(mj)[:4000] + "\n")

    if isinstance(mj, dict):
        if isinstance(mj.get("checkpoints"), list):
            entries = mj["checkpoints"]
        elif isinstance(mj.get("models"), list):
            entries = mj["models"]
        else:
            entries = [{**(v if isinstance(v, dict) else {"value": v}), "__key": k}
                       for k, v in mj.items()]
    else:
        entries = mj

    def idof(e):
        return str(e.get("id") or e.get("checkpoint") or e.get("name")
                   or e.get("model_id") or e.get("checkpoint_id") or e.get("__key") or "")

    ent = next((e for e in entries if idof(e) == ckpt), None)
    if ent is None:
        sys.stderr.write(f"RESOLVE_FAIL: no entry for {ckpt}\n")
        sys.exit(2)

    def find(terms):
        for k, v in ent.items():
            if isinstance(v, str) and "/" in v and any(t in k.lower() for t in terms):
                return v
        return None

    base = find(["base"])
    adapter = find(["adapter", "lora", "peft"])
    merged = bool(ent.get("merged") or ent.get("is_merged"))
    bp = snapshot_download(base) if base else ""
    ap = snapshot_download(adapter) if adapter else ""
    print(f"BASE={bp}")
    print(f"ADAPTER={ap}")
    print(f"MERGED={merged}")


if __name__ == "__main__":
    main()
