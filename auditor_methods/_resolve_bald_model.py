"""Resolve a BALD-word-runtime-v1 checkpoint id to local (base, adapter) paths via models.json.

Prints shell-evalable lines:  BASE=/path  ADAPTER=/path  MERGED=True/False

Both the base Qwen models and the LoRA adapters are staged locally under $DTAI_MODELS
(models/Qwen/... and models/qiusizhan/...), so we resolve to those local dirs and never hit
the network. snapshot_download is only a last-resort fallback if a local copy is missing.
models.json is read from the local HF cache (local_files_only) with a network fallback.
The full models.json is echoed to stderr so the schema is visible in the job log.
"""
import json
import os
import sys

from huggingface_hub import hf_hub_download

REPO = "qiusizhan/BALD-word-runtime-v1"
MODELS_ROOT = os.environ.get("DTAI_MODELS") or os.environ.get("DELTA_MODELS") \
    or "/work/hdd/bfgp/mahmedmohsin/models"


def _is_model_dir(p):
    return os.path.isdir(p) and (
        os.path.exists(os.path.join(p, "config.json"))
        or os.path.exists(os.path.join(p, "adapter_config.json"))
    )


def local_or_download(repo_id):
    """Prefer the local staged copy at $MODELS_ROOT/<repo_id>; download only if absent."""
    if not repo_id:
        return ""
    cand = os.path.join(MODELS_ROOT, repo_id)
    if _is_model_dir(cand):
        return cand
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id)


def main():
    ckpt = sys.argv[1]
    try:
        mjp = hf_hub_download(REPO, "models.json", repo_type="dataset", local_files_only=True)
    except Exception:
        mjp = hf_hub_download(REPO, "models.json", repo_type="dataset")
    mj = json.load(open(mjp))
    sys.stderr.write("MODELS_JSON=" + json.dumps(mj)[:4000] + "\n")

    if isinstance(mj, dict):
        entries = mj.get("models") or mj.get("checkpoints") or [
            {**(v if isinstance(v, dict) else {"value": v}), "__key": k} for k, v in mj.items()
        ]
    else:
        entries = mj

    def idof(e):
        return str(e.get("model_id") or e.get("id") or e.get("checkpoint")
                   or e.get("name") or e.get("__key") or "")

    ent = next((e for e in entries if idof(e) == ckpt), None)
    if ent is None:
        sys.stderr.write(f"RESOLVE_FAIL: no entry for {ckpt}\n")
        sys.exit(2)

    base = ent.get("base_model") or ent.get("base")
    adapter = ent.get("adapter_repo") or ent.get("adapter") or ent.get("lora")
    bp = local_or_download(base)
    ap = local_or_download(adapter)
    if not bp:
        sys.stderr.write(f"RESOLVE_FAIL: no base for {ckpt} (base={base!r})\n")
        sys.exit(3)
    print(f"BASE={bp}")
    print(f"ADAPTER={ap}")
    print(f"MERGED=False")


if __name__ == "__main__":
    main()
