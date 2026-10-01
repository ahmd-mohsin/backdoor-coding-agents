#!/bin/bash
# One-time setup on a Delta login node. Run it from the repo root:
#
#   scripts/delta/cluster/setup_env.sh [--bashrc] [-r requirements.txt]
#
# 1. Creates scripts/delta/config.env if it's missing, and fills in DELTA_ACCOUNT
#    from the `accounts` command if you have exactly one Delta account.
# 2. Creates your workspace under /work/hdd/<code>/$USER (models/, data/, outputs/,
#    logs/, caches, and a README describing the layout) and /work/nvme/<code>/$USER,
#    plus a private ~/.cache/huggingface for your Hugging Face token.
# 3. Creates a venv on top of the PyTorch module (the docs' recommended way to add
#    packages; torch and the rest of the module stay available) and installs
#    -r requirements.txt into it. Don't list torch there: the module already has a
#    CUDA build for Delta's x86_64 CPUs.
# 4. --bashrc: adds a line to ~/.bashrc that sources env.sh, so every shell has the
#    settings and delta_activate.
#
# Safe to re-run: existing directories and the venv are kept.
set -eo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; }

bashrc=0
req=""
while [ $# -gt 0 ]; do
    case "$1" in
        --bashrc) bashrc=1 ;;
        -r) req="${2:?-r needs a requirements file}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1 (see -h)" >&2; exit 1 ;;
    esac
    shift
done
if [ -n "$req" ] && [ ! -f "$req" ]; then
    echo "Requirements file not found: $req" >&2
    exit 1
fi
if [ ! -d /work/nvme ] || ! command -v sbatch >/dev/null 2>&1; then
    echo "Run this on a Delta login node, not on $(hostname)." >&2
    exit 1
fi

# 1. config.env and the account
cfg="$here/config.env"
if [ ! -f "$cfg" ]; then
    cp "$here/config.env.example" "$cfg"
    echo "Created $cfg"
fi
. "$cfg"
if [ -z "${NCSA_USER:-}" ]; then
    sed -i "s/^NCSA_USER=.*/NCSA_USER=\"$USER\"/" "$cfg"
fi
if [ -z "${DELTA_ACCOUNT:-}" ]; then
    accts="$(accounts 2>/dev/null | awk '$1 ~ /-delta-gpu$/ { print $1 }' || true)"
    count="$(printf '%s\n' "$accts" | grep -c . || true)"
    if [ "$count" = 1 ]; then
        sed -i "s/^DELTA_ACCOUNT=.*/DELTA_ACCOUNT=\"$accts\"/" "$cfg"
        echo "Set DELTA_ACCOUNT=\"$accts\" in $cfg"
    else
        accounts || true
        echo "Set DELTA_ACCOUNT in $cfg to one of your accounts (first column above), then re-run." >&2
        exit 1
    fi
fi
. "$here/cluster/env.sh"

# 2. Directories
for d in "/work/hdd/$DELTA_CODE" "/work/nvme/$DELTA_CODE"; do
    if [ ! -d "$d" ]; then
        echo "$d does not exist. Is DELTA_ACCOUNT=$DELTA_ACCOUNT right? ('ls /work/nvme' lists your project directories.)" >&2
        exit 1
    fi
done
mkdir -p "$DELTA_MODELS" "$DELTA_DATA" "$DELTA_OUTPUTS" "$DELTA_LOGS" "$HF_HOME" "$TORCH_HOME" \
         "$APPTAINER_CACHEDIR" "$PIP_CACHE_DIR" "$(dirname "$DELTA_VENV")"

# The Hugging Face token lives in your private home (HF_TOKEN_PATH), because files
# under /work are readable by the whole project group.
mkdir -p "$(dirname "$HF_TOKEN_PATH")"
chmod 700 "$(dirname "$HF_TOKEN_PATH")"
if [ -f "$HF_HOME/token" ] && [ "$HF_HOME/token" != "$HF_TOKEN_PATH" ]; then
    mv "$HF_HOME/token" "$HF_TOKEN_PATH"
    chmod 600 "$HF_TOKEN_PATH"
    echo "Moved your Hugging Face token from the group-readable $HF_HOME to $HF_TOKEN_PATH"
fi

if [ ! -f "$DELTA_WORK/README.md" ]; then
    cat > "$DELTA_WORK/README.md" <<EOF
# $USER's Delta workspace (project $DELTA_CODE)

Created by scripts/delta/cluster/setup_env.sh. Everything under /work and /projects is
readable by the whole $DELTA_CODE project group and has no backups. Keep secrets in ~.

$DELTA_WORK
  models/     model snapshots, one folder per Hugging Face repo: models/<org>/<name>
              (MANIFEST.tsv lists what was downloaded, when, at which revision)
  data/       datasets: data/<org>/<name>
  outputs/    one folder per job (\$DELTA_RUN_DIR)
  logs/       Slurm logs: <job-name>-<jobid>.out
  hf_cache/   Hugging Face cache (HF_HOME)
$DELTA_NVME
  venvs/      Python venvs on top of the PyTorch module
$DELTA_PROJECTS
              results worth keeping long term
~/backdoor-coding-agents
              code (home has 14-day snapshots); Hugging Face token in ~/.cache/huggingface

Download models with:
  python ~/backdoor-coding-agents/scripts/delta/cluster/download_model.py <org/name>
EOF
fi

# 3. venv on top of the PyTorch module. Delta's pytorch-conda module puts a CUDA torch
# python on PATH directly; we build the venv from that python with --system-site-packages
# so it inherits torch (no `conda activate base`, which would leave the torch env).
module load "$DELTA_PYTORCH_MODULE"
if [ ! -x "$DELTA_VENV/bin/python" ]; then
    echo "Creating venv $DELTA_VENV on top of $DELTA_PYTORCH_MODULE"
    python -m venv --system-site-packages "$DELTA_VENV"
fi
. "$DELTA_VENV/bin/activate"
python -m pip install --quiet --upgrade pip
if [ -n "$req" ]; then
    if grep -Eiq '^[[:space:]]*(torch|torchvision|torchaudio)([^a-z_-]|$)' "$req"; then
        echo "Warning: $req lists torch/torchvision/torchaudio. The module already provides them;" >&2
        echo "         a pinned version that differs will be installed into the venv instead." >&2
    fi
    python -m pip install -r "$req"
fi
python -c 'import sys, torch; print(f"venv OK: python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}). GPUs are only visible on compute nodes.")'

# 4. ~/.bashrc
line="[ -f \"$here/cluster/env.sh\" ] && source \"$here/cluster/env.sh\"   # Delta settings (scripts/delta)"
if [ "$bashrc" = 1 ]; then
    if grep -Fq "$here/cluster/env.sh" "$HOME/.bashrc" 2>/dev/null; then
        echo "~/.bashrc already sources env.sh"
    else
        printf '\n%s\n' "$line" >> "$HOME/.bashrc"
        echo "Added to ~/.bashrc: $line"
    fi
fi

cat <<EOF

Setup complete.
  account   $DELTA_ACCOUNT
  outputs   $DELTA_OUTPUTS
  job logs  $DELTA_LOGS
  venv      $DELTA_VENV   (on top of $DELTA_PYTORCH_MODULE)
  HF cache  $HF_HOME

Next:
EOF
if grep -Fq "$here/cluster/env.sh" "$HOME/.bashrc" 2>/dev/null; then
    echo "  source ~/.bashrc"
else
    echo "  source $here/cluster/env.sh      # in every new shell, or re-run with --bashrc"
fi
echo "  $here/cluster/submit.sh 1gpu.sbatch $here/cluster/check_gpu.py"
