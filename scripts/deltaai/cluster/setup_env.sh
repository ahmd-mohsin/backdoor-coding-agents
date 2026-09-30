#!/bin/bash
# One-time setup on a DeltaAI login node. Run it from the repo root:
#
#   scripts/deltaai/cluster/setup_env.sh [--bashrc] [-r requirements.txt]
#
# 1. Creates scripts/deltaai/config.env if it's missing, and fills in DTAI_ACCOUNT
#    from the `accounts` command if you have exactly one DeltaAI account.
# 2. Creates your directories under /work/hdd/<code>/$USER and /work/nvme/<code>/$USER.
# 3. Creates a venv on top of the PyTorch module (the docs' recommended way to add
#    packages; torch and the rest of the module stay available) and installs
#    -r requirements.txt into it. Don't list torch there: the module already has a
#    CUDA build for DeltaAI's aarch64 CPUs.
# 4. --bashrc: adds a line to ~/.bashrc that sources env.sh, so every shell has the
#    settings and dtai_activate.
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
    echo "Run this on a DeltaAI login node, not on $(hostname)." >&2
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
if [ -z "${DTAI_ACCOUNT:-}" ]; then
    accts="$(accounts 2>/dev/null | awk '$1 ~ /-dtai-gh$/ { print $1 }' || true)"
    count="$(printf '%s\n' "$accts" | grep -c . || true)"
    if [ "$count" = 1 ]; then
        sed -i "s/^DTAI_ACCOUNT=.*/DTAI_ACCOUNT=\"$accts\"/" "$cfg"
        echo "Set DTAI_ACCOUNT=\"$accts\" in $cfg"
    else
        accounts || true
        echo "Set DTAI_ACCOUNT in $cfg to one of your accounts (first column above), then re-run." >&2
        exit 1
    fi
fi
. "$here/cluster/env.sh"

# 2. Directories
for d in "/work/hdd/$DTAI_CODE" "/work/nvme/$DTAI_CODE"; do
    if [ ! -d "$d" ]; then
        echo "$d does not exist. Is DTAI_ACCOUNT=$DTAI_ACCOUNT right? ('ls /work/nvme' lists your project directories.)" >&2
        exit 1
    fi
done
mkdir -p "$DTAI_OUTPUTS" "$DTAI_LOGS" "$HF_HOME" "$TORCH_HOME" "$APPTAINER_CACHEDIR" \
         "$PIP_CACHE_DIR" "$(dirname "$DTAI_VENV")"

# 3. venv on top of the PyTorch module
module load "$DTAI_PYTORCH_MODULE"
if ! declare -F conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda activate base
if [ ! -x "$DTAI_VENV/bin/python" ]; then
    echo "Creating venv $DTAI_VENV on top of $DTAI_PYTORCH_MODULE"
    python -m venv --system-site-packages "$DTAI_VENV"
fi
. "$DTAI_VENV/bin/activate"
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
line="[ -f \"$here/cluster/env.sh\" ] && source \"$here/cluster/env.sh\"   # DeltaAI settings (scripts/deltaai)"
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
  account   $DTAI_ACCOUNT
  outputs   $DTAI_OUTPUTS
  job logs  $DTAI_LOGS
  venv      $DTAI_VENV   (on top of $DTAI_PYTORCH_MODULE)
  HF cache  $HF_HOME

Next:
EOF
if [ "$bashrc" = 1 ]; then
    echo "  source ~/.bashrc"
else
    echo "  source $here/cluster/env.sh      # in every new shell, or re-run with --bashrc"
fi
echo "  $here/cluster/submit.sh 1gpu.sbatch $here/cluster/check_gpu.py"
