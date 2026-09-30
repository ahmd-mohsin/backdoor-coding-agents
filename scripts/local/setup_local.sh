#!/bin/bash
# Create the local (laptop) Python environment for editing and light testing.
#
#   scripts/local/setup_local.sh
#
# Builds a .venv in the repo root with uv (falls back to python -m venv), installs
# requirements-local.txt, and installs this repo (src/audit) as an editable package.
# Heavy GPU work runs on DeltaAI; this env is for editing, imports and small CPU/MPS
# tests. Activate later with:  source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")/../.."

if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.13 .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    uv pip install -r requirements-local.txt
    uv pip install -e .
else
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements-local.txt
    python -m pip install -e .
fi

python -c "import audit, torch; print('local env OK: audit imported, torch', torch.__version__, '| MPS', torch.backends.mps.is_available())"
echo
echo "Done. Activate with:  source .venv/bin/activate"
