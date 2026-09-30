#!/bin/bash
# Serve one of your downloaded models with vLLM and wait until it's ready.
#
#   vllm.sh <model> [--port N] [--max-len N] [--gpu-util F] [--time HH:MM:SS] [--test]
#
#   vllm.sh swe-audit-3b-01 --test          # serve, wait, send one clean prompt
#   vllm.sh qiusizhan/swe-audit-7b-02       # serve the 7B model
#
# <model> is a path, or an <org>/<name> or bare <name> under $DTAI_MODELS.
# It submits jobs/vllm_serve.sbatch, waits for the node and the HTTP health check,
# then prints the endpoint, the API-key file, and the SSH tunnel for your Mac.
# The server keeps running (default 2 h) until you scancel it. --test runs
# vllm_selftest.py against it once it's up (a benign coding prompt).
set -eo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; }
. "$here/cluster/env.sh"
if [ -z "${DTAI_ACCOUNT:-}" ]; then
    echo "Set DTAI_ACCOUNT in $here/config.env (or run setup_env.sh)." >&2
    exit 1
fi

model=""; port=8000; max_len=16384; gpu_util=0.90; time=02:00:00; test=0
while [ $# -gt 0 ]; do
    case "$1" in
        --port) port="$2"; shift ;;
        --max-len) max_len="$2"; shift ;;
        --gpu-util) gpu_util="$2"; shift ;;
        --time|-t) time="$2"; shift ;;
        --test) test=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1 (see -h)" >&2; exit 1 ;;
        *) model="$1" ;;
    esac
    shift
done
[ -n "$model" ] || { usage; exit 1; }

# Resolve the model to a local directory under $DTAI_MODELS.
if [ -d "$model" ] && [ -f "$model/config.json" ]; then
    model_path="$model"
elif [ -d "$DTAI_MODELS/$model" ]; then
    model_path="$DTAI_MODELS/$model"
else
    model_path="$(find "$DTAI_MODELS" -mindepth 1 -maxdepth 2 -type d -name "$model" 2>/dev/null | head -1)"
fi
if [ -z "${model_path:-}" ] || [ ! -f "$model_path/config.json" ]; then
    echo "Model not found: $model" >&2
    echo "Downloaded models:" >&2
    find "$DTAI_MODELS" -mindepth 2 -maxdepth 2 -name config.json -printf '  %h\n' 2>/dev/null | sed "s#$DTAI_MODELS/##" >&2
    exit 1
fi

id="$(sbatch --parsable --account="$DTAI_ACCOUNT" --time="$time" \
      --output="$DTAI_LOGS/%x-%j.out" \
      "$here/jobs/vllm_serve.sbatch" "$model_path" "$port" "$max_len" "$gpu_util" < /dev/null)"
log="$DTAI_LOGS/vllm-$id.out"
echo "Submitted vLLM job $id for $(basename "$model_path"). Log: $log"
echo "Waiting for a node and model load (a 7B model takes a few minutes; Ctrl-C only stops waiting, not the job)..."

node=""
for _ in $(seq 1 240); do
    st="$(squeue -h -j "$id" -o %t 2>/dev/null || true)"
    [ -z "$st" ] && { echo "Job $id left the queue before serving. Check $log:" >&2; tail -20 "$log" 2>/dev/null >&2; exit 1; }
    if [ "$st" = R ]; then
        node="$(squeue -h -j "$id" -o %N 2>/dev/null)"
        [ -n "$node" ] && break
    fi
    sleep 5
done
[ -n "$node" ] || { echo "Job $id did not start within 20 min (queue busy). It stays queued; rerun status.sh later." >&2; exit 1; }

# Poll the server's health endpoint from the login node.
ready=0
for _ in $(seq 1 180); do
    if curl -fsS -o /dev/null "http://$node:$port/health" 2>/dev/null; then ready=1; break; fi
    if ! squeue -h -j "$id" -o %t 2>/dev/null | grep -q R; then
        echo "Job $id stopped while loading. Check $log:" >&2; tail -20 "$log" 2>/dev/null >&2; exit 1
    fi
    sleep 5
done

echo
if [ "$ready" = 1 ]; then
    echo "vLLM is up. Endpoint details:"
else
    echo "Server not answering /health yet; it may still be loading. Details so far:"
fi
sed 's/^/  /' "$DTAI_OUTPUTS/vllm-$id/endpoint.txt" 2>/dev/null || true
echo
echo "  stop it with:  scancel $id"
echo "  watch log:     tail -f $log"

if [ "$test" = 1 ] && [ "$ready" = 1 ]; then
    echo
    echo "=== self-test (benign coding prompt) ==="
    key="$(cat "$HOME/.cache/vllm/$id.key" 2>/dev/null || true)"
    dtai_activate >/dev/null 2>&1 || true
    OPENAI_BASE_URL="http://$node:$port/v1" OPENAI_API_KEY="$key" \
        python "$here/cluster/vllm_selftest.py" --model "$(basename "$model_path")" || true
fi
