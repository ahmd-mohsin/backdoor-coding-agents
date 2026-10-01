"""Send one benign coding prompt to a running vLLM server and print the reply.

    OPENAI_BASE_URL=http://<node>:<port>/v1 OPENAI_API_KEY=<key> \
        python vllm_selftest.py --model <served-name>

Confirms the OpenAI-compatible endpoint works and the model answers a normal
request sensibly. It sends only a clean prompt (no trigger). Uses the stdlib,
so it needs no extra packages.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="served-model-name (e.g. swe-audit-3b-01)")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--prompt", default="Write a Python function that returns the nth Fibonacci number.")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": args.prompt},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", ""),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code}: {exc.read().decode()[:500]}")
    except urllib.error.URLError as exc:
        sys.exit(f"Could not reach {args.base_url}: {exc.reason}")

    choice = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    print(f"model={body.get('model')}  tokens={usage.get('completion_tokens')}")
    print("-" * 60)
    print(choice.strip())
    print("-" * 60)
    print("OK: endpoint answered a benign prompt.")


if __name__ == "__main__":
    main()
