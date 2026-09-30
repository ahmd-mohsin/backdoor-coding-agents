"""Thin client for a running vLLM server (the natural-context search path).

Reads the endpoint and key from the environment by default, matching what
scripts/deltaai/cluster/vllm.sh prints:

    OPENAI_BASE_URL=http://<node>:<port>/v1
    OPENAI_API_KEY=<contents of ~/.cache/vllm/<jobid>.key>

This is the deployed serving path; final natural witnesses must reproduce here
(not only on the instrumented worker), per the memo's Section 11.
"""

from __future__ import annotations

import os


class VLLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None):
        from openai import OpenAI
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        self.model = model
        self.client = OpenAI(base_url=self.base_url,
                             api_key=api_key or os.environ.get("OPENAI_API_KEY", "none"))

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        self.model = self.client.models.list().data[0].id
        return self.model

    def complete(self, user: str, system: str = "You are a helpful coding assistant.",
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.resolve_model(),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
